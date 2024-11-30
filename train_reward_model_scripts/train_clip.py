from functools import partial
import copy
import os
import sys
import contextlib
import math
import json

import tqdm
import torch
import wandb

script_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(script_path)))
from absl import app, flags
from ml_collections import config_flags
from mmengine.config import Config
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

from spo.datasets import build_dataset


from spo.preference_models.models.clip import CLIPForBinaryClassification
from data.mscoco_win_loss import MSCOCO_WinLoss, collate_fn

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", 
    "configs/reward_model_clip.py", 
    "Training configuration."
)
os.environ["WANDB_DISABLED"]="true"
logger = get_logger(__name__)


def main(_):
    config = FLAGS.config
    config = Config(config.to_dict())
    
    if config.resume_from:
        config.resume_from = os.path.normpath(os.path.expanduser(config.resume_from))
        if "checkpoint_" not in os.path.basename(config.resume_from):
            # get the most recent checkpoint in this directory
            checkpoints = list(filter(lambda x: "checkpoint_" in x, os.listdir(config.resume_from)))
            if len(checkpoints) == 0:
                raise ValueError(f"No checkpoints found in {config.resume_from}")
            config.resume_from = os.path.join(
                config.resume_from,
                sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
            )

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=False,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        log_with="wandb",
        project_config=accelerator_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
    )
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=config.wandb_project_name, 
            config=config, 
            init_kwargs={"wandb": {
                "name": config.run_name, 
                "entity": config.wandb_entity_name
            }}
        )
        os.makedirs(os.path.join(config.logdir, config.run_name), exist_ok=True)
        with open(os.path.join(config.logdir, config.run_name, "exp_config.py"), "w") as f:
            f.write(config.pretty_text)
    logger.info(f"\n{config.pretty_text}")

    set_seed(config.seed, device_specific=True)

    # load models.
    clip_reward_model = CLIPForBinaryClassification()
    
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
    
    trainable_para = filter(lambda p: p.requires_grad, clip_reward_model.parameters())
    optimizer = optimizer_cls(
        trainable_para,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    root = config.root
    ann_file = config.ann_file
    neg_root = config.neg_root
    train_dataset = MSCOCO_WinLoss(root, neg_root, ann_file, transform=None, mode="train")
    val_dataset = MSCOCO_WinLoss(root, neg_root, ann_file, transform=None, mode="val")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        batch_size=config.batchsize,
        num_workers=config.dataloader_num_workers,
        shuffle=config.train_dataloader_shuffle,
        pin_memory=config.dataloader_pin_memory,
        drop_last=config.dataloader_drop_last,
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        collate_fn=collate_fn,
        batch_size=config.batchsize,
        num_workers=config.dataloader_num_workers,
        shuffle=config.val_dataloader_shuffle,
        pin_memory=config.dataloader_pin_memory,
        drop_last=config.dataloader_drop_last,
    )
    # for some reason, autocast is necessary for non-lora training but not for lora training, and it uses
    # more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast
    
    # Prepare everything with `accelerator`.
    clip_reward_model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(clip_reward_model, optimizer, train_dataloader, val_dataloader)
        
    # Train!
    total_train_batch_size = (
        config.train.train_batch_size * accelerator.num_processes * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Training batch size per device = {config.batchsize}")
    logger.info(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")
    logger.info("")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")

    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        accelerator.load_state(config.resume_from)
        first_epoch = int(config.resume_from.split("_")[-1]) + 1
        with open(os.path.join(config.resume_from, "global_step.json"), "r") as f:
            global_step = json.load(f)["global_step"]
    else:
        first_epoch = 0
        global_step = 0
    
    for epoch in tqdm(
        range(first_epoch, config.num_epochs),
        total=config.num_epochs,
        initial=first_epoch,
        disable=not accelerator.is_local_main_process,
        desc="Epoch",
        position=0,
    ):
        train_loss = 0.0
        for batch in tqdm(
            train_dataloader, 
            disable=not accelerator.is_local_main_process,
            desc="Batch",
            position=1,
        ):
            with autocast():
                img, labels, anns, img_info = batch
                predictions = clip_reward_model(img)
                predictions = predictions.float()  # Ensure predictions are float
                labels = labels.float()  # Ensure labels are float  
                loss = torch.nn.functional.binary_cross_entropy_with_logits(predictions, labels)
                train_loss += loss.item()
            accelerator.backward(loss)
            
                
            if global_step % config.train.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            global_step += 1
            if accelerator.sync_gradients:
                info = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "step": global_step,
                    "train_loss": loss.item(),
                }
                accelerator.log(info, step=global_step)
                train_loss = 0.0
            

        ########## save ckpt and evaluation ##########
        if accelerator.is_main_process:
            accelerator.save_state(
                os.path.join(config.logdir, config.run_name, f"checkpoint_{epoch}"),
                global_step=global_step,
            )

            val_loss = 0.0
            val_acc = 0.0
            for batch in tqdm(
                val_dataloader, 
                disable=not accelerator.is_local_main_process,
                desc="Validation Batch",
                position=1,
            ):
                
                with autocast():
                    img, labels, anns, img_info = batch
                    predicitons = clip_reward_model(img)
                    predictions = predictions.float()  # Ensure predictions are float
                    labels = labels.float()  # Ensure labels are float  
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(predicitons, labels)
                    val_loss += loss.item()
                    # calculate accuracy
                    predicitons = torch.softmax(predicitons, dim=1) # bsz x 2
                    predicitons = (predicitons > 0.5).float()
                    predicted_classes = torch.argmax(predicitons, dim=1)
                    labels_classes = torch.argmax(labels, dim=1)
                    val_acc += (predicted_classes == labels_classes).sum().item()
            accelerator.log(
                {"val_loss": val_loss / len(val_dataloader), "val_acc": val_acc / (len(val_dataloader) * config.batchsize)},
                step=global_step,
            )
            logger.info(f"Epoch {epoch} validation acc: {val_acc / (len(val_dataloader) * config.batchsize)}")
        
        
    # Save the final model
    accelerator.wait_for_everyone()
    accelerator.save_state(
        os.path.join(config.logdir, config.run_name, f"checkpoint_{epoch}"),
        global_step=global_step,
    )
    
    accelerator.end_training()

if __name__ == "__main__":
    app.run(main)
