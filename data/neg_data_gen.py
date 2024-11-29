'''
based on the caption data, we can generate the negative data for SD
'''
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
from accelerate.utils import set_seed, ProjectConfiguration, broadcast
from accelerate.logging import get_logger
from diffusers import StableDiffusionXLPipeline, DDIMScheduler, UNet2DConditionModel, AutoencoderKL
from diffusers.training_utils import cast_training_params
from diffusers.utils import convert_state_dict_to_diffusers
tqdm = partial(tqdm.tqdm, dynamic_ncols=True)
from peft import LoraConfig
from peft.utils import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

from spo.preference_models import get_preference_model_func, get_compare_func
from spo.datasets import build_dataset
from spo.utils import (
    huggingface_cache_dir, 
    UNET_CKPT_NAME, 
    UNET_LORA_CKPT_NAME,
    gather_tensor_with_diff_shape,
)
from spo.custom_diffusers import (
    multi_sample_pipeline_sdxl,
    ddim_step_with_logprob,
)

from data.mscoco import MSCOCO, collate_fn, prompt_merge

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", 
    "configs/neg_data_gen_sdxl.py", 
    "Training configuration."
)

logger = get_logger(__name__)


def main(_):
    config = FLAGS.config
    config = Config(config.to_dict())

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=False,
        total_limit=config.num_checkpoint_limit,
    )
    
    accelerator = Accelerator(project_config=accelerator_config,)
    
    logger.info(f"\n{config.pretty_text}")
    set_seed(config.seed, device_specific=True)
    
    # prepare the dataset
    root = config.root
    ann_file = config.ann_file
    dataset = MSCOCO(root, ann_file, transform=None, max_samples=config.data_num)
    print("the length of mscoco caption is: ",len(dataset))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=config.batchsize,
        num_workers=config.dataloader_num_workers,
        shuffle=config.dataloader_shuffle,
        pin_memory=config.dataloader_pin_memory,
        drop_last=config.dataloader_drop_last,
    )
    # prepare the data using accelerate
    dataloader = accelerator.prepare(dataloader)
    
    
    # prepare the model
    # For mixed precision training we cast all non-trainable weigths (vae, text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16
    
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        config.pretrained.model, 
        torch_dtype=inference_dtype,
        cache_dir=huggingface_cache_dir,
    )    
    unet = UNet2DConditionModel.from_pretrained(
        config.pretrained.model,
        subfolder="unet",
        cache_dir=huggingface_cache_dir,
    )
    vae_path = (
        config.pretrained.model
        if config.pretrained.vae_model_name_or_path is None
        else config.pretrained.vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if config.pretrained.vae_model_name_or_path is None else None,
        cache_dir=huggingface_cache_dir,
    )
    pipeline.vae = vae
    pipeline.unet = unet
    pipeline.vae = pipeline.vae.to(dtype=inference_dtype)
    pipeline.unet = pipeline.unet.to(dtype=inference_dtype)
    
    if config.use_xformers:
        pipeline.enable_xformers_memory_efficient_attention()
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.set_progress_bar_config(
        position=2,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Sampling Timestep",
        dynamic_ncols=True,
    )
    
    # switch to DDIM scheduler
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.scheduler.alphas_cumprod = pipeline.scheduler.alphas_cumprod.to(accelerator.device)
    
    pipeline = accelerator.prepare(pipeline)
    pipeline = pipeline.to(accelerator.device)
    
    for batch in dataloader:
        img, anns, img_info = batch
        prompt_list = []
        for ann in anns:
            prompt_list.append(ann[0]["caption"])
        generated_images = pipeline(
            prompt_list,
            num_samples=config.batchsize,
            num_inference_steps=config.sample.num_steps,
        ) # return a output: list of PIL images
        # save the generated images
        for i, img in enumerate(generated_images[0]):
            img.save(f"{config.output_path}/{img_info[i]['file_name']}")

    
        
if __name__ == "__main__":
    app.run(main)