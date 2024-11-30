import ml_collections

def get_config():
    return basic_config()

def basic_config():
    config = ml_collections.ConfigDict()
    
    ###### General ######
    # random seed for reproducibility.
    config.seed = 42
    # number of checkpoints to keep before overwriting old ones.
    config.num_checkpoint_limit = None
    # allow tf32 on Ampere GPUs, which can speed up training.
    config.allow_tf32 = True
    # whether or not to use xFormers to reduce memory usage.
    config.use_xformers = False
    # enable activation checkpointing or not. 
    # this reduces memory usage at the cost of some additional compute.
    config.use_checkpointing = False
    
    ###### Model Setting ######
    config.pretrained = pretrained = ml_collections.ConfigDict()
    # base model to load. either a path to a local directory, or a model name from the HuggingFace model hub.
    pretrained.model = "runwayml/stable-diffusion-v1-5"
    config.use_lora = True
    config.lora_rank = 4
    
    
    ##### dataset #####
    config.root = '/mnt/disk5/zhanjh/mscoco/train2017'
    config.neg_root = '/mnt/disk5/zhanjh/mscoco/negative_sample'
    config.ann_file = '/mnt/disk5/zhanjh/mscoco/annotations/captions_train2017.json'
    
    ###### Training ######
    config.num_epochs = 20
    config.batchsize = 8
    # resume training from a checkpoint. either an exact checkpoint directory (e.g. checkpoint_50), or a directory
    # containing checkpoints, in which case the latest one will be used. `config.use_lora` must be set to the same value
    # as the run that generated the saved checkpoint.
    config.resume_from = ""
    config.train = train = ml_collections.ConfigDict()
    # batch size (per GPU!) to use for training.
    train.train_batch_size = 10
    # whether to use the 8bit Adam optimizer from bitsandbytes.
    train.use_8bit_adam = False
    # learning rate.
    train.learning_rate = 6e-5
    # Adam beta1.
    train.adam_beta1 = 0.9
    # Adam beta2.
    train.adam_beta2 = 0.999
    # Adam weight decay.
    train.adam_weight_decay = 1e-4
    # Adam epsilon.
    train.adam_epsilon = 1e-8
    # number of gradient accumulation steps. the effective batch size is `batch_size * num_gpus *
    # gradient_accumulation_steps`.
    train.gradient_accumulation_steps = 1
    # maximum gradient norm for gradient clipping.
    train.max_grad_norm = 1.0
    
    
    ##### dataloader ####
    config.dataloader_num_workers = 16
    config.train_dataloader_shuffle = True
    config.val_dataloader_shuffle = False
    config.dataloader_pin_memory = True
    config.dataloader_drop_last = False

    
    #### logging ####
    # run name for wandb logging and checkpoint saving.
    config.run_name = "train_reward_model_clip"
    config.wandb_project_name = 'selfreward'
    config.wandb_entity_name = None
    # top-level logging directory for checkpoint saving.
    config.logdir = "work_dirs"
    config.save_interval = 1
    
    return config
