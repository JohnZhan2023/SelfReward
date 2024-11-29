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
