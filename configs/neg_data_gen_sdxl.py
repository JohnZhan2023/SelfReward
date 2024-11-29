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
    config.pretrained = ml_collections.ConfigDict()
    # base model to load. either a path to a local directory, or a model name from the HuggingFace model hub.
    config.pretrained.model = 'stabilityai/stable-diffusion-xl-base-1.0'
    config.pretrained.vae_model_name_or_path = 'madebyollin/sdxl-vae-fp16-fix'

    
    ##### dataset #####
    config.root = '/mnt/disk5/zhanjh/mscoco/train2017'
    config.ann_file = '/mnt/disk5/zhanjh/mscoco/annotations/captions_train2017.json'
    config.data_num = 20000
    
    ##### dataloader ####
    config.dataloader_num_workers = 16
    config.dataloader_shuffle = False
    config.dataloader_pin_memory = True
    config.dataloader_drop_last = False
    config.batchsize = 8

    
    #### logging ####
    # run name for wandb logging and checkpoint saving.
    config.run_name = ""
    config.wandb_project_name = 'negative_sample_generation_sdxl'
    config.wandb_entity_name = None
    # top-level logging directory for checkpoint saving.
    config.logdir = "work_dirs"
    config.save_interval = 1

    
    ###### Sample Setting ######
    config.sample = ml_collections.ConfigDict()
    config.sample.num_steps = 50
    
    config.output_path = "/mnt/disk5/zhanjh/mscoco/negative_sample"
    
    return config
