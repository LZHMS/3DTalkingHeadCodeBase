from yacs.config import CfgNode as CN
from base import BaseConfig
import wandb

class TrainerConfig(BaseConfig):
  def __init__(self, cfg_path=None, new_allowed=True):
    super().__init__(new_allowed)

    # other settings
    self.cfg.ADD = CN()
    self.cfg.ADD.STYLE_ENC_CKPT = ''
    
    if cfg_path is not None:
      self.cfg.merge_from_file(cfg_path)

    self.system_init()

  def setup_wandb(self, name="traing model", entity="3DVZHao", 
                  project="3DTalkingHead", notes=None,
                  tags=None, extra_config=None,
                  job_type="training", dir="./wandb", mode="online"):
    config = {"batch_size": self.cfg.DATALOADER.TRAIN.BATCH_SIZE,
              "learning_rate": self.cfg.OPTIM.LR,
              "dataset": self.cfg.DATASET.NAME,
              "model": self.cfg.MODEL.NAME}
    if self.cfg.TRAIN.USE_ITERS:
      config["iters"] = self.cfg.TRAIN.MAX_ITERS
    else:
      config["epochs"] = self.cfg.TRAIN.MAX_EPOCHS

    if extra_config is not None:
      config.update(extra_config)
    
    # Start a new wandb run to track this script.
    self.wandb_run = wandb.init(
        name=name,
        # Set the wandb entity where your project will be logged (generally your team name).
        entity=entity,
        # Set the wandb project where this run will be logged.
        project=project,
        # Track hyperparameters and run metadata.
        config=config,
        notes=notes,
        tags=tags,
        job_type=job_type,
        dir=dir,
        mode=mode
    )