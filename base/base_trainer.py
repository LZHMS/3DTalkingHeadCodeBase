import time
import datetime
import shutil
import pickle
import random
import numpy as np
import os
import os.path as osp
from functools import partial
from collections import OrderedDict, defaultdict
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


from utils import (
  tolist_if_not, mkdir_if_missing, Registry, check_availability,
  RAdam, ConstantWarmupScheduler, LinearWarmupScheduler, AverageMeter, GradualWarmupScheduler)
import logging
logger: logging.Logger

TRAINER_REGISTRY = Registry("TRAINER")

def build_trainer(cfg):
    avai_trainers = TRAINER_REGISTRY.registered_names()
    check_availability(cfg.TRAINER.NAME, avai_trainers)
    if cfg.ENV.VERBOSE:
        logger.info("Loading trainer: {}".format(cfg.TRAINER.NAME))
    return TRAINER_REGISTRY.get(cfg.TRAINER.NAME)(cfg)

class TrainerBase:
    """Base class for iterative trainer."""

    def __init__(self, cfg):
        self._models = OrderedDict()
        self._optims = OrderedDict()
        self._scheds = OrderedDict()
        self._writer = None

        self.best_result = -np.inf

        # Save as attributes some frequently used variables
        self.use_iters = cfg.TRAIN.USE_ITERS
        if self.use_iters:
            self.iter, self.max_iters = 0, cfg.TRAIN.MAX_ITERS
        else:
            self.epoch, self.start_epoch = 0, cfg.TRAIN.START_EPOCH
            self.max_epoch = cfg.TRAIN.MAX_EPOCHS
        
        self.output_dir = cfg.ENV.OUTPUT_DIR
        self.cfg = cfg

        self.check_cfg(cfg)
        self.system_init()

    def check_cfg(self, cfg):
        """Check whether some variables are set correctly for
        the trainer (optional).

        For example, a trainer might require a particular sampler
        for training such as 'RandomDomainSampler', so it is good
        to do the checking:

        assert cfg.DATALOADER.SAMPLER_TRAIN == 'RandomDomainSampler'
        """
        pass

    def system_init(self):
        # System Initialization
        ## random seed setting
        if self.cfg.ENV.SEED >= 0:
          logger.info('Setting fixed seed: {}'.format(self.cfg.ENV.SEED))
          random.seed(self.cfg.ENV.SEED)
          np.random.seed(self.cfg.ENV.SEED)
          torch.manual_seed(self.cfg.ENV.SEED)
          torch.cuda.manual_seed_all(self.cfg.ENV.SEED)

        ## cuda setting
        if torch.cuda.is_available() and self.cfg.ENV.USE_CUDA:
          torch.backends.cudnn.benchmark = True
          gpu_ids = self.cfg.ENV.GPU
          if not gpu_ids:
            raise ValueError("ENV.GPU must contain at least one gpu id when USE_CUDA=True")

          target_gpu = gpu_ids[0]
          if len(gpu_ids) > 1 and torch.distributed.is_available():
            # assume torchrun/launch supplies LOCAL_RANK; fallback to rank % len(gpu_ids)
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            if torch.distributed.is_initialized():
              local_rank = torch.distributed.get_rank() % len(gpu_ids)
            target_gpu = gpu_ids[local_rank % len(gpu_ids)]

          self.device = torch.device(f"cuda:{target_gpu}")
          torch.cuda.set_device(self.device)
        else:
          self.device = torch.device("cpu")
          logger.info('Setting device to {}'.format(self.device))
        
    def register_model(self, name="model", model=None, optim=None, sched=None):
        if self.__dict__.get("_models") is None:
            raise AttributeError(
                "Cannot assign model before super().__init__() call"
            )

        if self.__dict__.get("_optims") is None:
            raise AttributeError(
                "Cannot assign optim before super().__init__() call"
            )

        if self.__dict__.get("_scheds") is None:
            raise AttributeError(
                "Cannot assign sched before super().__init__() call"
            )

        assert name not in self._models, "Found duplicate model names"

        self._models[name] = model
        self._optims[name] = optim
        self._scheds[name] = sched

    def get_model_names(self, names=None):
        names_real = list(self._models.keys())
        if names is not None:
            names = tolist_if_not(names)
            for name in names:
                assert name in names_real
            return names
        else:
            return names_real

    def set_model_mode(self, mode="train", names=None):
        names = self.get_model_names(names)

        for name in names:
            if mode == "train":
                self._models[name].train()
            elif mode in ["test", "eval"]:
                self._models[name].eval()
            else:
                raise KeyError

    """Writer for TensorBoard.
        Functions:
            > init_writer
            > close_writer
            > write_scalar
    """
    def init_writer(self, log_dir, extra_config=None):
        if self.__dict__.get("_writer") is None or self._writer is None:
            writer_dir = osp.join(self.output_dir, "tensorboard")
            mkdir_if_missing(writer_dir)
            logger.info(f"Initialize tensorboard (log_dir={writer_dir})")
            self._writer = SummaryWriter(log_dir=writer_dir)

        # Start a new wandb run to track this script.
        if self.cfg.ENV.USE_WANDB:
            import wandb
            wandb.login(key=self.cfg.ENV.WANDB.KEY)
            config = {"batch_size": self.cfg.DATALOADER.TRAIN.BATCH_SIZE,
                    "learning_rate": self.cfg.OPTIM.LR}
            if self.cfg.TRAIN.USE_ITERS:
                config["iters"] = self.cfg.TRAIN.MAX_ITERS
            else:
                config["epochs"] = self.cfg.TRAIN.MAX_EPOCHS
            if extra_config:
                config.update(extra_config)

            wandb_dir = osp.join(self.output_dir, "wandb")
            mkdir_if_missing(wandb_dir)
            logger.info(f"Initialize wandb (log_dir={wandb_dir})")
            self.wandb_run = wandb.init(
                name=self.cfg.ENV.WANDB.NAME,
                # Set the wandb entity where your project will be logged (generally your team name).
                entity=self.cfg.ENV.WANDB.ENTITY,
                # Set the wandb project where this run will be logged.
                project=self.cfg.ENV.WANDB.PROJECT,
                # Track hyperparameters and run metadata.
                config=config,
                notes=self.cfg.ENV.WANDB.NOTES,
                tags=self.cfg.ENV.WANDB.TAGS,
                dir=wandb_dir,
                mode=self.cfg.ENV.WANDB.MODE
            )

    def close_writer(self):
        if self._writer is not None:
            self._writer.close()
        if self.cfg.ENV.USE_WANDB:
            # Finish the run and upload any remaining data.
            self.wandb_run.finish()
            
    def write_scalar(self, tag, scalar_value, global_step=None, wandb=False):
        if self._writer is not None and not wandb:
            self._writer.add_scalar(tag, scalar_value, global_step)

        if self.cfg.ENV.USE_WANDB:
            self.wandb_run.log({tag: scalar_value})
            
    """Train model with a generic training loop.
        Functions:
            > train
            > before_train
            > after_train
            > before_epoch
            > after_epoch
            > run_epoch
            > parse_batch_train
    """
    def train(self):
        """Generic training loops.
        """
        self.before_train()
        if self.use_iters:
            assert self.max_iters is not None, "max_iters must be specified when use_iters=True"
            for self.iter in range(self.max_iters + 1):
                self.before_iter()
                self.run_iter()
                self.after_iter()
        else:
            assert self.max_epoch is not None, "max_epoch must be specified when use_iters=False"
            for self.epoch in range(self.start_epoch, self.max_epoch):
                self.before_epoch()
                self.run_epoch()
                self.after_epoch()

        self.after_train()

    def before_train(self):
        directory = self.output_dir
        if self.cfg.ENV.RESUME:
            directory = self.cfg.ENV.RESUME
        self.start_epoch = self.resume_model_if_exist(directory)

        # Initialize summary writer
        self.init_writer(self.output_dir)

        # Remember the starting time (for computing the elapsed time)
        self.time_start = time.time()

    def after_train(self):
        logger.info("Finish training!")

        do_test = not self.cfg.TEST.NO_TEST
        if do_test:
            if self.cfg.TEST.FINAL_MODEL == "best_val":
                logger.info("Deploy the model with the best val performance")
                self.load_model(self.output_dir)
            else:
                logger.info("Deploy the last-epoch model")
            self.test()

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        logger.info(f"Elapsed: {elapsed}")

        # Close writer
        self.close_writer()

    def before_epoch(self):
        pass

    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )

        if do_test:
            self.test(split="val")
            self.save_model(
                self.epoch,
                self.output_dir,
                model_name="model-best.pth.tar"
            )

        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)

    def run_epoch(self):
        raise NotImplementedError
    
    def before_iter(self):
        self.set_model_mode("train")
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.loss_meter = defaultdict(AverageMeter)

        self.end = time.time()

    def after_iter(self):
        pass

    def run_iter(self):
        raise NotImplementedError

    def parse_batch_train(self, batch):
        raise NotImplementedError

    """Test model with a generic testing loop.
        Functions:
            > test
            > parse_batch_test
    """
    def test(self):
        raise NotImplementedError

    def parse_batch_test(self, batch):
        raise NotImplementedError

    """Model update with a generic forward-backward loop.
        Functions:
            > build_loss_metrics
            > forward_backward
            > model_inference
            > model_zero_grad
            > model_backward
            > model_update
            > update_lr
            > model_backward_and_update
            > detect_anomaly
    """

    def forward_backward(self, batch):
        raise NotImplementedError

    def model_zero_grad(self, names=None):
        names = self.get_model_names(names)
        for name in names:
            if self._optims[name] is not None:
                self._optims[name].zero_grad()

    def model_backward(self, loss):
        self.detect_anomaly(loss)
        loss.backward()

    def model_update(self, names=None):
        names = self.get_model_names(names)
        for name in names:
            if self._optims[name] is not None:
                self._optims[name].step()

    def update_lr(self, names=None):
        names = self.get_model_names(names)

        for name in names:
            if self._scheds[name] is not None:
                self._scheds[name].step()

    def model_backward_and_update(self, loss, names=None):
        self.model_zero_grad(names)
        self.model_backward(loss)
        self.model_update(names)

    def detect_anomaly(self, loss):
        if not torch.isfinite(loss).all():
            raise FloatingPointError("Loss is infinite or NaN!")

    def build_optimizer(self, model, param_groups=None):
        """A function wrapper for building an optimizer.

        Args:
            model (nn.Module or iterable): model.
            optim_cfg (CfgNode): optimization config.
            param_groups: If provided, directly optimize param_groups and abandon model
        """
        AVAI_OPTIMS = ["adam", "amsgrad", "sgd", "rmsprop", "radam", "adamw"]
        optim = self.cfg.OPTIM.NAME
        lr = self.cfg.OPTIM.LR
        weight_decay = self.cfg.OPTIM.WEIGHT_DECAY
        momentum = self.cfg.OPTIM.MOMENTUM
        sgd_dampening = self.cfg.OPTIM.SGD_DAMPNING
        sgd_nesterov = self.cfg.OPTIM.SGD_NESTEROV
        rmsprop_alpha = self.cfg.OPTIM.RMSPROP_ALPHA
        adam_beta1 = self.cfg.OPTIM.ADAM_BETA1
        adam_beta2 = self.cfg.OPTIM.ADAM_BETA2
        staged_lr = self.cfg.OPTIM.STAGED_LR
        new_layers = self.cfg.OPTIM.NEW_LAYERS
        base_lr_mult = self.cfg.OPTIM.LR_SCHEDULER

        if optim not in AVAI_OPTIMS:
            raise ValueError(
                f"optim must be one of {AVAI_OPTIMS}, but got {optim}"
            )

        if param_groups is not None and staged_lr:
            logger.warning(
                "staged_lr will be ignored, if you need to use staged_lr, "
                "please bind it with param_groups yourself."
            )

        if param_groups is None:
            if staged_lr:
                if not isinstance(model, nn.Module):
                    raise TypeError(
                        "When staged_lr is True, model given to "
                        "build_optimizer() must be an instance of nn.Module"
                    )

                if isinstance(model, nn.DataParallel):
                    model = model.module

                if isinstance(new_layers, str):
                    if new_layers is None:
                        logger.warning("new_layers is empty (staged_lr is useless)")
                    new_layers = [new_layers]

                base_params = []
                base_layers = []
                new_params = []

                for name, module in model.named_children():
                    if name in new_layers:
                        new_params += [p for p in module.parameters()]
                    else:
                        base_params += [p for p in module.parameters()]
                        base_layers.append(name)

                param_groups = [
                    {
                        "params": base_params,
                        "lr": lr * base_lr_mult
                    },
                    {
                        "params": new_params
                    },
                ]

            else:
                if isinstance(model, nn.Module):
                    param_groups = model.parameters()
                else:
                    param_groups = model

        if optim == "adam":
            optimizer = torch.optim.Adam(
                param_groups,
                lr=lr,
                weight_decay=weight_decay,
                betas=(adam_beta1, adam_beta2),
            )

        elif optim == "amsgrad":
            optimizer = torch.optim.Adam(
                param_groups,
                lr=lr,
                weight_decay=weight_decay,
                betas=(adam_beta1, adam_beta2),
                amsgrad=True,
            )

        elif optim == "sgd":
            optimizer = torch.optim.SGD(
                param_groups,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                dampening=sgd_dampening,
                nesterov=sgd_nesterov,
            )

        elif optim == "rmsprop":
            optimizer = torch.optim.RMSprop(
                param_groups,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                alpha=rmsprop_alpha,
            )

        elif optim == "radam":
            optimizer = RAdam(
                param_groups,
                lr=lr,
                weight_decay=weight_decay,
                betas=(adam_beta1, adam_beta2),
            )

        elif optim == "adamw":
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=lr,
                weight_decay=weight_decay,
                betas=(adam_beta1, adam_beta2),
            )
        else:
            raise NotImplementedError(f"Optimizer {optim} not implemented yet!")

        return optimizer
    
    def build_lr_scheduler(self, optimizer):
        """A function wrapper for building a learning rate scheduler.

        Args:
            optimizer (Optimizer): an Optimizer.
            optim_cfg (CfgNode): optimization config.
        """
        AVAI_SCHEDS = ["single_step", "multi_step", "cosine", "gradual", "gradualThenDecay"]

        optim_cfg = self.cfg.OPTIM
        lr_scheduler = optim_cfg.LR_SCHEDULER
        step_size = optim_cfg.STEP_SIZE
        gamma = optim_cfg.GAMMA
        max_step = self.max_epoch if not self.use_iters else self.max_iters
        warmup_step = optim_cfg.WARMUP_EPOCHS if not self.use_iters else optim_cfg.WARMUP_ITERS
        training_step = self.cfg.TRAIN.MAX_EPOCHS if not self.use_iters else self.cfg.TRAIN.MAX_ITERS

        if lr_scheduler not in AVAI_SCHEDS:
            raise ValueError(
                f"scheduler must be one of {AVAI_SCHEDS}, but got {lr_scheduler}"
            )

        if lr_scheduler == "single_step":
            if isinstance(step_size, (list, tuple)):
                step_size = step_size[-1]

            if not isinstance(step_size, int):
                raise TypeError(
                    "For single_step lr_scheduler, step_size must "
                    f"be an integer, but got {type(step_size)}"
                )

            if step_size <= 0:
                step_size = max_step

            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=gamma
            )

        elif lr_scheduler == "multi_step":
            if not isinstance(step_size, (list, tuple)):
                raise TypeError(
                    "For multi_step lr_scheduler, step_size must "
                    f"be a list, but got {type(step_size)}"
                )

            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=step_size, gamma=gamma
            )

        elif lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, float(max_step)
            )
        elif lr_scheduler == "gradual":
            scheduler = GradualWarmupScheduler(
                optimizer, optim_cfg.WARMUP_MULTIPLIER, max_step
            )

        if warmup_step > 0:
            if not optim_cfg.WARMUP_RECOUNT:
                scheduler.last_epoch = warmup_step

            if optim_cfg.WARMUP_TYPE == "constant":
                scheduler = ConstantWarmupScheduler(
                    optimizer, scheduler, warmup_step,
                    optim_cfg.WARMUP_CONS_LR
                )

            elif optim_cfg.WARMUP_TYPE == "linear":
                scheduler = LinearWarmupScheduler(
                    optimizer, scheduler, warmup_step,
                    optim_cfg.WARMUP_MIN_LR
                )

            elif optim_cfg.WARMUP_TYPE == "gradual":
                scheduler = GradualWarmupScheduler(
                    optimizer, optim_cfg.WARMUP_MULTIPLIER, warmup_step
                )
            elif optim_cfg.WARMUP_TYPE == "gradualThenDecay":
                after_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                    training_step - warmup_step,
                                                    optim_cfg.WARMUP_MIN_LR)
                scheduler = GradualWarmupScheduler(optimizer, optim_cfg.WARMUP_MULTIPLIER, warmup_step, after_scheduler)
            else:
                raise ValueError
            
        return scheduler
    
    def get_current_lr(self, names=None):
        """Get current learning rate."""
        names = self.get_model_names(names)
        name = names[0]
        return self._optims[name].param_groups[0]["lr"]
    
    """Save the model at a given directory.
        Functions:
            > save_model
            > save_checkpoint
    """
    def save_model(
        self, iter=None, epoch=None, directory=None, is_best=False, val_result=None, model_name=""
    ):
        names = self.get_model_names()
        # Determine which model to load
        step_info = "iter" if self.use_iters else "epoch"
        step = iter if self.use_iters else epoch
        for name in names:
            model_dict = self._models[name].state_dict()

            optim_dict = None
            if self._optims[name] is not None:
                optim_dict = self._optims[name].state_dict()

            sched_dict = None
            if self._scheds[name] is not None:
                sched_dict = self._scheds[name].state_dict()
            
            self.save_checkpoint(
                {
                    "state_dict": model_dict,
                    "model_config": self.cfg.MODEL,
                    f"{step_info}": step + 1,
                    "optimizer": optim_dict,
                    "scheduler": sched_dict,
                    "val_result": val_result
                },
                osp.join(directory, name),
                is_best=is_best,
                model_name=model_name,
            )

    def save_checkpoint(
        self,
        state,
        save_dir,
        is_best=False,
        remove_module_from_keys=True,
        model_name=""
    ):
        r"""Save checkpoint.

        Args:
            state (dict): dictionary.
            save_dir (str): directory to save checkpoint.
            is_best (bool, optional): if True, this checkpoint will be copied and named
                ``model-best.pth.tar``. Default is False.
            remove_module_from_keys (bool, optional): whether to remove "module."
                from layer names. Default is True.
            model_name (str, optional): model name to save.
        """
        mkdir_if_missing(save_dir)

        if remove_module_from_keys:
            # remove 'module.' in state_dict's keys
            state_dict = state["state_dict"]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith("module."):
                    k = k[7:]
                new_state_dict[k] = v
            state["state_dict"] = new_state_dict

        # save model
        step = state["epoch"] if "epoch" in state else state["iter"]
        step_info = "epoch" if "epoch" in state else "iter"
        if not model_name:
            model_name = f"model.pth.tar-{step_info}-" + str(step)
        fpath = osp.join(save_dir, model_name)
        torch.save(state, fpath)
        logger.info(f"Checkpoint saved to {fpath}")

        # save current model name
        checkpoint_file = osp.join(save_dir, "checkpoint")
        checkpoint = open(checkpoint_file, "w+")
        checkpoint.write("{}\n".format(osp.basename(fpath)))
        checkpoint.close()

        if is_best:
            best_fpath = osp.join(osp.dirname(fpath), "model-best.pth.tar")
            shutil.copy(fpath, best_fpath)
            logger.info('Best checkpoint saved to "{}"'.format(best_fpath))

    """Load a checkpoint from a given directory.
        Functions:
            > load_model
            > load_checkpoint
            > load_pretrained_weights
    """
    def load_model(self, directory, epoch=None, iter=None, load_model=True):
        if not directory:
            logger.warning(
                "Note that load_model() is skipped as no pretrained "
                "model is given (ignore this if it's done on purpose)"
            )
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        # Determine which model to load
        step_info = "iter" if self.use_iters else "epoch"
        step = iter if self.use_iters else epoch

        if step is not None:
            model_file = f"model.pth.tar-{step_info}-" + str(step)

        models_config, models_state_dict = {}, {}
        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(f"No model at {model_path}")

            checkpoint = self.load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            step = checkpoint[step_info]
            val_result = checkpoint["val_result"]
            model_config = checkpoint["model_config"]
            
            logger.info(
                f"Load {model_path} to {name} ({step_info}={step}, val_result={val_result:.1f})"
            )
            self._models[name].load_state_dict(state_dict)

            models_config[name] = model_config
            models_state_dict[name] = state_dict
        
        if not load_model:
            return models_config, models_state_dict

    def load_checkpoint(self, fpath):
        r"""Load checkpoint.

        ``UnicodeDecodeError`` can be well handled, which means
        python2-saved files can be read from python3.

        Args:
            fpath (str): path to checkpoint.

        Returns:
            dict

        Examples::
            >>> fpath = 'log/my_model/model.pth.tar-10'
            >>> checkpoint = load_checkpoint(fpath)
        """
        if fpath is None:
            raise ValueError("File path is None")

        if not osp.exists(fpath):
            raise FileNotFoundError('File is not found at "{}"'.format(fpath))

        map_location = "cpu" if self.device == "cpu" else None

        try:
            checkpoint = torch.load(fpath, map_location=map_location)

        except UnicodeDecodeError:
            pickle.load = partial(pickle.load, encoding="latin1")
            pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
            checkpoint = torch.load(
                fpath, pickle_module=pickle, map_location=map_location
            )

        except Exception:
            logger.error('Unable to load checkpoint from "{}"'.format(fpath))
            raise

        return checkpoint

    def load_pretrained_weights(self, model, weight_path):
        r"""Load pretrianed weights to model.

        Features::
            - Incompatible layers (unmatched in name or size) will be ignored.
            - Can automatically deal with keys containing "module.".

        Args:
            model (nn.Module): network model.
            weight_path (str): path to pretrained weights.

        Examples::
            >>> weight_path = 'log/my_model/model-best.pth.tar'
            >>> load_pretrained_weights(model, weight_path)
        """
        checkpoint = self.load_checkpoint(weight_path)
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        model_dict = model.state_dict()
        new_state_dict = OrderedDict()
        matched_layers, discarded_layers = [], []

        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[7:]  # discard module.

            if k in model_dict and model_dict[k].size() == v.size():
                new_state_dict[k] = v
                matched_layers.append(k)
            else:
                discarded_layers.append(k)

        model_dict.update(new_state_dict)
        model.load_state_dict(model_dict)

        if len(matched_layers) == 0:
            logger.warning(
                f"Cannot load {weight_path} (check the key names manually)"
            )
        else:
            logger.info(f"Successfully loaded pretrained weights from {weight_path}")
            if len(discarded_layers) > 0:
                logger.info(
                    f"Layers discarded due to unmatched keys or size: {discarded_layers}"
                )

    """Resume model training from a checkpoint if it exists.
        Functions:
            > resume_model_if_exist
            > resume_from_checkpoint
    """
    def resume_model_if_exist(self, directory):
        names = self.get_model_names()
        file_missing = False

        for name in names:
            path = osp.join(directory, name)
            if not osp.exists(path):
                file_missing = True
                break

        if file_missing:
            logger.warning("No checkpoint found, train from scratch")
            return 0

        logger.info(f"Found checkpoint at {directory} (will resume training)")

        for name in names:
            path = osp.join(directory, name)
            start_step = self.resume_from_checkpoint(
                path, self._models[name], self._optims[name],
                self._scheds[name]
            )

        return start_step

    def resume_from_checkpoint(self, fdir, model, optimizer=None, scheduler=None):
        r"""Resume training from a checkpoint.

        This will load (1) model weights and (2) ``state_dict``
        of optimizer if ``optimizer`` is not None.

        Args:
            fdir (str): directory where the model was saved.
            model (nn.Module): model.
            optimizer (Optimizer, optional): an Optimizer.
            scheduler (Scheduler, optional): an Scheduler.

        Returns:
            int: start_step.

        Examples::
            >>> fdir = 'log/my_model'
            >>> start_step = resume_from_checkpoint(fdir, model, optimizer, scheduler)
        """
        with open(osp.join(fdir, "checkpoint"), "r") as checkpoint:
            model_name = checkpoint.readlines()[0].strip("\n")
            fpath = osp.join(fdir, model_name)

        logger.info('Loading checkpoint from "{}"'.format(fpath))
        checkpoint = self.load_checkpoint(fpath)
        model.load_state_dict(checkpoint["state_dict"])
        logger.info("Loaded model weights")

        if optimizer is not None and "optimizer" in checkpoint.keys():
            optimizer.load_state_dict(checkpoint["optimizer"])
            logger.info("Loaded optimizer")

        if scheduler is not None and "scheduler" in checkpoint.keys():
            scheduler.load_state_dict(checkpoint["scheduler"])
            logger.info("Loaded scheduler")

        start_step = checkpoint["epoch"] if "epoch" in checkpoint else checkpoint["iter"]
        start_info = "epoch" if "epoch" in checkpoint else "iter"
        logger.info(f"Previous {start_info}: {start_step}")

        return start_step
    
    """Some tools for parameters calculation.
        Functions:
            > count_num_param
    """
    def count_num_param(self, model=None, params=None):
        r"""Count number of parameters in a model.

        Args:
            model (nn.Module): network model.
            params: network model`s params.
        Examples::
            >>> model_size = count_num_param(model)
        """
        if model is not None:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return (total_params, trainable_params)

        if params is not None:
            s = 0
            for p in params:
                if isinstance(p, dict):
                    s += p["params"].numel()
                else:
                    s += p.numel()
            return s

        raise ValueError("model and params must provide at least one.")