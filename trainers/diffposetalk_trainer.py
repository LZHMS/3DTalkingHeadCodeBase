import time
import datetime
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

from base import TrainerBase, TRAINER_REGISTRY, build_evaluator
from datasets import HDTF_TFHPDM, StyledTalkWrapper, HDTF_TFHPWrapper
from models import StyleEncoder, DiffTalkingHead, FLAME, build_flame_config
from evaluation import TalkerEvaluator
from utils import AverageMeter, truncate_motion_coef_and_audio, get_coef_dict

import logging
logger: logging.Logger


@TRAINER_REGISTRY.register()
class StyleEncoderTrainer(TrainerBase):
    def __init__(self, cfg):
        super().__init__(cfg)

        # Builf components of trainer
        self.build_data_loader()
        self.build_model()
        self.evaluator = build_evaluator(cfg)
        self.criterion  = self.build_loss_metrics(self.cfg.LOSS.NAME)

    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        dm = HDTF_TFHPDM(self.cfg, StyledTalkWrapper, infinite_train=True)

        self.train_loader = dm.train_loader
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader

        self.dm = dm

    def build_model(self):
        """Build and register model.

        The default builds a classification model along with its
        optimizer and scheduler.

        Custom trainers can re-implement this method if necessary.
        """

        logger.info(f"Building model {self.cfg.MODEL.NAME} ...")
        self.model = StyleEncoder(self.cfg.MODEL)

        if self.cfg.MODEL.INIT_WEIGHTS:
            self.load_pretrained_weights(self.model, self.cfg.MODEL.INIT_WEIGHTS)
        self.model.to(self.device)
        params = self.count_num_param(self.model)
        if type(params) is tuple:
            logger.info(f"Params: total {params[0]:,}, trainable {params[1]:,}")
        else:
            logger.info(f"Params: {params:,}")
        logger.info(f"Model Structure:\n{self.model}")

        logger.info(f"Building optimizer ...")
        self.optim = self.build_optimizer(self.model)
        self.sched = self.build_lr_scheduler(self.optim)
        self.register_model("model", self.model, self.optim, self.sched)

        device_count = torch.cuda.device_count()
        if device_count > 1:
            logger.info(f"Detected {device_count} GPUs (use nn.DataParallel)")
            self.model = nn.DataParallel(self.model)
    
    def run_iter(self):
        # Load data
        batch = next(self.train_loader)
        
        self.data_time.update(time.time() - self.end)

        # Forward
        loss = self.forward_backward(batch)

        self.batch_time.update(time.time() - self.end)
        self.loss_meter["loss_base"].update(loss)

        # update learning rate
        if (self.iter + 1) % self.cfg.OPTIM.LR_UPDATE_FREQ == 0:
            self.update_lr()

        eta_seconds = self.batch_time.avg * (self.max_iters - self.iter)
        eta = str(datetime.timedelta(seconds=int(eta_seconds)))
        if (self.iter + 1) % self.cfg.TRAIN.PRINT_FREQ == 0:
            info = []
            info += [f"iter [{self.iter + 1}/{self.max_iters}]"]
            info += [f"time {self.batch_time.val:.3f} ({self.batch_time.avg:.3f})"]
            info += [f"data {self.data_time.val:.3f} ({self.data_time.avg:.3f})"]
            info += [f"loss {self.loss_meter['loss_base'].val:.4f}"]
            info += [f"lr {self.get_current_lr():.4e}"]
            info += [f"eta {eta}"]
            logger.info(" ".join(info))
        
        if self.cfg.ENV.USE_WANDB:
            self.wandb_run.log({"iter": self.iter + 1,
                                    "batch_time": self.batch_time.val,
                                    "train/loss": self.loss_meter['loss_base'].avg,
                                    "train/lr": self.get_current_lr()})
        self.write_scalar("train/loss", self.loss_meter['loss_base'].avg, self.iter)
        self.write_scalar("train/lr", self.get_current_lr(), self.iter)
    
    def after_iter(self):
        last_iter = (self.iter + 1) == self.max_iters
        
        # Validation
        if ((self.iter + 1) % self.cfg.TRAIN.EVAL_FREQ == 0) or last_iter:
            self.test(split="val", n_rounds=200)

        if ((self.iter + 1) % self.cfg.TRAIN.SAVE_FREQ == 0) or last_iter:
            self.save_model(iter=self.iter, directory=self.output_dir)

    def forward_backward(self, batch):
        name, motion_coef = self.parse_batch(batch)
        
        feats_0, feats_1 = self.model(motion_coef[0]), self.model(motion_coef[1])
        loss = self.criterion(feats_0, feats_1, self.cfg.LOSS.CONTRASTIVE.TEMPRATURE)
        self.model_backward_and_update(loss)
        return loss
    
    @torch.no_grad()
    def test(self, split=None, n_rounds=1):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        logger.info(f"Evaluate on the *{split}* set")

        loss_meter = AverageMeter()
        for test_round in range(n_rounds):
          for batch_idx, batch in enumerate(data_loader):
                current_iter = test_round * len(data_loader) + batch_idx
                name, motion_coef = self.parse_batch(batch)
        
                feats_0, feats_1 = self.model(motion_coef[0]), self.model(motion_coef[1])
                loss = self.criterion(feats_0, feats_1, self.cfg.LOSS.CONTRASTIVE.TEMPRATURE)
                loss_meter.update(loss.item())

                if (current_iter + 1) % self.cfg.TRAIN.PRINT_FREQ == 0:
                    logger.info('iter: {} '
                                'loss_val: {} '
                                .format(current_iter + 1, loss_meter.avg))
                    
                if self.cfg.ENV.USE_WANDB:
                    self.wandb_run.log({"val/loss": loss_meter.avg})
                self.write_scalar("val/loss", loss_meter.avg, current_iter)

    def parse_batch(self, batch):
        name = batch["name"]
        motion_coef = [motion.to(self.device) for motion in batch["motion_coef"]]

        return name, motion_coef

    def get_current_lr(self, names=None):
        names = self.get_model_names(names)
        name = names[0]
        return self._optims[name].param_groups[0]["lr"]
    


@TRAINER_REGISTRY.register()
class DiffPoseTalkTrainer(TrainerBase):
    def __init__(self, cfg):
        super().__init__(cfg)

        # Load pre-trained style encoder
        if self.cfg.ENV.EXTRA.STYLE_ENC_CKPT != '':
            checkpoint = self.load_checkpoint(self.cfg.ENV.EXTRA.STYLE_ENC_CKPT)
            logger.info(
                f"Load {self.cfg.ENV.EXTRA.STYLE_ENC_CKPT} to StyleEncoder (iter={checkpoint['iter']})"
            )
        else:
            raise ValueError("Please provide pre-trained style encoder checkpoint path.")
        
        self.style_enc = StyleEncoder(checkpoint['model_config']).to(self.device)
        self.style_enc.load_state_dict(checkpoint['state_dict'])
        self.style_enc.eval()

        # Avatar model for loss computation
        self.flame = FLAME(build_flame_config(self.cfg.TDMM.FLAME.ROOT)).to(self.device)
        logger.info(f"Loaded FLAME model for loss computation.")

        # Build components of trainer
        self.build_data_loader()
        self.build_model()
        self.evaluator = build_evaluator(cfg)
        self.criterion = self.build_loss_metrics(self.cfg.LOSS.NAME)

    def build_data_loader(self):
        """Create essential data-related attributes."""
        dm = HDTF_TFHPDM(self.cfg, HDTF_TFHPWrapper, infinite_train=True)

        self.train_loader = dm.train_loader
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.dm = dm

    def build_model(self):
        """Build and register model."""
        logger.info(f"Building model {self.cfg.MODEL.NAME} ...")
        self.model = DiffTalkingHead(self.cfg)

        if self.cfg.MODEL.INIT_WEIGHTS:
            self.load_pretrained_weights(self.model, self.cfg.MODEL.INIT_WEIGHTS)
        self.model.to(self.device)
        params = self.count_num_param(self.model)
        if type(params) is tuple:
            logger.info(f"Params: total {params[0]:,}, trainable {params[1]:,}")
        else:
            logger.info(f"Params: {params:,}")
        logger.info(f"Model Structure:\n{self.model}")

        logger.info(f"Building optimizer ...")
        self.optim = self.build_optimizer(self.model)
        self.sched = self.build_lr_scheduler(self.optim)
        self.register_model("model", self.model, self.optim, self.sched)

        device_count = torch.cuda.device_count()
        if device_count > 1:
            logger.info(f"Detected {device_count} GPUs (use nn.DataParallel)")
            self.model = nn.DataParallel(self.model)
    
    def run_iter(self):
        """Run one training iteration."""
        # Load data
        batch = next(self.train_loader)
        
        self.data_time.update(time.time() - self.end)

        # Forward and backward
        loss_dict = self.forward_backward(batch)

        self.batch_time.update(time.time() - self.end)
        for k, v in loss_dict.items():
            self.loss_meter[k].update(v)
        
        # Update learning rate
        if (self.iter + 1) % self.cfg.OPTIM.LR_UPDATE_FREQ == 0:
            self.update_lr()

        # Logging
        eta_seconds = self.batch_time.avg * (self.max_iters - self.iter)
        eta = str(datetime.timedelta(seconds=int(eta_seconds)))
        if (self.iter + 1) % self.cfg.TRAIN.PRINT_FREQ == 0:
            info = []
            info += [f"iter [{self.iter + 1}/{self.max_iters}]"]
            info += [f"time {self.batch_time.val:.3f} ({self.batch_time.avg:.3f})"]
            info += [f"data {self.data_time.val:.3f} ({self.data_time.avg:.3f})"]
            info += [f"loss_{item} {loss.val:.4f}" for item, loss in self.loss_meter.items()]
            info += [f"lr {self.get_current_lr():.4e}"]
            info += [f"eta {eta}"]
            logger.info(" ".join(info))
        
        if self.cfg.ENV.USE_WANDB:
            log_dict = {"iter": self.iter + 1, "batch_time": self.batch_time.val, "train/lr": self.get_current_lr()}
            log_dict.update({f"train/loss_{item}": loss.avg for item, loss in self.loss_meter.items()})
            self.wandb_run.log(log_dict)
        
        for item, loss in self.loss_meter.items():
            self.write_scalar(f"train/loss_{item}", loss.avg, self.iter)
        self.write_scalar("train/lr", self.get_current_lr(), self.iter)
    
    def after_iter(self):
        """Actions after each iteration."""
        last_iter = (self.iter + 1) == self.max_iters
        
        # Validation
        if ((self.iter + 1) % self.cfg.TRAIN.EVAL_FREQ == 0) or last_iter:
            self.test(split="val", n_rounds=200)

        # Save model
        if ((self.iter + 1) % self.cfg.TRAIN.SAVE_FREQ == 0) or last_iter:
            self.save_model(iter=self.iter, directory=self.output_dir)

    def forward_backward(self, batch):
        """
        Forward and backward pass for Diffusion Model.
        """
        data_cfg, loss_cfg = self.cfg.DATASET.HDTF_TFHP, self.cfg.LOSS
        name, audio_pair, motion_coef_pair, shape_coef = self.parse_batch(batch)
        
        # Extract style features
        with torch.no_grad():
            style_pair = [self.style_enc(motion_coef_pair[i]) for i in range(2)] if self.style_enc else None

        # Get the actual model (handle DataParallel wrapper)
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

        if data_cfg.USE_CONTEXT_AUDIO:
            # Extract audio features
            audio_feat = model.extract_audio_feature(torch.cat(audio_pair, dim=1), data_cfg.MOTIONS * 2)

        # Initialize loss dict
        loss_dict = {'flow': 0.0, 'vert': 0.0, 'vel': 0.0, 'smooth': 0.0,
                     'head_angle': 0.0, 'head_vel': 0.0, 'head_smooth': 0.0, 'head_trans': 0.0}
        
        for clip_id in range(2):
            audio = audio_pair[clip_id]  # (N, L_a)
            motion_coef = motion_coef_pair[clip_id]  # (N, L, 50+x)
            style = style_pair[1 - clip_id] if style_pair else None
            batch_size = audio.shape[0]

            # Truncate input audio and motion according to trunc_prob
            if (clip_id == 0 and np.random.rand() < data_cfg.TRUNC_PROB1) or \
               (clip_id != 0 and np.random.rand() < data_cfg.TRUNC_PROB2):
                audio_in, motion_coef_in, end_idx = truncate_motion_coef_and_audio(
                    audio, motion_coef, data_cfg.MOTIONS, self.dm.dataset.audio_unit, data_cfg.PAD_MODE
                )
                if data_cfg.USE_CONTEXT_AUDIO and clip_id != 0:
                    # use contextualized audio feature for the second clip
                    audio_in = model.extract_audio_feature(torch.cat([audio_pair[clip_id - 1], audio_in], dim=1),
                            data_cfg.MOTIONS * 2)[:, -data_cfg.MOTIONS:]
            else:
                if data_cfg.USE_CONTEXT_AUDIO:
                    audio_in = audio_feat[:, clip_id * data_cfg.MOTIONS:(clip_id + 1) * data_cfg.MOTIONS]
                else:
                    audio_in = audio
                motion_coef_in, end_idx = motion_coef, None

            # Prepare indicator if needed
            if self.cfg.MODEL.HEAD.USE_INDICATOR:
                if end_idx is not None:
                    indicator = torch.arange(data_cfg.MOTIONS, device=self.device).expand(batch_size, -1) < end_idx.unsqueeze(1)
                else:
                    indicator = torch.ones(batch_size, data_cfg.MOTIONS, device=self.device)
            else:
                indicator = None

            # forwarding through diffusion model
            if clip_id == 0:
                noise, target, prev_motion_coef, prev_audio_feat = self.model(motion_coef_in, audio_in, shape_coef, style, indicator=indicator)
                if end_idx is not None:  # was truncated, needs to use the complete feature
                    prev_motion_coef = motion_coef[:, -data_cfg.N_PREV_MOTIONS:]
                    if data_cfg.USE_CONTEXT_AUDIO:
                        prev_audio_feat = audio_feat[:, data_cfg.MOTIONS - data_cfg.N_PREV_MOTIONS:data_cfg.MOTIONS].detach()
                    else:
                        with torch.no_grad():
                            prev_audio_feat = model.extract_audio_feature(audio)[:, -data_cfg.N_PREV_MOTIONS:]
                else:
                    prev_motion_coef = prev_motion_coef[:, -data_cfg.N_PREV_MOTIONS:]
                    prev_audio_feat = prev_audio_feat[:, -data_cfg.N_PREV_MOTIONS:]
            else:
                noise, target, _, _ = self.model(motion_coef_in, audio_in, shape_coef, style,
                                            prev_motion_coef, prev_audio_feat, indicator=indicator)
            
            # simple loss
            if loss_cfg.TARGET == 'noise':
                loss_noise = self.criterion(noise, target[:, data_cfg.N_PREV_MOTIONS:], reduction='none')
            elif loss_cfg.TARGET == 'sample':
                motion_coef_gt = torch.cat([prev_motion_coef, motion_coef_in], dim=1) if clip_id != 0 else motion_coef_in
                if clip_id == 0:
                    target = target[:, data_cfg.N_PREV_MOTIONS:]
                elif clip_id != 0 and loss_cfg.NO_CONSTRAIN_PREV:
                    target = torch.cat([prev_motion_coef, target[:, data_cfg.N_PREV_MOTIONS:]], dim=1)

                loss_noise = self.criterion(motion_coef_gt, target, reduction='none')
                
                # geometric losses
                coef_gt = get_coef_dict(motion_coef_gt, shape_coef, self.dm.dataset.coef_stats, with_global_pose=False,
                                    rot_repr=self.cfg.MODEL.HEAD.ROT_REPR)
                coef_pred = get_coef_dict(target, shape_coef, self.dm.dataset.coef_stats, with_global_pose=False,
                                    rot_repr=self.cfg.MODEL.HEAD.ROT_REPR)
                
                verts_gt, _, _ = self.flame(coef_gt['shape'].view(-1, 100), coef_gt['exp'].view(-1, 50),
                                       coef_gt['pose'].view(-1, 6), return_lm2d=False, return_lm3d=False)
                verts_pred, _, _ = self.flame(coef_pred['shape'].view(-1, 100), coef_pred['exp'].view(-1, 50),
                                         coef_pred['pose'].view(-1, 6), return_lm2d=False, return_lm3d=False)
                
                seq_len = target.shape[1]
                verts_gt, verts_pred = verts_gt.view(-1, seq_len, 5023, 3), verts_pred.view(-1, seq_len, 5023, 3)
                loss_vert = self.criterion(verts_gt, verts_pred, reduction='none') if loss_cfg.GEOMETRIC.W_VERTEX > 0 else None

                vel_gt, vel_pred = verts_gt[:, 1:] - verts_gt[:, :-1], verts_pred[:, 1:] - verts_pred[:, :-1]
                loss_vel = self.criterion(vel_gt, vel_pred, reduction='none') if loss_cfg.GEOMETRIC.W_VELOCITY > 0 else None

                vel_pred = verts_pred[:, 1:] - verts_pred[:, :-1]
                loss_smooth = self.criterion(vel_pred[:, 1:], vel_pred[:, :-1], reduction='none') if loss_cfg.GEOMETRIC.W_SMOOTH > 0 else None

                # head pose losss
                if not self.cfg.MODEL.HEAD.NO_HEAD_POSE:
                    head_pose_gt = motion_coef_gt[:, :, 50:53]
                    head_pose_pred = target[:, :, 50:53]

                    loss_head_angle = self.criterion(head_pose_gt, head_pose_pred, reduction='none') if loss_cfg.HEAD.W_ANGLE > 0 else None

                    head_vel_gt, head_vel_pred = head_pose_gt[:, 1:] - head_pose_gt[:, :-1], head_pose_pred[:, 1:] - head_pose_pred[:, :-1]
                    loss_head_vel = self.criterion(head_vel_gt, head_vel_pred, reduction='none') if loss_cfg.HEAD.W_VELOCITY > 0 else None
                    
                    head_vel_pred = head_pose_pred[:, 1:] - head_pose_pred[:, :-1]
                    loss_head_smooth = self.criterion(head_vel_pred[:, 1:], head_vel_pred[:, :-1], reduction='none') if loss_cfg.HEAD.W_SMOOTH > 0 else None

                    if clip_id != 0 and loss_cfg.HEAD.W_TRANS > 0:
                        # # version 1: constrain both the predicted previous and current motions (x_{-3} ~ x_{2})
                        # head_pose_trans = head_pose_pred[:, args.n_prev_motions - 3:args.n_prev_motions + 3]
                        # head_vel_pred = head_pose_trans[:, 1:] - head_pose_trans[:, :-1]
                        # head_accel_pred = head_vel_pred[:, 1:] - head_vel_pred[:, :-1]

                        # version 2: constrain only the predicted current motions (x_{0} ~ x_{2})
                        head_pose_trans = torch.cat([head_pose_gt[:, data_cfg.N_PREV_MOTIONS - 3:data_cfg.N_PREV_MOTIONS],
                                                    head_pose_pred[:, data_cfg.N_PREV_MOTIONS:data_cfg.N_PREV_MOTIONS + 3]], dim=1)
                        head_vel_pred = head_pose_trans[:, 1:] - head_pose_trans[:, :-1]
                        head_accel_pred = head_vel_pred[:, 1:] - head_vel_pred[:, :-1]

                        # will constrain x_{-2|0} ~ x_{1}
                        loss_head_trans_vel = self.criterion(head_vel_pred[:, 2:4], head_vel_pred[:, 1:3], reduction='none')
                        # will constrain x_{-3|0} ~ x_{2}
                        loss_head_trans_accel = self.criterion(head_accel_pred[:, 1:], head_accel_pred[:, :-1], reduction='none')
            else:
                raise ValueError(f'Unknown diffusion target: {loss_cfg.TARGET}')
            
            if end_idx is None:
                mask = torch.ones((target.shape[0], data_cfg.MOTIONS), dtype=torch.bool, device=target.device)
            else:
                mask = torch.arange(data_cfg.MOTIONS, device=target.device).expand(target.shape[0], -1) < end_idx.unsqueeze(1)

            if loss_cfg.TARGET == 'sample' and clip_id != 0:
                if loss_cfg.NO_CONSTRAIN_PREV:
                    # Warning: this option will be deprecated in the future
                    mask = torch.cat([torch.zeros_like(mask[:, :data_cfg.N_PREV_MOTIONS]), mask], dim=1)
                else:
                    mask = torch.cat([torch.ones_like(mask[:, :data_cfg.N_PREV_MOTIONS]), mask], dim=1)

            loss_dict['noise'] += (loss_noise[mask].mean() / 2)
            if loss_cfg.TARGET == 'sample':
                loss_dict['vert'] += (loss_vert[mask].mean() / 2) if loss_vert is not None else 0.0
                loss_dict['vel'] += (loss_vel[mask[:, 1:]].mean() / 2) if loss_vel is not None and torch.numel(loss_vel) > 0 else 0.0
                loss_dict['smooth'] += (loss_smooth[mask[:, 2:]].mean() / 2) if loss_smooth is not None and torch.numel(loss_smooth) > 0 else 0.0
                loss_dict['head_angle'] += (loss_head_angle[mask].mean() / 2) if loss_head_angle is not None else 0.0
                loss_dict['head_vel'] += (loss_head_vel[mask[:, 1:]].mean() / 2) if loss_head_vel is not None and torch.numel(loss_head_vel) > 0 else 0.0
                loss_dict['head_smooth'] += (loss_head_smooth[mask[:, 2:]].mean() / 2) if loss_head_smooth is not None and torch.numel(loss_head_smooth) > 0 else 0.0
                
                if clip_id != 0 and loss_cfg.HEAD.W_TRANS > 0:
                    vel_mask = mask[:, data_cfg.N_PREV_MOTIONS:data_cfg.N_PREV_MOTIONS + 2]
                    accel_mask = mask[:, data_cfg.N_PREV_MOTIONS:data_cfg.N_PREV_MOTIONS + 3]
                    loss_head_trans_vel = loss_head_trans_vel[vel_mask].mean()
                    loss_head_trans_accel = loss_head_trans_accel[accel_mask].mean()
                    loss_dict['head_smooth'] += (loss_head_trans_vel + loss_head_trans_accel)
        
        loss = loss_dict['noise'] + \
                loss_cfg.GEOMETRIC.W_VERTEX * loss_dict['vert'] + \
                loss_cfg.GEOMETRIC.W_VELOCITY * loss_dict['vel'] + \
                loss_cfg.GEOMETRIC.W_SMOOTH * loss_dict['smooth'] + \
                loss_cfg.HEAD.W_ANGLE * loss_dict['head_angle'] + \
                loss_cfg.HEAD.W_VELOCITY * loss_dict['head_vel'] + \
                loss_cfg.HEAD.W_SMOOTH * loss_dict['head_smooth'] + \
                loss_cfg.HEAD.W_TRANS * loss_dict['head_trans']

        loss_dict['vert'] = (loss_cfg.GEOMETRIC.W_VERTEX * loss_dict['vert'])
        loss_dict['vel'] = (loss_cfg.GEOMETRIC.W_VELOCITY * loss_dict['vel'])
        loss_dict['smooth'] = (loss_cfg.GEOMETRIC.W_SMOOTH * loss_dict['smooth'])
        loss_dict['head_angle'] = (loss_cfg.HEAD.W_ANGLE * loss_dict['head_angle'])
        loss_dict['head_vel'] = (loss_cfg.HEAD.W_VELOCITY * loss_dict['head_vel'])
        loss_dict['head_smooth'] = (loss_cfg.HEAD.W_SMOOTH * loss_dict['head_smooth'])
        loss_dict['head_trans'] = (loss_cfg.HEAD.W_TRANS * loss_dict['head_trans'])
        loss_dict['total'] = loss
        
        self.model_backward_and_update(loss)
        return loss_dict
    
    @torch.no_grad()
    def test(self, split=None, n_rounds=1):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        logger.info(f"Evaluate on the *{split}* set")

        loss_meter = defaultdict(AverageMeter)
        for test_round in range(n_rounds):
            for batch_idx, batch in enumerate(data_loader):
                current_iter = test_round * len(data_loader) + batch_idx
                data_cfg, loss_cfg = self.cfg.DATASET.HDTF_TFHP, self.cfg.LOSS
                name, audio_pair, motion_coef_pair, shape_coef = self.parse_batch(batch)
        
                # Extract style features
                with torch.no_grad():
                    style_pair = [self.style_enc(motion_coef_pair[i]) for i in range(2)] if self.style_enc else None

                # Get the actual model (handle DataParallel wrapper)
                model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

                if data_cfg.USE_CONTEXT_AUDIO:
                    # Extract audio features
                    audio_feat = model.extract_audio_feature(torch.cat(audio_pair, dim=1), data_cfg.MOTIONS * 2)  # (N, 2L, :)

                loss_dict = {'noise': 0.0, 'vert': 0.0, 'vel': 0.0, 'smooth': 0.0,
                     'head_angle': 0.0, 'head_vel': 0.0, 'head_smooth': 0.0, 'head_trans': 0.0}
                for clip_id in range(2):
                    audio = audio_pair[clip_id]  # (N, L_a)
                    motion_coef = motion_coef_pair[clip_id]  # (N, L, 50+x)
                    style = style_pair[1 - clip_id] if style_pair else None
                    batch_size = audio.shape[0]
                
                    # truncate input audio and motion according to trunc_prob
                    if (clip_id == 0 and np.random.rand() < data_cfg.TRUNC_PROB1) or (clip_id != 0 and np.random.rand() < data_cfg.TRUNC_PROB2):
                        audio_in, motion_coef_in, end_idx = truncate_motion_coef_and_audio(audio, motion_coef, data_cfg.MOTIONS,
                                                                                        self.dm.dataset.audio_unit, data_cfg.PAD_MODE)
                        if data_cfg.USE_CONTEXT_AUDIO and clip_id != 0:
                            # use contextualized audio feature for the second clip
                            audio_in = model.extract_audio_feature(torch.cat([audio_pair[clip_id - 1], audio_in], dim=1),
                                    data_cfg.MOTIONS * 2)[:, -data_cfg.MOTIONS:]
                    else:
                        if data_cfg.USE_CONTEXT_AUDIO:
                            audio_in = audio_feat[:, clip_id * data_cfg.MOTIONS:(clip_id + 1) * data_cfg.MOTIONS]
                        else:
                            audio_in = audio
                        motion_coef_in, end_idx = motion_coef, None

                    # prepare indicator if needed
                    if self.cfg.MODEL.HEAD.USE_INDICATOR:
                        if end_idx is not None:
                            indicator = torch.arange(data_cfg.MOTIONS, device=self.device).expand(batch_size, -1) < end_idx.unsqueeze(1)
                        else:
                            indicator = torch.ones(batch_size, data_cfg.MOTIONS, device=self.device)
                    else:
                        indicator = None

                    # forwarding through diffusion model
                    if clip_id == 0:
                        noise, target, prev_motion_coef, prev_audio_feat = self.model(motion_coef_in, audio_in, shape_coef, style, indicator=indicator)
                        if end_idx is not None:  # was truncated, needs to use the complete feature
                            prev_motion_coef = motion_coef[:, -data_cfg.N_PREV_MOTIONS:]
                            if data_cfg.USE_CONTEXT_AUDIO:
                                prev_audio_feat = audio_feat[:, data_cfg.MOTIONS - data_cfg.N_PREV_MOTIONS:data_cfg.MOTIONS].detach()
                            else:
                                with torch.no_grad():
                                    prev_audio_feat = model.extract_audio_feature(audio)[:, -data_cfg.N_PREV_MOTIONS:]
                        else:
                            prev_motion_coef = prev_motion_coef[:, -data_cfg.N_PREV_MOTIONS:]
                            prev_audio_feat = prev_audio_feat[:, -data_cfg.N_PREV_MOTIONS:]
                    else:
                        noise, target, _, _ = self.model(motion_coef_in, audio_in, shape_coef, style,
                                                    prev_motion_coef, prev_audio_feat, indicator=indicator)
                

                    # simple loss
                    if loss_cfg.TARGET == 'noise':
                        loss_noise = self.criterion(noise, target[:, data_cfg.N_PREV_MOTIONS:], reduction='none')
                    elif loss_cfg.TARGET == 'sample':
                        motion_coef_gt = torch.cat([prev_motion_coef, motion_coef_in], dim=1) if clip_id != 0 else motion_coef_in
                        if clip_id == 0:
                            target = target[:, data_cfg.N_PREV_MOTIONS:]
                        elif clip_id != 0 and loss_cfg.NO_CONSTRAIN_PREV:
                            target = torch.cat([prev_motion_coef, target[:, data_cfg.N_PREV_MOTIONS:]], dim=1)

                        loss_noise = self.criterion(motion_coef_gt, target, reduction='none')
                        
                        # geometric losses
                        coef_gt = get_coef_dict(motion_coef_gt, shape_coef, self.dm.dataset.coef_stats, with_global_pose=False,
                                            rot_repr=self.cfg.MODEL.HEAD.ROT_REPR)
                        coef_pred = get_coef_dict(target, shape_coef, self.dm.dataset.coef_stats, with_global_pose=False,
                                            rot_repr=self.cfg.MODEL.HEAD.ROT_REPR)
                        
                        verts_gt, _, _ = self.flame(coef_gt['shape'].view(-1, 100), coef_gt['exp'].view(-1, 50),
                                            coef_gt['pose'].view(-1, 6), return_lm2d=False, return_lm3d=False)
                        verts_pred, _, _ = self.flame(coef_pred['shape'].view(-1, 100), coef_pred['exp'].view(-1, 50),
                                                coef_pred['pose'].view(-1, 6), return_lm2d=False, return_lm3d=False)
                        
                        seq_len = target.shape[1]
                        verts_gt, verts_pred = verts_gt.view(-1, seq_len, 5023, 3), verts_pred.view(-1, seq_len, 5023, 3)
                        loss_vert = self.criterion(verts_gt, verts_pred, reduction='none') if loss_cfg.GEOMETRIC.W_VERTEX > 0 else None

                        vel_gt, vel_pred = verts_gt[:, 1:] - verts_gt[:, :-1], verts_pred[:, 1:] - verts_pred[:, :-1]
                        loss_vel = self.criterion(vel_gt, vel_pred, reduction='none') if loss_cfg.GEOMETRIC.W_VELOCITY > 0 else None

                        vel_pred = verts_pred[:, 1:] - verts_pred[:, :-1]
                        loss_smooth = self.criterion(vel_pred[:, 1:], vel_pred[:, :-1], reduction='none') if loss_cfg.GEOMETRIC.W_SMOOTH > 0 else None

                        # head pose losss
                        if not self.cfg.MODEL.HEAD.NO_HEAD_POSE:
                            head_pose_gt = motion_coef_gt[:, :, 50:53]
                            head_pose_pred = target[:, :, 50:53]

                            loss_head_angle = self.criterion(head_pose_gt, head_pose_pred, reduction='none') if loss_cfg.HEAD.W_ANGLE > 0 else None

                            head_vel_gt, head_vel_pred = head_pose_gt[:, 1:] - head_pose_gt[:, :-1], head_pose_pred[:, 1:] - head_pose_pred[:, :-1]
                            loss_head_vel = self.criterion(head_vel_gt, head_vel_pred, reduction='none') if loss_cfg.HEAD.W_VELOCITY > 0 else None
                            
                            head_vel_pred = head_pose_pred[:, 1:] - head_pose_pred[:, :-1]
                            loss_head_smooth = self.criterion(head_vel_pred[:, 1:], head_vel_pred[:, :-1], reduction='none') if loss_cfg.HEAD.W_SMOOTH > 0 else None

                            if clip_id != 0 and loss_cfg.HEAD.W_TRANS > 0:
                                # # version 1: constrain both the predicted previous and current motions (x_{-3} ~ x_{2})
                                # head_pose_trans = head_pose_pred[:, args.n_prev_motions - 3:args.n_prev_motions + 3]
                                # head_vel_pred = head_pose_trans[:, 1:] - head_pose_trans[:, :-1]
                                # head_accel_pred = head_vel_pred[:, 1:] - head_vel_pred[:, :-1]

                                # version 2: constrain only the predicted current motions (x_{0} ~ x_{2})
                                head_pose_trans = torch.cat([head_pose_gt[:, data_cfg.N_PREV_MOTIONS - 3:data_cfg.N_PREV_MOTIONS],
                                                            head_pose_pred[:, data_cfg.N_PREV_MOTIONS:data_cfg.N_PREV_MOTIONS + 3]], dim=1)
                                head_vel_pred = head_pose_trans[:, 1:] - head_pose_trans[:, :-1]
                                head_accel_pred = head_vel_pred[:, 1:] - head_vel_pred[:, :-1]

                                # will constrain x_{-2|0} ~ x_{1}
                                loss_head_trans_vel = self.criterion(head_vel_pred[:, 2:4], head_vel_pred[:, 1:3], reduction='none')
                                # will constrain x_{-3|0} ~ x_{2}
                                loss_head_trans_accel = self.criterion(head_accel_pred[:, 1:], head_accel_pred[:, :-1], reduction='none')
                    else:
                        raise ValueError(f'Unknown diffusion target: {loss_cfg.TARGET}')
            
                    if end_idx is None:
                        mask = torch.ones((target.shape[0], data_cfg.MOTIONS), dtype=torch.bool, device=target.device)
                    else:
                        mask = torch.arange(data_cfg.MOTIONS, device=target.device).expand(target.shape[0], -1) < end_idx.unsqueeze(1)

                    if loss_cfg.TARGET == 'sample' and clip_id != 0:
                        if loss_cfg.NO_CONSTRAIN_PREV:
                            # Warning: this option will be deprecated in the future
                            mask = torch.cat([torch.zeros_like(mask[:, :data_cfg.N_PREV_MOTIONS]), mask], dim=1)
                        else:
                            mask = torch.cat([torch.ones_like(mask[:, :data_cfg.N_PREV_MOTIONS]), mask], dim=1)

                    loss_dict['noise'] += (loss_noise[mask].mean() / 2)
                    if loss_cfg.TARGET == 'sample':
                        loss_dict['vert'] += (loss_vert[mask].mean() / 2) if loss_vert is not None else 0.0
                        loss_dict['vel'] += (loss_vel[mask[:, 1:]].mean() / 2) if loss_vel is not None and torch.numel(loss_vel) > 0 else 0.0
                        loss_dict['smooth'] += (loss_smooth[mask[:, 2:]].mean() / 2) if loss_smooth is not None and torch.numel(loss_smooth) > 0 else 0.0
                        loss_dict['head_angle'] += (loss_head_angle[mask].mean() / 2) if loss_head_angle is not None else 0.0
                        loss_dict['head_vel'] += (loss_head_vel[mask[:, 1:]].mean() / 2) if loss_head_vel is not None and torch.numel(loss_head_vel) > 0 else 0.0
                        loss_dict['head_smooth'] += (loss_head_smooth[mask[:, 2:]].mean() / 2) if loss_head_smooth is not None and torch.numel(loss_head_smooth) > 0 else 0.0
                        
                        if clip_id != 0 and loss_cfg.HEAD.W_TRANS > 0:
                            vel_mask = mask[:, data_cfg.N_PREV_MOTIONS:data_cfg.N_PREV_MOTIONS + 2]
                            accel_mask = mask[:, data_cfg.N_PREV_MOTIONS:data_cfg.N_PREV_MOTIONS + 3]
                            loss_head_trans_vel = loss_head_trans_vel[vel_mask].mean()
                            loss_head_trans_accel = loss_head_trans_accel[accel_mask].mean()
                            loss_dict['head_smooth'] += (loss_head_trans_vel + loss_head_trans_accel)

                loss = loss_dict['noise'] + \
                        loss_cfg.GEOMETRIC.W_VERTEX * loss_dict['vert'] + \
                        loss_cfg.GEOMETRIC.W_VELOCITY * loss_dict['vel'] + \
                        loss_cfg.GEOMETRIC.W_SMOOTH * loss_dict['smooth'] + \
                        loss_cfg.HEAD.W_ANGLE * loss_dict['head_angle'] + \
                        loss_cfg.HEAD.W_VELOCITY * loss_dict['head_vel'] + \
                        loss_cfg.HEAD.W_SMOOTH * loss_dict['head_smooth'] + \
                        loss_cfg.HEAD.W_TRANS * loss_dict['head_trans']

                loss_meter['total'].update(loss.item())
                loss_meter['vert'].update(loss_cfg.GEOMETRIC.W_VERTEX * loss_dict['vert'])
                loss_meter['vel'].update(loss_cfg.GEOMETRIC.W_VELOCITY * loss_dict['vel'])
                loss_meter['smooth'].update(loss_cfg.GEOMETRIC.W_SMOOTH * loss_dict['smooth'])
                loss_meter['head_angle'].update(loss_cfg.HEAD.W_ANGLE * loss_dict['head_angle'])
                loss_meter['head_vel'].update(loss_cfg.HEAD.W_VELOCITY * loss_dict['head_vel'])
                loss_meter['head_smooth'].update(loss_cfg.HEAD.W_SMOOTH * loss_dict['head_smooth'])
                loss_meter['head_trans'].update(loss_cfg.HEAD.W_TRANS * loss_dict['head_trans'])

                if (current_iter + 1) % self.cfg.TRAIN.PRINT_FREQ == 0:
                    loss_info = ' '.join([f'loss_{item} {loss.avg:.4f}' for item, loss in loss_meter.items()])
                    logger.info(f'iter: {current_iter + 1} {loss_info}')
                    
                for item, loss in loss_meter.items():
                    if self.cfg.ENV.USE_WANDB:
                        self.wandb_run.log({f"val/loss_{item}": loss.avg})
                    self.write_scalar(f"val/loss_{item}", loss.avg, current_iter)

    def parse_batch(self, batch):
        """Parse batch data."""
        name = batch["name"]
        motion_coef_pair = [motion.to(self.device) for motion in batch["motion_coef"]]
        shape_coef = batch["shape_coef"][:, 0, :].to(self.device)
        audio_pair = [audio.to(self.device) for audio in batch["audio"]]

        return name, audio_pair, motion_coef_pair, shape_coef

    def get_current_lr(self, names=None):
        """Get current learning rate."""
        names = self.get_model_names(names)
        name = names[0]
        return self._optims[name].param_groups[0]["lr"]
