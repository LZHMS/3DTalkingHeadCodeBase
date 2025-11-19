"""
FlowMatching Trainer for 3D Talking Head.
Adapted from DiffPoseTalkTrainer and MeanAudio RunnerFlowMatching.
"""
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

from base import TrainerBase, TRAINER_REGISTRY, build_evaluator
from datasets import HDTF_TFHPDM, StyledTalkWrapper, HDTF_TFHPWrapper
from models import StyleEncoder, FlowMatchingHead, FLAME, FLAMEConfig
from utils import AverageMeter, truncate_motion_coef_and_audio, get_coef_dict

import logging
logger: logging.Logger


@TRAINER_REGISTRY.register()
class FlowMatchingTrainer(TrainerBase):
    """Trainer for Flow Matching based talking head generation."""
    
    def __init__(self, assistant):
        super().__init__(assistant)

        # Load pre-trained style encoder
        if self.assistant.cfg.ADD.STYLE_ENC_CKPT != '':
            checkpoint = self.load_checkpoint(self.assistant.cfg.ADD.STYLE_ENC_CKPT)
            logger.info(
                f"Load {self.assistant.cfg.ADD.STYLE_ENC_CKPT} to StyleEncoder (iter={checkpoint['iter']})"
            )
        else:
            raise ValueError("Please provide pre-trained style encoder checkpoint path.")
        
        self.style_enc = StyleEncoder(checkpoint['model_config']).to(self.device)
        self.style_enc.load_state_dict(checkpoint['state_dict'])
        self.style_enc.eval()

        # Avatar model for loss computation
        self.flame = FLAME(FLAMEConfig).to(self.device)
        logger.info(f"Loaded FLAME model for loss computation.")

        # Build components of trainer
        self.build_data_loader()
        self.build_model()
        self.evaluator = build_evaluator(assistant)
        self.criterion = self.build_loss_metrics(self.assistant.cfg.LOSS.NAME)

    def build_data_loader(self):
        """Create essential data-related attributes."""
        dm = HDTF_TFHPDM(self.assistant.cfg, HDTF_TFHPWrapper, infinite_train=True)

        self.train_loader = dm.train_loader
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.dm = dm

    def build_model(self):
        """Build and register model."""
        logger.info(f"Building model FlowMatchingHead ...")
        self.model = FlowMatchingHead(self.assistant.cfg)

        if self.assistant.cfg.MODEL.INIT_WEIGHTS:
            self.load_pretrained_weights(self.model, self.assistant.cfg.MODEL.INIT_WEIGHTS)
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
        if (self.iter + 1) % self.assistant.cfg.OPTIM.LR_UPDATE_FREQ == 0:
            self.update_lr()

        # Logging
        eta_seconds = self.batch_time.avg * (self.max_iters - self.iter)
        eta = str(datetime.timedelta(seconds=int(eta_seconds)))
        if (self.iter + 1) % self.assistant.cfg.TRAIN.PRINT_FREQ == 0:
            info = []
            info += [f"iter [{self.iter + 1}/{self.max_iters}]"]
            info += [f"time {self.batch_time.val:.3f} ({self.batch_time.avg:.3f})"]
            info += [f"data {self.data_time.val:.3f} ({self.data_time.avg:.3f})"]
            info += [f"loss_{item} {loss.val:.4f}" for item, loss in self.loss_meter.items()]
            info += [f"lr {self.get_current_lr():.4e}"]
            info += [f"eta {eta}"]
            logger.info(" ".join(info))
        
        if self.assistant.cfg.ENV.USE_WANDB:
            log_dict = {"iter": self.iter + 1, "batch_time": self.batch_time.val, "train/lr": self.get_current_lr()}
            log_dict.update({f"train/loss_{item}": loss.avg for item, loss in self.loss_meter.items()})
            self.assistant.wandb_run.log(log_dict)
        
        for item, loss in self.loss_meter.items():
            self.write_scalar(f"train/loss_{item}", loss.avg, self.iter)
        self.write_scalar("train/lr", self.get_current_lr(), self.iter)
    
    def after_iter(self):
        """Actions after each iteration."""
        last_iter = (self.iter + 1) == self.max_iters
        
        # Validation
        if ((self.iter + 1) % self.assistant.cfg.TRAIN.EVAL_FREQ == 0) or last_iter:
            self.test(split="val", n_rounds=200)

        # Save model
        if ((self.iter + 1) % self.assistant.cfg.TRAIN.SAVE_FREQ == 0) or last_iter:
            self.save_model(iter=self.iter, directory=self.output_dir)

    def forward_backward(self, batch):
        """
        Forward and backward pass for Flow Matching.
        Adapted from DiffPoseTalkTrainer but using flow matching instead of diffusion.
        """
        data_cfg, loss_cfg = self.assistant.cfg.DATASET.HDTF_TFHP, self.assistant.cfg.LOSS
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
            audio = audio_pair[clip_id]
            motion_coef = motion_coef_pair[clip_id]
            style = style_pair[1 - clip_id] if style_pair else None
            batch_size = audio.shape[0]

            # Truncate input audio and motion according to trunc_prob
            if (clip_id == 0 and np.random.rand() < data_cfg.TRUNC_PROB1) or \
               (clip_id != 0 and np.random.rand() < data_cfg.TRUNC_PROB2):
                audio_in, motion_coef_in, end_idx = truncate_motion_coef_and_audio(
                    audio, motion_coef, data_cfg.MOTIONS, self.dm.dataset.audio_unit, data_cfg.PAD_MODE
                )
                if data_cfg.USE_CONTEXT_AUDIO and clip_id != 0:
                    audio_in = model.extract_audio_feature(
                        torch.cat([audio_pair[clip_id - 1], audio_in], dim=1), data_cfg.MOTIONS * 2
                    )[:, -data_cfg.MOTIONS:]
            else:
                if data_cfg.USE_CONTEXT_AUDIO:
                    audio_in = audio_feat[:, clip_id * data_cfg.MOTIONS:(clip_id + 1) * data_cfg.MOTIONS]
                else:
                    audio_in = audio
                motion_coef_in, end_idx = motion_coef, None

            # Prepare indicator if needed
            if self.assistant.cfg.MODEL.HEAD.USE_INDICATOR:
                if end_idx is not None:
                    indicator = torch.arange(data_cfg.MOTIONS, device=self.device).expand(batch_size, -1) < end_idx.unsqueeze(1)
                else:
                    indicator = torch.ones(batch_size, data_cfg.MOTIONS, device=self.device)
            else:
                indicator = None

            # Forward through flow matching model
            if clip_id == 0:
                predicted_v, target_v, prev_motion_coef, prev_audio_feat = self.model(
                    motion_coef_in, audio_in, shape_coef, style, indicator=indicator
                )
                if end_idx is not None:
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
                predicted_v, target_v, _, _ = self.model(
                    motion_coef_in, audio_in, shape_coef, style,
                    prev_motion_coef, prev_audio_feat, indicator=indicator
                )
            
            # Flow matching loss
            loss_flow = self.criterion(predicted_v, target_v[:, data_cfg.N_PREV_MOTIONS:], reduction='none')
            
            # Geometric losses (optional, can be computed from predicted samples)
            # For flow matching, we can optionally compute geometric losses
            # by predicting x0 from the flow and comparing with ground truth
            if loss_cfg.GEOMETRIC.W_VERTEX > 0 or loss_cfg.GEOMETRIC.W_VELOCITY > 0 or loss_cfg.GEOMETRIC.W_SMOOTH > 0:
                # This part is optional and can be enabled if needed
                # For now, we focus on flow matching loss
                pass

            # Mask handling
            if end_idx is None:
                mask = torch.ones((predicted_v.shape[0], data_cfg.MOTIONS), dtype=torch.bool, device=predicted_v.device)
            else:
                mask = torch.arange(data_cfg.MOTIONS, device=predicted_v.device).expand(predicted_v.shape[0], -1) < end_idx.unsqueeze(1)

            loss_dict['flow'] += (loss_flow[mask].mean() / 2)

        # Total loss
        loss = loss_dict['flow']
        loss_dict['total'] = loss
        
        self.model_backward_and_update(loss)
        return loss_dict
    
    @torch.no_grad()
    def test(self, split=None, n_rounds=1):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.assistant.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"
            data_loader = self.test_loader

        logger.info(f"Evaluate on the *{split}* set")

        loss_meter = defaultdict(AverageMeter)
        for test_round in range(n_rounds):
            for batch_idx, batch in enumerate(data_loader):
                current_iter = test_round * len(data_loader) + batch_idx
                data_cfg, loss_cfg = self.assistant.cfg.DATASET.HDTF_TFHP, self.assistant.cfg.LOSS
                name, audio_pair, motion_coef_pair, shape_coef = self.parse_batch(batch)
        
                # Extract style features
                with torch.no_grad():
                    style_pair = [self.style_enc(motion_coef_pair[i]) for i in range(2)] if self.style_enc else None

                model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

                if data_cfg.USE_CONTEXT_AUDIO:
                    audio_feat = model.extract_audio_feature(torch.cat(audio_pair, dim=1), data_cfg.MOTIONS * 2)

                loss_dict = {'flow': 0.0}
                for clip_id in range(2):
                    audio = audio_pair[clip_id]
                    motion_coef = motion_coef_pair[clip_id]
                    style = style_pair[1 - clip_id] if style_pair else None
                    batch_size = audio.shape[0]
                
                    if data_cfg.USE_CONTEXT_AUDIO:
                        audio_in = audio_feat[:, clip_id * data_cfg.MOTIONS:(clip_id + 1) * data_cfg.MOTIONS]
                    else:
                        audio_in = audio
                    motion_coef_in = motion_coef

                    indicator = torch.ones(batch_size, data_cfg.MOTIONS, device=self.device) \
                        if self.assistant.cfg.MODEL.HEAD.USE_INDICATOR else None

                    if clip_id == 0:
                        predicted_v, target_v, prev_motion_coef, prev_audio_feat = self.model(
                            motion_coef_in, audio_in, shape_coef, style, indicator=indicator
                        )
                        prev_motion_coef = prev_motion_coef[:, -data_cfg.N_PREV_MOTIONS:]
                        prev_audio_feat = prev_audio_feat[:, -data_cfg.N_PREV_MOTIONS:]
                    else:
                        predicted_v, target_v, _, _ = self.model(
                            motion_coef_in, audio_in, shape_coef, style,
                            prev_motion_coef, prev_audio_feat, indicator=indicator
                        )

                    loss_flow = self.criterion(predicted_v, target_v[:, data_cfg.N_PREV_MOTIONS:], reduction='none')
                    loss_dict['flow'] += (loss_flow.mean() / 2)

                loss = loss_dict['flow']
                loss_meter['total'].update(loss.item())
                loss_meter['flow'].update(loss_dict['flow'].item())

                if (current_iter + 1) % self.assistant.cfg.TRAIN.PRINT_FREQ == 0:
                    loss_info = ' '.join([f'loss_{item} {loss.avg:.4f}' for item, loss in loss_meter.items()])
                    logger.info(f'iter: {current_iter + 1} {loss_info}')
                    
                for item, loss in loss_meter.items():
                    if self.assistant.cfg.ENV.USE_WANDB:
                        self.assistant.wandb_run.log({f"val/loss_{item}": loss.avg})
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
