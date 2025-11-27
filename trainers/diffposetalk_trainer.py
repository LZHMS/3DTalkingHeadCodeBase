import time
import datetime
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

from base.base_trainer import TrainerBase, TRAINER_REGISTRY
from dataset.HDTF_TFHP import HDTF_TFHPDM, HDTF_TFHPWrapper
from models.diffposetalk import StyleEncoder, DiffTalkingHead
from base.base_evaluator import build_evaluator 
from evaluation.TalkerEvaluator import TDTalkerEvaluator

from utils.meters import AverageMeter
from utils.data_tool import truncate_motion_coef_and_audio

import logging
logger: logging.Logger


@TRAINER_REGISTRY.register()
class StyleEncoderTrainer(TrainerBase):
    def __init__(self, cfg):
        super().__init__(cfg)

        # Builf components of trainer
        self.build_data_loader()
        self.build_model()
        self.evaluator = build_evaluator(cfg, self.dm.dataset.coef_stats,
                                         audio_sr=cfg.DATASET.HDTF_TFHP.AUDIO_SR,
                                         coef_fps=cfg.DATASET.HDTF_TFHP.COEF_FPS,
                                         device=self.device)
        
    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        dm = HDTF_TFHPDM(self.cfg, HDTF_TFHPWrapper, infinite_train=True)

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
        self.loss_meter["loss"].update(loss)

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
            info += [f"loss {self.loss_meter['loss'].val:.4f}"]
            info += [f"lr {self.get_current_lr():.4e}"]
            info += [f"eta {eta}"]
            logger.info(" ".join(info))
        
        self.write_scalar("train/loss", self.loss_meter['loss'].avg, self.iter)
        self.write_scalar("train/lr", self.get_current_lr(), self.iter)
        self.write_scalar("train/train_step", self.iter, self.iter, wandb=True)
    
    def after_iter(self):
        last_iter = (self.iter + 1) == self.max_iters
        
        # Validation
        if self.cfg.TRAIN.EVALUATE:
            if ((self.iter + 1) % self.cfg.TRAIN.EVAL_FREQ == 0) or last_iter:
                self.test(split="val", n_rounds=200)

        if ((self.iter + 1) % self.cfg.TRAIN.SAVE_FREQ == 0) or last_iter:
            self.save_model(iter=self.iter, directory=self.output_dir)

    def forward_backward(self, batch):
        name, motion_coef = self.parse_batch(batch)
        
        feats_0, feats_1 = self.model(motion_coef[0]), self.model(motion_coef[1])
        loss = self.evaluator.criterion(feats_0, feats_1, self.cfg.EVALUATE.LOSS.CONTRASTIVE.TEMPRATURE)
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
                loss = self.evaluator.criterion(feats_0, feats_1, self.cfg.EVALUATE.LOSS.CONTRASTIVE.TEMPRATURE)
                loss_meter.update(loss.item())

                if (current_iter + 1) % self.cfg.TRAIN.PRINT_FREQ == 0:
                    logger.info('iter: {} '
                                'loss: {} '
                                .format(current_iter + 1, loss_meter.avg))

                self.write_scalar("val/loss", loss_meter.avg, current_iter)
                self.write_scalar("val/val_step", current_iter, current_iter, wandb=True)

    def parse_batch(self, batch):
        name = batch["name"]
        motion_coef = [motion.to(self.device) for motion in batch["motion_coef"]]

        return name, motion_coef
    

@TRAINER_REGISTRY.register()
class DiffPoseTalkTrainer(TrainerBase):
    def __init__(self, cfg):
        super().__init__(cfg)

        # Build components of trainer
        self.build_data_loader()
        self.build_model()
        self.evaluator = build_evaluator(cfg, self.dm.dataset.coef_stats,
                                         audio_sr=cfg.DATASET.HDTF_TFHP.AUDIO_SR,
                                         coef_fps=cfg.DATASET.HDTF_TFHP.COEF_FPS,
                                         device=self.device)

    def build_data_loader(self):
        """Create essential data-related attributes."""
        dm = HDTF_TFHPDM(self.cfg, HDTF_TFHPWrapper, infinite_train=True)

        self.train_loader = dm.train_loader
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.dm = dm

    def build_model(self):
        """Build and register model."""
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
        
        for item, loss in self.loss_meter.items():
            self.write_scalar(f"train/loss_{item}", loss.avg, self.iter)
        self.write_scalar("train/lr", self.get_current_lr(), self.iter)
    
    def after_iter(self):
        """Actions after each iteration."""
        last_iter = (self.iter + 1) == self.max_iters
        
        # Validation
        if self.cfg.TRAIN.EVALUATE:
            if ((self.iter + 1) % self.cfg.TRAIN.EVAL_FREQ == 0) or last_iter:
                self.test(split="val", n_rounds=200)

        # Save model
        if ((self.iter + 1) % self.cfg.TRAIN.SAVE_FREQ == 0) or last_iter:
            self.save_model(iter=self.iter, directory=self.output_dir)

    def forward_backward(self, batch):
        """
        Forward and backward pass for Diffusion Model.
        """
        data_cfg, eval_cfg = self.cfg.DATASET.HDTF_TFHP, self.cfg.EVALUATE
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
        loss_total, loss_dict = 0.0, {'noise': 0.0}
        self.evaluator.reset(shape_coef)
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
            if eval_cfg.TARGET == 'noise':
                loss_noise = self.evaluator.simple_loss(noise, target[:, data_cfg.N_PREV_MOTIONS:], end_idx=end_idx)
            elif eval_cfg.TARGET == 'sample':
                motion_coef_gt = torch.cat([prev_motion_coef, motion_coef_in], dim=1) if clip_id != 0 else motion_coef_in
                if clip_id == 0:
                    target = target[:, data_cfg.N_PREV_MOTIONS:]
                elif clip_id != 0 and eval_cfg.NO_CONSTRAIN_PREV:
                    target = torch.cat([prev_motion_coef, target[:, data_cfg.N_PREV_MOTIONS:]], dim=1)

                loss_noise = self.evaluator.simple_loss(motion_coef_gt, target, end_idx=end_idx)
                
                # geometric losses
                geometric_losses_dict = self.evaluator.geometric_losses(motion_coef_gt, target, end_idx)
            else:
                raise ValueError(f'Unknown diffusion target: {eval_cfg.TARGET}')
            
            # Total loss
            loss_total += loss_noise
            loss_dict['noise'] += loss_noise
            for geo_item, loss in geometric_losses_dict.items():
                loss_total += loss
                if geo_item in loss_dict:
                    loss_dict[geo_item] += loss
                else:
                    loss_dict[geo_item] = loss

        loss_dict['total'] = loss_total
        self.model_backward_and_update(loss_total)
        return loss_dict
    
    @torch.no_grad()
    def test(self, split=None, n_rounds=1):
        """A generic testing pipeline."""
        self.set_model_mode("eval")

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        logger.info(f"Evaluate on the *{split}* set")
        data_cfg, eval_cfg = self.cfg.DATASET.HDTF_TFHP, self.cfg.EVALUATE
        loss_meter = defaultdict(AverageMeter)
        for test_round in range(n_rounds):
            for batch_idx, batch in enumerate(data_loader):
                current_iter = test_round * len(data_loader) + batch_idx
                
                name, audio_pair, motion_coef_pair, shape_coef = self.parse_batch(batch)
                # Extract style features
                with torch.no_grad():
                    style_pair = [self.style_enc(motion_coef_pair[i]) for i in range(2)] if self.style_enc else None

                # Get the actual model (handle DataParallel wrapper)
                model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

                if data_cfg.USE_CONTEXT_AUDIO:
                    # Extract audio features
                    audio_feat = model.extract_audio_feature(torch.cat(audio_pair, dim=1), data_cfg.MOTIONS * 2)  # (N, 2L, :)

                # Initialize loss dict
                loss_total, loss_dict = 0.0, {'noise': 0.0}
                self.evaluator.reset(shape_coef)
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
                    if eval_cfg.TARGET == 'noise':
                        loss_noise = self.evaluator.simple_loss(noise, target[:, data_cfg.N_PREV_MOTIONS:], end_idx=end_idx)
                    elif eval_cfg.TARGET == 'sample':
                        motion_coef_gt = torch.cat([prev_motion_coef, motion_coef_in], dim=1) if clip_id != 0 else motion_coef_in
                        if clip_id == 0:
                            target = target[:, data_cfg.N_PREV_MOTIONS:]
                        elif clip_id != 0 and eval_cfg.NO_CONSTRAIN_PREV:
                            target = torch.cat([prev_motion_coef, target[:, data_cfg.N_PREV_MOTIONS:]], dim=1)

                        loss_noise = self.evaluator.simple_loss(motion_coef_gt, target, end_idx=end_idx)
                        
                        # geometric losses
                        geometric_losses_dict = self.evaluator.geometric_losses(motion_coef_gt, target, end_idx)
                    else:
                        raise ValueError(f'Unknown diffusion target: {eval_cfg.TARGET}')

                    # render and save results (only render the first sample in batch)
                    if self.cfg.EVALUATE.LOAD_RENDER and (current_iter + 1) % self.cfg.RENDER_FREQ == 0:
                        sample_name = name[0] if isinstance(name, list) else name
                        sample_name = sample_name.replace('/', '_')  # avoid path separator in filename
                        self.evaluator.render_and_save(
                            sample_name, target[0], audio[0], clip_id, self.output_dir,
                            shape_coef=shape_coef[0]
                        )

                    # Total loss
                    loss_total += loss_noise
                    loss_dict['noise'] += loss_noise
                    for geo_item, loss in geometric_losses_dict.items():
                        loss_total += loss
                        if geo_item in loss_dict:
                            loss_dict[geo_item] += loss
                        else:
                            loss_dict[geo_item] = loss

                loss_meter['total'].update(loss_total.item())
                for loss_item, loss in loss_dict.items():
                    loss_meter[loss_item].update(loss.item())

                # logging for output
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
