import time
import datetime
import torch
import torch.nn as nn
import numpy as np
import os.path as osp
from base import TrainerBase, TRAINER_REGISTRY, build_evaluator
from datasets import CodeTalkerDataManager, HDTF_TFHPDM
from models import VQAutoEncoder
from utils import mkdir_if_missing, MetricMeter, AverageMeter
import logging
logger: logging.Logger


@TRAINER_REGISTRY.register()
class CodeStyle(TrainerBase):
    def __init__(self, assistant):
        super().__init__(assistant)

        # Save as attributes some frequently used variables
        self.start_epoch = assistant.cfg.TRAIN.START_EPOCH
        self.max_epoch = assistant.cfg.TRAIN.MAX_EPOCHS
        self.output_dir = assistant.cfg.ENV.OUTPUT_DIR
        self.assistant = assistant

        self.build_data_loader()
        self.build_model()
        self.evaluator = build_evaluator(assistant)
        self.criterion  = self.build_loss_metrics(self.assistant.cfg.LOSS.NAME)
        self.best_result = -np.inf

    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        dm = HDTF_TFHPDM(self.assistant.cfg)

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

        logger.info(f"Building model {self.assistant.cfg.MODEL.NAME} ...")
        self.model = StyleVQAE(self.assistant.cfg.MODEL)
        if self.assistant.cfg.MODEL.INIT_WEIGHTS:
            self.load_pretrained_weights(self.model, self.assistant.cfg.MODEL.INIT_WEIGHTS)
        self.model.to(self.device)
        logger.info(f"Params: {self.count_num_param(self.model):,}")
        logger.info(f"Model Structure:\n{self.model}")

        logger.info(f"Building optimizer ...")
        self.optim = self.build_optimizer(self.model)
        self.sched = self.build_lr_scheduler(self.optim)
        self.register_model("model", self.model, self.optim, self.sched)

        device_count = torch.cuda.device_count()
        if device_count > 1:
            logger.info(f"Detected {device_count} GPUs (use nn.DataParallel)")
            self.model = nn.DataParallel(self.model)

    def train(self):
        super().train(self.start_epoch, self.max_epoch)

    def before_train(self):
        directory = self.output_dir
        if self.assistant.cfg.ENV.RESUME:
            directory = self.assistant.cfg.ENV.RESUME
        self.start_epoch = self.resume_model_if_exist(directory)

        # Initialize summary writer
        writer_dir = osp.join(self.output_dir, "tensorboard")
        mkdir_if_missing(writer_dir)
        self.init_writer(writer_dir)

        # Remember the starting time (for computing the elapsed time)
        self.time_start = time.time()

    def after_train(self):
        logger.info("Finish training!")

        do_test = not self.assistant.cfg.TEST.NO_TEST
        if do_test:
            if self.assistant.cfg.TEST.FINAL_MODEL == "best_val":
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

    def run_epoch(self):
        self.set_model_mode("train")
        batch_time = AverageMeter()
        data_time = AverageMeter()
        rec_loss_meter = AverageMeter()
        quant_loss_meter = AverageMeter()
        pp_meter = AverageMeter()

        self.num_batches = len(self.train_loader)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader):
            data_time.update(time.time() - end)
            loss_details, info = self.forward_backward(batch)
            batch_time.update(time.time() - end)

            for m, x in zip([rec_loss_meter, quant_loss_meter, pp_meter],
                [loss_details[0], loss_details[1], info[0]]): #info[0] is perplexity
                m.update(x.item(), 1)
            
            # update learning rate
            if (self.batch_idx + 1) == self.num_batches:
                self.update_lr()

            meet_freq = (self.batch_idx + 1) % self.assistant.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.assistant.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"loss {rec_loss_meter.val:.4f}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                logger.info(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in zip(["rec_loss", "quant_loss"],
                                   [rec_loss_meter, quant_loss_meter]):
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()
        
        logger.info('epoch: {} '
                        'loss_train: {} '
                        'pp_train: {} '
                        .format(self.epoch + 1, rec_loss_meter.avg, pp_meter.avg)
                        )
        for m, s in zip([rec_loss_meter.avg, quant_loss_meter.avg, pp_meter.avg],
                        ["train/rec_loss", "train/quant_loss", "train/perplexity"]):
            self.write_scalar(s, m, self.epoch + 1)

    def forward_backward(self, batch):
        name, audio, coefficients = self.parse_batch(batch)
        
        output, quant_loss, info = self.model(coefficients)
        loss, loss_details = self.criterion(output, 
                                     coefficients, 
                                     quant_loss, 
                                     quant_loss_weight=self.assistant.cfg.LOSS.QUANT_LOSS_WEIGHT)
        
        self.model_backward_and_update(loss)
        return loss_details, info
    
    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.assistant.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.assistant.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.assistant.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
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

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.assistant.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        logger.info(f"Evaluate on the *{split}* set")

        rec_loss_meter = AverageMeter()
        quant_loss_meter = AverageMeter()
        pp_meter = AverageMeter()
        for batch_idx, batch in enumerate(data_loader):
            name, vertices, template, _ = self.parse_batch(batch)
            output, quant_loss, info = self.model(vertices, template)
            loss, loss_details = self.criterion(output, 
                                     vertices, 
                                     quant_loss, 
                                     quant_loss_weight=self.assistant.cfg.LOSS.QUANT_LOSS_WEIGHT)

            for m, x in zip([rec_loss_meter, quant_loss_meter, pp_meter],
                [loss_details[0], loss_details[1], info[0]]): #info[0] is perplexity
                m.update(x.item(), 1)
        
        rec_loss_val, quant_loss_val, pp_val = rec_loss_meter.avg, quant_loss_meter.avg, pp_meter.avg
        logger.info('epoch: {} '
                            'loss_val: {} '
                            'pp_val: {} '
                            .format(self.epoch + 1, rec_loss_val, pp_val)
                            )

        for m, s in zip([rec_loss_val, quant_loss_val, pp_val],
                        ["val/rec_loss", "val/quant_loss", "val/perplexity"]):
            self.write_scalar(s, m, self.epoch)


    def parse_batch(self, batch):
        name = batch["name"]
        coefficients = []
        for k in ['shape', 'exp', 'pose']:
            coefficients.append(batch["coefficients"][k].to(self.device).float())
        
        if "audio" in batch:
            audio = batch["audio"].to(self.device)
            return name, audio.float(), coefficients
        
        return name, None, coefficients

    def get_current_lr(self, names=None):
        names = self.get_model_names(names)
        name = names[0]
        return self._optims[name].param_groups[0]["lr"]

@TRAINER_REGISTRY.register()
class CodeTalkerTrainer(TrainerBase):
    def __init__(self, assistant):
        super().__init__(assistant)

        # Save as attributes some frequently used variables
        self.start_epoch = self.epoch = 0
        self.max_epoch = assistant.cfg.OPTIM.MAX_EPOCH
        self.output_dir = assistant.cfg.ENV.OUTPUT_DIR
        self.assistant = assistant

        self.build_data_loader()
        self.build_model()
        self.evaluator = build_evaluator(assistant)
        self.criterion  = self.build_loss_metrics(self.assistant.cfg.LOSS.NAME)
        self.best_result = -np.inf

    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        dm = HDTF_TFHPDM(self.assistant.cfg)

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

        logger.info(f"Building model {self.assistant.cfg.MODEL.NAME} ...")
        self.model = VQAutoEncoder(self.assistant.cfg.MODEL)
        if self.assistant.cfg.MODEL.INIT_WEIGHTS:
            self.load_pretrained_weights(self.model, self.assistant.cfg.MODEL.INIT_WEIGHTS)
        self.model.to(self.device)
        logger.info(f"Params: {self.count_num_param(self.model):,}")
        logger.info(f"Model Structure:\n{self.model}")

        logger.info(f"Building optimizer ...")
        self.optim = self.build_optimizer(self.model)
        self.sched = self.build_lr_scheduler(self.optim)
        self.register_model("model", self.model, self.optim, self.sched)

        device_count = torch.cuda.device_count()
        if device_count > 1:
            logger.info(f"Detected {device_count} GPUs (use nn.DataParallel)")
            self.model = nn.DataParallel(self.model)

    def train(self):
        super().train(self.start_epoch, self.max_epoch)

    def before_train(self):
        directory = self.output_dir
        if self.assistant.cfg.ENV.RESUME:
            directory = self.assistant.cfg.ENV.RESUME
        self.start_epoch = self.resume_model_if_exist(directory)

        # Initialize summary writer
        writer_dir = osp.join(self.output_dir, "tensorboard")
        mkdir_if_missing(writer_dir)
        self.init_writer(writer_dir)

        # Remember the starting time (for computing the elapsed time)
        self.time_start = time.time()

    def after_train(self):
        logger.info("Finish training!")

        do_test = not self.assistant.cfg.TEST.NO_TEST
        if do_test:
            if self.assistant.cfg.TEST.FINAL_MODEL == "best_val":
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

    def run_epoch(self):
        self.set_model_mode("train")
        batch_time = AverageMeter()
        data_time = AverageMeter()
        rec_loss_meter = AverageMeter()
        quant_loss_meter = AverageMeter()
        pp_meter = AverageMeter()

        self.num_batches = len(self.train_loader)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader):
            data_time.update(time.time() - end)
            loss_details, info = self.forward_backward(batch)
            batch_time.update(time.time() - end)

            for m, x in zip([rec_loss_meter, quant_loss_meter, pp_meter],
                [loss_details[0], loss_details[1], info[0]]): #info[0] is perplexity
                m.update(x.item(), 1)
            
            # update learning rate
            if (self.batch_idx + 1) == self.num_batches:
                self.update_lr()

            meet_freq = (self.batch_idx + 1) % self.assistant.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.assistant.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"loss {rec_loss_meter.val:.4f}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                logger.info(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in zip(["rec_loss", "quant_loss"],
                                   [rec_loss_meter, quant_loss_meter]):
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()
        
        logger.info('epoch: {} '
                        'loss_train: {} '
                        'pp_train: {} '
                        .format(self.epoch + 1, rec_loss_meter.avg, pp_meter.avg)
                        )
        for m, s in zip([rec_loss_meter.avg, quant_loss_meter.avg, pp_meter.avg],
                        ["train/rec_loss", "train/quant_loss", "train/perplexity"]):
            self.write_scalar(s, m, self.epoch + 1)

    def forward_backward(self, batch):
        name, vertices, template, _ = self.parse_batch(batch)
        
        output, quant_loss, info = self.model(vertices, template)
        loss, loss_details = self.criterion(output, 
                                     vertices, 
                                     quant_loss, 
                                     quant_loss_weight=self.assistant.cfg.LOSS.QUANT_LOSS_WEIGHT)
        
        self.model_backward_and_update(loss)
        return loss_details, info
    
    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.assistant.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.assistant.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.assistant.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
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

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.assistant.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        logger.info(f"Evaluate on the *{split}* set")

        rec_loss_meter = AverageMeter()
        quant_loss_meter = AverageMeter()
        pp_meter = AverageMeter()
        for batch_idx, batch in enumerate(data_loader):
            name, vertices, template, _ = self.parse_batch(batch)
            output, quant_loss, info = self.model(vertices, template)
            loss, loss_details = self.criterion(output, 
                                     vertices, 
                                     quant_loss, 
                                     quant_loss_weight=self.assistant.cfg.LOSS.QUANT_LOSS_WEIGHT)

            for m, x in zip([rec_loss_meter, quant_loss_meter, pp_meter],
                [loss_details[0], loss_details[1], info[0]]): #info[0] is perplexity
                m.update(x.item(), 1)
        
        rec_loss_val, quant_loss_val, pp_val = rec_loss_meter.avg, quant_loss_meter.avg, pp_meter.avg
        logger.info('epoch: {} '
                            'loss_val: {} '
                            'pp_val: {} '
                            .format(self.epoch + 1, rec_loss_val, pp_val)
                            )

        for m, s in zip([rec_loss_val, quant_loss_val, pp_val],
                        ["val/rec_loss", "val/quant_loss", "val/perplexity"]):
            self.write_scalar(s, m, self.epoch)


    def parse_batch(self, batch):
        name = batch["name"]
        vertices = batch["vertices"].to(self.device, non_blocking=True)
        template = batch["template"].to(self.device, non_blocking=True)

        if "audio" in batch:
            audio = batch["audio"].to(self.device)
            return name, vertices.float(), template.float(), audio.float()
        
        return name, vertices.float(), template.float(), None

    def get_current_lr(self, names=None):
        names = self.get_model_names(names)
        name = names[0]
        return self._optims[name].param_groups[0]["lr"]