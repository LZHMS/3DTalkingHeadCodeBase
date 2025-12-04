"""
MNIST Trainer for Handwritten Digit Recognition
"""
import time
import torch
import datetime

from base.base_trainer import TrainerBase, TRAINER_REGISTRY
from dataset.MINIST import MNISTDM
from models.toymodel import ToyModel
from utils.meters import AverageMeter
from base.base_evaluator import build_evaluator
from evaluator.ToyEvaluator import ToyEvaluator

import logging
logger: logging.Logger


@TRAINER_REGISTRY.register()
class ToyTrainer(TrainerBase):
    """Trainer for MNIST handwritten digit recognition"""
    
    def __init__(self, cfg):
        super().__init__(cfg)
        
        # Build components
        self.build_data_loader()
        self.build_model()
        self.evaluator = build_evaluator(cfg, device=self.device)
        
    def build_data_loader(self):
        """Create MNIST data loaders"""
        
        # Build dataset using DataManager (non-infinite mode for epoch training)
        dm = MNISTDM(self.cfg, infinite_train=False)
        
        self.train_loader = dm.train_loader
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.num_batches = len(self.train_loader)
        self.dm = dm
    
    def build_model(self):
        """Build MNIST model"""
        
        logger.info(f"Building model {self.cfg.MODEL.NAME} ...")
        self.model = ToyModel(self.cfg.MODEL.MLP)
        
        # Load pretrained weights if specified
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
        
        # Build optimizer and scheduler
        self.optim = self.build_optimizer(self.model)
        self.sched = self.build_lr_scheduler(self.optim)
        
        # Wrap with DDP if distributed
        if self.is_distributed:
            self.model = self.wrap_model_with_ddp(self.model, find_unused_parameters=False)
        self.register_model("model", self.model, self.optim, self.sched)
  

    def run_epoch(self):
        """Run one training epoch"""
        
        for batch_idx, batch in enumerate(self.train_loader):
            self.data_time.update(time.time() - self.end)
            
            # Forward and backward
            loss, accuracy = self.forward_backward(batch)
            
            # Update meters
            self.batch_time.update(time.time() - self.end)
            self.loss_meter["loss"].update(loss)
            self.acc_meter["accuracy"].update(accuracy)
            
            # Logging
            if (batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0:
                # Calculate ETA
                batches_remaining = (self.max_epoch - self.epoch) * self.num_batches - batch_idx
                eta_seconds = self.batch_time.avg * batches_remaining
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                
                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {self.batch_time.val:.3f} ({self.batch_time.avg:.3f})"]
                info += [f"data {self.data_time.val:.3f} ({self.data_time.avg:.3f})"]
                info += [f"loss {self.loss_meter['loss'].val:.4f} ({self.loss_meter['loss'].avg:.4f})"]
                info += [f"acc {self.acc_meter['accuracy'].val:.2f}% ({self.acc_meter['accuracy'].avg:.2f}%)"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                info = info + [f"rank {self.rank}"] if self.is_distributed else info
                logger.info(" ".join(info))
            
            self.end = time.time()
        
        # Log epoch-level metrics
        current_step = (self.epoch + 1) * self.num_batches
        self.write_scalar("train/loss", self.loss_meter['loss'].avg, current_step)
        self.write_scalar("train/accuracy", self.acc_meter['accuracy'].avg, current_step)
        self.write_scalar("train/lr", self.get_current_lr(), current_step)
        self.write_scalar("train/epoch", self.epoch + 1, current_step, wandb=True)
        
        # Update learning rate (per epoch)
        self.update_lr()
    
    def forward_backward(self, batch):
        """Forward and backward pass"""
        
        # Parse batch
        images, labels = self.parse_batch(batch)
        
        # Forward pass
        outputs = self.model(images)
        loss = self.evaluator.evaluate(outputs, labels)
        
        # Compute accuracy
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == labels).float().mean() * 100
        
        self.model_backward_and_update(loss)
        
        return loss.item(), accuracy.item()
    
    @torch.no_grad()
    def test(self, split=None):
        """Test/validation pipeline"""
        
        self.set_model_mode("eval")        
        # Select data loader
        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split, data_loader = "test", self.test_loader
        
        logger.info(f"Evaluate on the *{split}* set")
        loss_meter = AverageMeter()
        accuracy_meter = AverageMeter()
        
        # Confusion matrix for detailed analysis
        num_classes = 10
        confusion_matrix = torch.zeros(num_classes, num_classes)
        for batch_idx, batch in enumerate(data_loader):
            images, labels = self.parse_batch(batch)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.evaluator.evaluate(outputs, labels)
            
            # Compute accuracy
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == labels).float().mean() * 100
            
            # Update meters
            loss_meter.update(loss.item(), images.size(0))
            accuracy_meter.update(accuracy.item(), images.size(0))
            
            # Update confusion matrix
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
        
        # Log results
        logger.info(f"{'='*60}")
        logger.info(f"{split.upper()} Results:")
        logger.info(f"  Loss: {loss_meter.avg:.4f}")
        logger.info(f"  Accuracy: {accuracy_meter.avg:.2f}%")
        
        # Per-class accuracy
        per_class_acc = confusion_matrix.diag() / confusion_matrix.sum(1)
        logger.info(f"  Per-class Accuracy:")
        for i in range(num_classes):
            logger.info(f"    Digit {i}: {per_class_acc[i]*100:.2f}%")
        logger.info(f"{'='*60}")
        
        # TensorBoard logging
        current_step = (self.epoch + 1) * self.num_batches
        self.write_scalar(f"{split}/loss", loss_meter.avg, current_step)
        self.write_scalar(f"{split}/accuracy", accuracy_meter.avg, current_step)
        
        # Return to train mode
        self.set_model_mode("train")
        
        return accuracy_meter.avg
    
    def parse_batch(self, batch):
        """Parse batch data"""
        
        images = batch['image'].to(self.device)
        labels = batch['label'].to(self.device)
        
        return images, labels
