import torch.nn.functional as F

from utils.registry import Registry
from utils.tools import check_availability
from utils.loss import calc_vq_loss, calc_logit_loss, nt_xent_loss

import logging
logger: logging.Logger

EVALUATOR_REGISTRY = Registry("EVALUATOR")


def build_evaluator(cfg, *args, **kwargs):
    avai_evaluators = EVALUATOR_REGISTRY.registered_names()
    check_availability(cfg.EVALUATE.EVALUATOR, avai_evaluators)
    if cfg.ENV.VERBOSE:
        logger.info("Loading evaluator: {}".format(cfg.EVALUATE.EVALUATOR))
    return EVALUATOR_REGISTRY.get(cfg.EVALUATE.EVALUATOR)(cfg.EVALUATE, *args, **kwargs)  

class EvaluatorBase:
    """Base evaluator."""

    def __init__(self, cfg):
        self.cfg = cfg

    def reset(self):
        """
        Reset evaluator state for new evaluation sequence.
        """
        pass

    def process(self, mo, gt):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError
    
    def build_loss_metrics(self, loss_fc_name):
        """
        Build loss function based on configuration name.
        
        Args:
            loss_fc_name (str): Name of the loss function to use.
                               Supported: 'VQLoss', 'LogitLoss', 'NTXentLoss',
                               'L2Loss', 'L1Loss'.
                               
        Returns:
            callable: Loss function to be used for metric computation.
        """
        if loss_fc_name == "VQLoss":
            logger.info("Using VQ loss function for metrics ...")
            return calc_vq_loss
        elif loss_fc_name == "LogitLoss":
            logger.info("Using Logit loss function for metrics ...")
            return calc_logit_loss
        elif loss_fc_name == "NTXentLoss":
            logger.info("Using NT-Xent loss function for metrics ...")
            return nt_xent_loss
        elif loss_fc_name == "L2Loss":
            return F.mse_loss
        elif loss_fc_name == "L1Loss":
            return F.l1_loss
        elif loss_fc_name == "CELoss":
            return F.cross_entropy