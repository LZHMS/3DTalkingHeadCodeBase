from base.base_evaluator import EVALUATOR_REGISTRY, EvaluatorBase


import logging
logger: logging.Logger


@EVALUATOR_REGISTRY.register()
class ToyEvaluator(EvaluatorBase):
    """
    Evaluator for Toy Model.
    """

    def __init__(self, cfg, device='cpu'):
        """
        Initialize the TDTalkerEvaluator.
        """
        super().__init__(cfg)
        self.cfg, self.device = cfg, device

        # Build loss criterion based on configuration
        self.criterion = self.build_loss_metrics(cfg.LOSS.NAME)
    
    def evaluate(self, x_pre, x_gt):
        return self.criterion(x_pre, x_gt)