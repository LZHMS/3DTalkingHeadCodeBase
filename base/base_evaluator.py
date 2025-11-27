from utils import Registry, check_availability
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
        raise NotImplementedError

    def process(self, mo, gt):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError