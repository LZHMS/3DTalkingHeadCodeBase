from utils import Registry, check_availability
import logging
logger: logging.Logger

EVALUATOR_REGISTRY = Registry("EVALUATOR")


def build_evaluator(assistant):
    avai_evaluators = EVALUATOR_REGISTRY.registered_names()
    check_availability(assistant.cfg.TEST.EVALUATOR, avai_evaluators)
    if assistant.cfg.ENV.VERBOSE:
        logger.info("Loading evaluator: {}".format(assistant.cfg.TEST.EVALUATOR))
    return EVALUATOR_REGISTRY.get(assistant.cfg.TEST.EVALUATOR)(assistant.cfg)  

class EvaluatorBase:
    """Base evaluator."""

    def __init__(self, assistant):
        self.assistant = assistant

    def reset(self):
        raise NotImplementedError

    def process(self, mo, gt):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError