from detectron2.engine import DefaultTrainer
from ..modeling import build_model
import logging
from detectron2.config import CfgNode
from torch import nn
class Trainer(DefaultTrainer):

     
     
    @classmethod
    def build_model(cls, cfg):
        
        model = build_model(cfg)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        return model
    def build_evaluator():
        pass
    def build_train_loader(cls, cfg: CfgNode):
        pass 
    def build_optimizer(cls, cfg: CfgNode, model: nn.Module):
        pass
