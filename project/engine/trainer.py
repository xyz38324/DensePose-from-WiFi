
from densepose.engine import Trainer
from modeling import Combinedodel
import logging
from detectron2.config import CfgNode
from torch import nn
from modeling import build_combined_model
class MyTrainer(Trainer):
    
        
    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:

        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = build_combined_model(cfg)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        return model

        

     
  