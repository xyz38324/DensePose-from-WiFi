
# from densepose.engine import Trainer
from .customtrainer import CustomTrainer
import logging
from detectron2.engine.defaults import create_ddp_model
from detectron2.engine import DefaultTrainer
from detectron2.utils import comm
from ..modeling import build_student_model

from ..modeling.studentmodel.studentmodel import Student_Model_REGISTRY,StudentModel
from ..modeling.mtn.mtn import ModalityTranslationNetwork
class MyTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)  # 调用 TrainerBase 的构造函数
        

        self._trainer = CustomTrainer(cfg, self.model, self.data_loader, self.optimizer)

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:

        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = build_student_model(cfg)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        return model

        

     
  