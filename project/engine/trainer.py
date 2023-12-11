
from densepose.engine import Trainer
from .customtrainer import CustomTrainer
import logging
from detectron2.engine.defaults import create_ddp_model

from modeling import build_student_model
class MyTrainer(Trainer):
    def __init__(self, cfg):
        # 调用父类的初始化方法
        super().__init__(cfg)
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        model = create_ddp_model(model, broadcast_buffers=False)
        self._trainer = CustomTrainer(model, data_loader, optimizer)
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

        

     
  