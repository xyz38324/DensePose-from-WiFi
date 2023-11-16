
from densepose.engine import Trainer
from ..modeling import build_student_model


from detectron2.config import CfgNode
from torch import nn
class MyTrainer(Trainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.student_model=build_student_model(cfg)
        for param in self.m
     
  