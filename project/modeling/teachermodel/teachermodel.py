from detectron2.modeling.meta_arch import GeneralizedRCNN
from typing import Dict, List
import torch
from typing import Dict, List, Optional, Tuple
from torch import nn
from detectron2.config import configurable
from .build import Teacher_Model_REGISTRY
from ..backbone import build_backbone



@Teacher_Model_REGISTRY.register()
class TeacherModel(GeneralizedRCNN):
    @configurable
    def __init__(self,cfg):
        # 调用GeneralizedRCNN的初始化方法
        super().__init__(cfg.TEACHER)
        

   
    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

       
        return features