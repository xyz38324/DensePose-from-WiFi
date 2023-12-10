from detectron2.modeling.meta_arch import GeneralizedRCNN
from typing import Dict, List
import torch
from typing import Dict, List, Optional, Tuple
from torch import nn
from detectron2.config import configurable
from .build import Teacher_Model_REGISTRY
from ..backbone import build_backbone

from detectron2.modeling import build_proposal_generator,build_roi_heads,Backbone
@Teacher_Model_REGISTRY.register()
class TeacherModel(GeneralizedRCNN):
    @configurable
    def __init__(self,cfg):
        # 调用GeneralizedRCNN的初始化方法
        super().__init__(
            backbone=build_backbone(cfg),
            proposal_generator=build_proposal_generator(cfg, backbone.output_shape()),
            roi_heads=build_roi_heads(cfg, backbone.output_shape()),
            pixel_mean=cfg.MODEL.PIXEL_MEAN,
            pixel_std=cfg.MODEL.PIXEL_STD,
            input_format=cfg.INPUT.FORMAT,
            vis_period=cfg.VIS_PERIOD,
        )
        

   
    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

       
        return features