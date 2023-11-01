import torch.nn as nn
import torch
from typing import Dict, List, Optional, Tuple
from .build import Combined_Model_REGISTRY
from detectron2.modeling import build_backbone, build_proposal_generator,build_roi_heads,Backbone
from densepose.modeling import build_densepose_head
from detectron2.config import configurable
from ..mtn.build import build_mtn
from ...preprocessing import preprocess
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage

@Combined_Model_REGISTRY.register()
class CombinedModel(nn.Module):
    """
    Generalized combinedmodel,
    1: ModilityTranslationNetwork, convert csi data to 3*720*1080 format and feed into densepose network
    2: DensePose model, Generalized R-CNN from detectron2 project
        Per-image feature extraction (aka backbone)
        Region proposal generation
        Per-region feature extraction and prediction
    """
    @configurable
    def __init__(
       
        
        self,
        *,
        mtn:nn.Module,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        #vis_period: int = 0,
        csi_mean:Tuple[float],
        csi_std:Tuple[float],
    ):
        super().__init__()
        self.mtn =  mtn      
        self.backbone = backbone
        self.roi_heads = roi_heads

        self.input_format = input_format
        #self.vis_period = vis_period
        # if vis_period > 0:
        #     assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"
      
        self.register_buffer("csi_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("csi_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        #the shape of this attribute need to change later!!!!
  
       
       
    @classmethod
    def from_config(cls,cfg):
        backbone = build_backbone(cfg)
        return {
            "mtn":build_mtn,
            "backbone": backbone,
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "csi_mean":cfg.CSI.MEAN,
            "csi_std":cfg.CSI_STD,
        }
     
    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)
        csi = self.preprocess_csi(batched_inputs)#process csi to proper shape 
        mtn_output = self.mtn(csi)
        assert mtn_output.shape == (3, 720, 1080), f"Unexpected output shape: {mtn_output.shape}"
        
        images = self.preprocess_image(mtn_output)# normalization
        
        
        #gt_instances = None
        features = self.backbone(images.tensor)
        assert "proposals" in batched_inputs[0]
        proposals = [x["proposals"].to(self.device) for x in batched_inputs]
        proposal_losses = {}
        
        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
      
        losses = {}
        losses.update(detector_losses)
       
        return losses
       
       
        

        
        
        
        
        
        
        

    def preprocess_csi(self,batch_inputs:List[Dict[str,torch.Tensor]]):#input should be batch or single rgb data?
        mean = self.csi_mean
        std = self.csi_std
        #return data should be tensor shape
        return csi
        pass
    
    def preprocess_image(self,csi_rgb: torch.Tensor):#
        assert csi_rgb.shape == (3, 720, 1080)
        images = ImageList.from_tensors(
            csi_rgb,
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,#think about it later 
        )
        return images
        
    
    def inference(self,):
        pass