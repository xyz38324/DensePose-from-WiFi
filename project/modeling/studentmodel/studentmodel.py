import torch.nn as nn
import torch
from typing import Dict, List, Optional, Tuple
from .build import Student_Model_REGISTRY
from detectron2.modeling import build_backbone, build_proposal_generator,Backbone
from ..roi_heads import build_roi_heads
from .build import Student_Model_REGISTRY
from detectron2.config import configurable
from ..mtn.build import build_mtn
from ...preprocessing import preprocess
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage

@Student_Model_REGISTRY.register()
class StudentModel(nn.Module):
    """
    Generalized studentdmodel,
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
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
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
        
        
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None


        features = self.backbone(images.tensor)
        """{
            "p2": <tensor of shape [batch_size, out_channels, H/4, W/4]>,
            "p3": <tensor of shape [batch_size, out_channels, H/8, W/8]>,
            "p4": <tensor of shape [batch_size, out_channels, H/16, W/16]>,
            ...
        }
        """
        
        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}
        
        loss_kp,loss_dp = self.roi_heads(images, features, proposals, gt_instances)
      
        losses = {}
        losses.update(loss_kp)
        losses.update(loss_dp)
        losses.update(proposal_losses)
       
        return losses, features
       
       
        

        
        
        
        
        
        
        

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

    @property
    def device(self):
        return self.pixel_mean.device