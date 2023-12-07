import torch.nn as nn
import torch
from typing import Dict, List, Optional, Tuple
from .build import Student_Model_REGISTRY
from detectron2.modeling import  build_proposal_generator,Backbone
from ..backbone import build_backbone
from .build import Student_Model_REGISTRY
from detectron2.config import configurable
from ..mtn.build import build_mtn

from detectron2.structures import ImageList,Instances

from ..roi_heads import build_dp_kp_rf_head
from ..teachermodel import build_teacher_model
import torch.nn.functional as F
from detectron2.modeling import GeneralizedRCNN
@Student_Model_REGISTRY.register()
class StudentModel(GeneralizedRCNN):
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
        teacher_model:nn.Module,
    ):
        super().__init__()
        self.mtn =  mtn      
        self.backbone = backbone
        self.roi_heads = roi_heads

        self.input_format = input_format
       

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"
      
        self.register_buffer("csi_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("csi_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        #the shape of this attribute need to change later!!!!
        self.teacher_model = teacher_model
        for param in self.teacher_model.parameters():
            param.requires_grad=False         #fix weight of teacher model
       
    @classmethod
    def from_config(cls,cfg):
        backbone = build_backbone(cfg)
        teacher_model= build_teacher_model(cfg)
        return {
            "mtn":build_mtn,
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_dp_kp_rf_head(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "csi_mean":cfg.CSI.MEAN,
            "csi_std":cfg.CSI_STD,
            "teacher_model":teacher_model,
        }
     
    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        batched_inputs: a list,
                * csi 
                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.
        """
        
        if not self.training:
            return self.inference(batched_inputs)
        
        csi = [x["csi"].to(self.device) for x in batched_inputs]
        
        mtn_output = self.mtn(csi)
        assert mtn_output.shape == (3, 720, 1080), f"Unexpected output shape: {mtn_output.shape}"

   
        images = self.preprocess_image(batched_inputs)# normalization
        # csi_images = self.preprocess_image(mtn_output)# normalization
        
        
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None


        features = self.backbone(mtn_output)
        features_teacher = self.teacher_model(images.tensor)
        transfer_loss = self._calculate_transfer_learning_loss(features_teacher,features)
        """{
            "p2": <tensor of shape [batch_size, out_channels, H/4, W/4]>,
            "p3": <tensor of shape [batch_size, out_channels, H/8, W/8]>,
            "p4": <tensor of shape [batch_size, out_channels, H/16, W/16]>,
            ...
        }
        """
        
        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(mtn_output, features, gt_instances)
        else:
            assert "proposals" in csi_images[0]
            proposals = [x["proposals"].to(self.device) for x in csi_images]
            proposal_losses = {}
        
        _,detector_losses = self.roi_heads(csi_images, features, proposals, gt_instances)
      
        losses = {}
        
        losses.update(transfer_loss)
        losses.update(proposal_losses)
        losses.update(detector_losses)
       
        return losses, features
       
       
        

        
        
        
        
    def _calculate_transfer_learning_loss(teacher_features, student_features):
        loss = 0.0
        for key in ['P2', 'P3', 'P4', 'P5']:
            teacher_feature = teacher_features[key]
            student_feature = student_features[key]
            loss += F.mse_loss(student_feature, teacher_feature)

        return loss
        
        

    def preprocess_csi(self,batch_inputs:List[Dict[str,torch.Tensor]]):#input should be batch or single rgb data?
        mean = self.csi_mean
        std = self.csi_std
        #return data should be tensor shape
        return csi
        pass
    

     
    def inference(
        self,
        batched_mtn: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        assert not self.training

        images = self.preprocess_image(batched_mtn)