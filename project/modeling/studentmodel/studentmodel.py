import torch.nn as nn
import torch
from typing import Dict, List, Optional, Tuple
from .build import Student_Model_REGISTRY
from detectron2.modeling import  build_proposal_generator,Backbone
from detectron2.modeling.backbone import Backbone,build_backbone
from detectron2.config import configurable
from ..mtn import build_mtn
from detectron2.layers import move_device_like
from detectron2.structures import ImageList,Instances
import torch.nn.functional as F
from ..roi_heads import build_roi_head
from detectron2.modeling import build_model
import torch.nn.functional as F
# from detectron2.modeling import GeneralizedRCNN
from detectron2.utils.events import get_event_storage
from ..teachermodel import build_teacher_model

__all__ = ["StudentModel"]
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
        input_format: Optional[str] = None,
        vis_period: int = 0,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # csi_mean:None,
        # csi_std:None,
        teacher_model:nn.Module,
    ):  
        super().__init__()

        self.mtn =  mtn      
        self.backbone = backbone
        self.proposal_generator = proposal_generator
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
            "mtn":build_mtn(cfg),
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),            
            "roi_heads": build_roi_head(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # "csi_mean":cfg.CSI.MEAN,
            # "csi_std":cfg.CSI.STD,
            "teacher_model":teacher_model,
        }
     
    def forward(self, batched_inputs):
        """
        batched_inputs: a dic,
                * csi 
                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.
        """
        
        if not self.training:
            return self.inference(batched_inputs)
        
        
        
        csi_phase = [x["csi"]['phase'].to(self.device) for x in batched_inputs]
        csi_phase_tensor = torch.stack(csi_phase)
        csi_amp = [x["csi"]['amp'].to(self.device) for x in batched_inputs]
        csi_amp_tensor = torch.stack(csi_amp)
       
        features_teacher = self.teacher_model(batched_inputs)
        mtn_output = self.mtn(csi_amp_tensor,csi_phase_tensor)
        
        output_list=[]
        for i in range(mtn_output.shape[0]):
            current_tensor = mtn_output[i]
            output_list.append(current_tensor.squeeze(0))

        mtn_image = self.preprocess_mtn(output_list)

        
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

       
        features = self.backbone(mtn_image.tensor)
       
        
        transfer_loss = self._calculate_transfer_learning_loss(features_teacher,features)
        """{
            "p2": <tensor of shape [batch_size, out_channels, H/4, W/4]>,
            "p3": <tensor of shape [batch_size, out_channels, H/8, W/8]>,
            "p4": <tensor of shape [batch_size, out_channels, H/16, W/16]>,
            ...
        }
        """
        
        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(mtn_image, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}
        
        detector_losses = self.roi_heads(mtn_image, features, proposals, gt_instances)

        
        losses = {}
        
        losses.update({'loss_transfer': transfer_loss})
        losses.update(proposal_losses)
        losses.update(detector_losses)
        
       
        return losses
       
   
    
    def preprocess_mtn(self, mtn_output: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
       

        images = ImageList.from_tensors(
            mtn_output,
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )
        return images 
        
    @property
    def device(self):
        return self.pixel_mean.device
    def _move_to_current_device(self, x):
        return move_device_like(x, self.pixel_mean)
        
        
    def _calculate_transfer_learning_loss(self,teacher_features, student_features):
        loss = 0.0
        for key in ['p5','p4','p3', 'p2']:
            teacher_feature = teacher_features[key]
            
            student_feature = student_features[key]
            teacher_feature_downsampled = F.interpolate(teacher_feature, size=student_feature.shape[-2:], mode='nearest')
            if torch.isnan(teacher_feature_downsampled).any() or torch.isnan(student_feature).any():
                print("exist Nan")
            loss += F.mse_loss(student_feature, teacher_feature_downsampled)

        return loss
        
        



     
    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training


        csi = [x["csi"].to(self.device) for x in batched_inputs]
        
        mtn_output = self.mtn(csi)

       
        features = self.backbone(mtn_output)

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(mtn_output, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(mtn_output, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

       
        return results