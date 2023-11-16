import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.modeling.roi_heads.keypoint_head import keypoint_rcnn_loss,keypoint_rcnn_inference
from densepose.modeling import densepose_inference
class RefinementUnit(nn.Module):
    def __init__(self, in_channels, intermediate_channels, out_channels):
        super(RefinementUnit, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=intermediate_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=intermediate_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.fcn = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.fcn(x)
        return x

class CombinedRefinement(nn.Module):
    def __init__(self,dp_channels, kp_channels,training,normalizer_kp,lossweight_kp,raw_instance_kp,densepose_loss_func,raw_instance_dp,embedder_dp):
        super(CombinedRefinement, self).__init__()
        # Separate refinement units for DensePose and Keypoint
        self.refinement_unit_dp = RefinementUnit(in_channels=dp_channels + kp_channels, intermediate_channels=64, out_channels=dp_channels)
        self.refinement_unit_kp = RefinementUnit(in_channels=dp_channels + kp_channels, intermediate_channels=64, out_channels=kp_channels)
        self.training=training
        self.raw_instance_kp = raw_instance_kp
        self.normalizer = normalizer_kp
        self.loss_weight = lossweight_kp
        self.densepose_loss_func = densepose_loss_func

        self.raw_instance_dp=raw_instance_dp
        self.embedder = embedder_dp

    def forward(self, instance_dp, instance_kp):
        # Downsample instance_dp to match instance_kp size
        instance_dp_resized = F.interpolate(instance_dp, size=instance_kp.shape[2:], mode='bilinear', align_corners=False)
        # Concatenate downsampled instance_dp with instance_kp
        combined_dp = torch.cat([instance_dp_resized, instance_kp], dim=1)
        # Process through refinement unit for DensePose
        refined_dp = self.refinement_unit_dp(combined_dp)

        # Upsample instance_kp to match instance_dp size
        instance_kp_resized = F.interpolate(instance_kp, size=instance_dp.shape[2:], mode='bilinear', align_corners=False)
        # Concatenate upsampled instance_kp with instance_dp
        combined_kp = torch.cat([instance_dp, instance_kp_resized], dim=1)
        # Process through refinement unit for Keypoint
        refined_kp = self.refinement_unit_kp(combined_kp)
        if self.training==True:
            return{

                "loss_keypoint":keypoint_rcnn_loss(refined_kp,self.raw_instance,self.normalizer)*self.loss_weight,
                "loss_densepose":self.densepose_loss_func(self.raw_instance_dp,refined_dp,embedder=self.embedder)
            }
            
        
        else:

            keypoint_rcnn_inference(refined_kp,self.raw_instance_kp)
            densepose_inference(refined_dp,self.raw_instance_dp)
            return {self.raw_instance_kp,self.raw_instance_dp}
        

        
