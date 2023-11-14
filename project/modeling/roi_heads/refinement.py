import torch
import torch.nn as nn
import torch.nn.functional as F

class RefinementUnit(nn.Module):
    def __init__(self, in_channels, intermediate_channels, out_channels):
        super(RefinementUnit, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=intermediate_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=intermediate_channels, out_channels=intermediate_channels, kernel_size=3, padding=1)
        self.fcn = nn.Conv2d(in_channels=intermediate_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.fcn(x)
        return x
    
class CombinedRefinement(nn.Module):
    def __init__(self, dp_channels, kp_channels):
        super(CombinedRefinement, self).__init__()
        # Initialize two separate refinement units
        self.refinement_unit_densepose = RefinementUnit(in_channels=dp_channels + kp_channels, intermediate_channels=64, out_channels=dp_channels)
        self.refinement_unit_keypoint = RefinementUnit(in_channels=dp_channels + kp_channels, intermediate_channels=64, out_channels=kp_channels)

    def forward(self, densepose_output, keypoint_output):
        # Resize keypoint_output to match densepose_output size
        keypoint_output_resized = F.interpolate(keypoint_output, size=densepose_output.shape[2:], mode='bilinear', align_corners=False)#[batch_size, 17, 112, 112]
        # Combine the outputs
        combined_output = densepose_output + keypoint_output_resized

        # Process combined output through each refinement unit
        refined_densepose = self.refinement_unit_densepose(combined_output)
        refined_keypoint = self.refinement_unit_keypoint(combined_output)

        # Resize outputs back to original dimensions
        refined_densepose_resized = F.interpolate(refined_densepose, size=densepose_output.shape[2:], mode='bilinear', align_corners=False)
        refined_keypoint_resized = F.interpolate(refined_keypoint, size=keypoint_output.shape[2:], mode='bilinear', align_corners=False)

        return refined_densepose_resized, refined_keypoint_resized
