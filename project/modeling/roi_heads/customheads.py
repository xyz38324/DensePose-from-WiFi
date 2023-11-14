from detectron2.modeling.roi_heads import KRCNNConvDeconvUpsampleHead
from typing import List
from detectron2.structures import Instances
from detectron2.modeling.roi_heads.keypoint_head import keypoint_rcnn_loss,keypoint_rcnn_inference
from detectron2.utils.registry import Registry
ROI_KEYPOINT_HEAD_REGISTRY = Registry("ROI_KEYPOINT_HEAD")
ROI_KEYPOINT_HEAD_REGISTRY.__doc__ = """
Registry for keypoint heads, which make keypoint predictions from per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""


def build_keypoint_head(cfg, input_shape):
    """
    Build a keypoint head from `cfg.MODEL.ROI_KEYPOINT_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_KEYPOINT_HEAD.NAME
    return ROI_KEYPOINT_HEAD_REGISTRY.get(name)(cfg, input_shape)


@ROI_KEYPOINT_HEAD_REGISTRY.register()
class CustomKeypointHead(KRCNNConvDeconvUpsampleHead):
    def forward(self, x, instances: List[Instances]):
        
            
        
        num_images = len(instances)
        normalizer = (
            None if self.loss_normalizer == "visible" else num_images * self.loss_normalizer
        )
      
        return {
            "keypoint_logits": x,  # or apply a softmax here if  want probabilities
            "normalizer":normalizer,
            "loss_weight":self.loss_weight
        }
        