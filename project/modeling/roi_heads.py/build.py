from detectron2.utils.registry import Registry

ROI_2_HEADS_REGISTRY = Registry("ROI_HEADS")
ROI_2_HEADS_REGISTRY.__doc__="""
Registry for ROI heads in a generalized R-CNN model
1. keypoint head
2. densepose head
"""

def build_roi_heads(cfg,input_shape):
    name = cfg.MODEL.ROI_HEADS.NAME
    return ROI_2_HEADS_REGISTRY.get(name)(cfg,input_shape)