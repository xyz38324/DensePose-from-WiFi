from detectron2.utils.registry import Registry

# ROI_2_HEADS_REGISTRY = Registry("ROI_HEADS")
# ROI_2_HEADS_REGISTRY.__doc__="""
# Registry for ROI heads in a generalized R-CNN model
# 1. keypoint head
# 2. densepose head
# """

# def build_roi_heads(cfg,input_shape):
#     name = cfg.MODEL.ROI_HEADS.NAME
#     return ROI_2_HEADS_REGISTRY.get(name)(cfg,input_shape)

ROI_KEYPOINT_HEAD_REGISTRY = Registry("ROI_KEYPOINT_HEAD")
ROI_KEYPOINT_HEAD_REGISTRY.__doc__ = """
Registry for keypoint heads, which make keypoint predictions from per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""



KP_DP_RF_HEAD_REGISTRY=Registry("KP_DP_RF_HEAD")
KP_DP_RF_HEAD_REGISTRY.__doc__="""
"""
def build_dp_kp_rf_head(cfg,input_shape):

    name = cfg.MODEL.KP_DP_RF_HEAD.NAME
    return KP_DP_RF_HEAD_REGISTRY.get(name)(cfg, input_shape)