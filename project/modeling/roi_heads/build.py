from detectron2.utils.registry import Registry

KP_DP_RF_HEAD_REGISTRY=Registry("KP_DP_RF_HEAD")
KP_DP_RF_HEAD_REGISTRY.__doc__="""
"""
def build_dp_kp_rf_head(cfg,input_shape):

    name = cfg.MODEL.KP_DP_RF_HEAD.NAME
    return KP_DP_RF_HEAD_REGISTRY.get(name)(cfg, input_shape)