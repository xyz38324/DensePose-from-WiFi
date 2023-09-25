from detectron2.modeling.backbone import BACKBONE_REGISTRY
from detectron2.layers import get_norm

from .mtn import ModalityTranslationNetwork
                

        
@BACKBONE_REGISTRY.register()
def build_mtn_module(cfg, input_shape):
    model = ModalityTranslationNetwork()
    return model