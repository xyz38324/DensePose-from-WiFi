from detectron2.utils.registry import Registry

ModalityTranslationNetwork_REGISTRY=Registry("ModalityTranslationNetwork")
ModalityTranslationNetwork_REGISTRY.__doc__="""
Registry for modality_translation_network, which convert 1D csi signal to 3*720*1080 format and input to densepose.

The registered object will be called with `obj(cfg)`.
The call should return a `nn.Module` object.
"""


def build_mtn(cfg):
    
    name = cfg.MODEL.MTN_BACKBONE
    
    
    
    return ModalityTranslationNetwork_REGISTRY.get(name)(cfg)
    
    
    
    