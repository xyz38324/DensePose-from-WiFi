from detectron2.utils.registry import Registry
Combined_Model_REGISTRY=Registry("Combine_Model")
Combined_Model_REGISTRY.__doc__="""
Registry for Combined_Model, which contains ModalityTranslationNetwork and Densepose architecture,
i.e. the whole model.

The registered object will be called with `obj(cfg)`.
The call should return a `nn.Module` object.
"""


def build_student_model(cfg):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    name = cfg.MODEL.META_ARCHITECTURE
    model = Combined_Model_REGISTRY.get(name)(cfg)
    model.to(torch.device(cfg.MODEL.DEVICE))
  
    return model