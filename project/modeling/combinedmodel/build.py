from detectron2.utils.registry import Registry
Combined_Model_REGISTRY=Registry("Combine_Model")
Combined_Model_REGISTRY.__doc__="""

"""


def build_combined_model(cfg):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    name = cfg.MODEL.COMBINED_MODEL
    model = Combined_Model_REGISTRY.get(name)(cfg)
    model.to(torch.device(cfg.MODEL.DEVICE))
  
    return model