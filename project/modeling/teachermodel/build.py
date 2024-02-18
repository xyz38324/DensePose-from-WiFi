from detectron2.utils.registry import Registry
import torch
Teacher_Model_REGISTRY=Registry("Teacher_Model")
Teacher_Model_REGISTRY.__doc__="""

"""


def build_teacher_model(cfg):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    name = cfg.MODEL.TEACHER_MODEL.NAME
    # print("注册表中的模型:", Teacher_Model_REGISTRY)
    model = Teacher_Model_REGISTRY.get(name)(cfg)
    model.to(torch.device(cfg.MODEL.DEVICE))
  
    return model