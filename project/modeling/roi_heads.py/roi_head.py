from .build import ROI_2_HEADS_REGISTRY
from torch import _nnpack_available
@ROI_2_HEADS_REGISTRY.register()
class Keypoint_DensePose_Head():
    