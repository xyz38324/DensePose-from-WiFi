from detectron2.config import CfgNode as CN

def add_custom_config(cfg: CN):
    """
    在 Detectron2 配置中添加自定义配置项
    """
    _C = cfg

    # 基本模型配置
    _C.MODEL.STUDENT_MODEL = ""
    # _C.MODEL.TEACHER_MODEL = "GeneralizedRCNN"
    # _C.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"
    _C.MODEL.BACKBONE.NAME = "build_resnet_fpn_backbone"
    _C.MODEL.RESNETS.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    _C.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    _C.MODEL.ANCHOR_GENERATOR.SIZES = [[32], [64], [128], [256], [512]]
    _C.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]

    _C.MODEL.TEACHER_MODEL=CN()
    _C.MODEL.TEACHER_MODEL.NAME = "TeacherModel"
    # 模态转换网络配置
    _C.MODEL.MTN = CN()
    _C.MODEL.MTN.NAME = "ModalityTranslationNetwork"

    # 细化头网络配置
    _C.MODEL.KP_DP_RF_HEAD = CN()
    _C.MODEL.KP_DP_RF_HEAD.NAME = "Kp_Dp_Refinement_Head"

    # ROI头网络配置
    _C.MODEL.STUDENT_ROI_HEAD = CN()
    _C.MODEL.STUDENT_ROI_HEAD.NAME = "WiFi_ROI_Head"
    
    _C.MODEL.ROI_HEADS.NAME = "DensePoseROIHeads"
    _C.MODEL.ROI_HEADS.IN_FEATURES = ["p2", "p3", "p4", "p5"]
    _C.MODEL.ROI_HEADS.NUM_CLASSES = 1
    _C.MODEL.ROI_BOX_HEAD.NAME = "FastRCNNConvFCHead"
    _C.MODEL.ROI_BOX_HEAD.NUM_FC = 2
    _C.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 7
    _C.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO = 2
    _C.MODEL.ROI_BOX_HEAD.POOLER_TYPE = "ROIAlign"
    _C.MODEL.ROI_DENSEPOSE_HEAD.NAME = "DensePoseV1ConvXHead"
    _C.MODEL.ROI_DENSEPOSE_HEAD.POOLER_TYPE = "ROIAlign"
    _C.MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS = 2

    # RPN配置
    _C.MODEL.RPN.IN_FEATURES = ["p2", "p3", "p4", "p5", "p6"]
    _C.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 2000
    _C.MODEL.RPN.PRE_NMS_TOPK_TEST = 1000
    _C.MODEL.RPN.POST_NMS_TOPK_TRAIN = 1000
    _C.MODEL.RPN.POST_NMS_TOPK_TEST = 1000

    #csi setup
    _C.CSI = CN()
    _C.CSI.MEAN=None
    _C.CSI.STD=None
    # 求解器配置
    _C.SOLVER.IMS_PER_BATCH = 16
    _C.SOLVER.BASE_LR = 0.01
    _C.SOLVER.STEPS = (60000, 80000)
    _C.SOLVER.MAX_ITER = 90000
    _C.SOLVER.WARMUP_FACTOR = 0.1

    # 输入配置
    _C.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)

    # 损失函数配置
    _C.LOSS = CN()
    _C.LOSS.cls = 1.0
    _C.LOSS.box = 1.0
    _C.LOSS.dp = 0.6
    _C.LOSS.kp = 0.3
    _C.LOSS.tr = 0.1

