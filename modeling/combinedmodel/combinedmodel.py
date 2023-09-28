import torch.nn as nn

from .build import Combined_Model_REGISTRY
from detectron2.modeling import build_backbone, build_proposal_generator,build_roi_heads,Backbone
from densepose.modeling import build_densepose_head
from detectron2.config import configurable
from ..mtn.build import build_mtn
@Combined_Model_REGISTRY.register()
class CombinedModel(nn.Module):
    """
    Generalized combinedmodel,
    1: ModilityTranslationNetwork, convert csi data to 3*720*1080 format and feed into densepose network
    2: DensePose model, Generalized R-CNN from detectron2 project
        Per-image feature extraction (aka backbone)
        Region proposal generation
        Per-region feature extraction and prediction
    """
    @configurable
    def __init__(
       
        
        self,
        *,
        mtn:nn.Module,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
    ):
        super().__init__()
        self.mtn =  mtn      
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"
      
        
        
        self.modality_translation_network = mtn
        self.backbone = build_backbone(cfg)  # or build_resnet_backbone(cfg) depending on your needs
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        
        self.densepose_head = build_densepose_head(cfg)
       
       
    @classmethod
    def from_config(cls,cfg):
        backbone = build_backbone(cfg)
        return {
            "mtn":build_mtn,
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }
     
    def forward(self, batched_inputs):
        csi_data = [x["csi"] for x in batched_inputs]
        images = self.modality_translation_network(csi_data)
        features = self.backbone(images.tensor)
        proposals, _ = self.proposal_generator(images, features)
        #results, _, _ = self.roi_heads(images, features, proposals)
        densepose_outputs = self.densepose_head(features, results)
        return densepose_outputs

