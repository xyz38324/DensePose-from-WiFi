from .build import build_dp_kp_rf_head
from typing import Dict, List, Optional
from torch import nn
import torch
from detectron2.structures import ImageList, Instances
from detectron2.modeling.roi_heads import select_foreground_proposals
from densepose.modeling.roi_heads import Decoder
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import ROIHeads
from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.modeling.roi_heads import build_box_head
from detectron2.modeling.roi_heads.roi_heads import select_proposals_with_visible_keypoints
from detectron2.modeling import FastRCNNOutputLayers
from .build import build_dp_kp_rf_head,ROI_HEAD_REGISTRY


from densepose.modeling import build_densepose_data_filter
  
    


@ROI_HEAD_REGISTRY.register()
class WiFi_ROI_Head(ROIHeads):
    @configurable
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        self._init_box_head(cfg, input_shape)
        self._init_kp_dp_rf_head(cfg, input_shape)
        self.densepose_data_filter = build_densepose_data_filter(cfg)
        self.use_decoder           = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECODER_ON
        self.decoder = Decoder(cfg, input_shape, self.in_features)

   
    def _init_kp_dp_rf_head(self,cfg,input_shape):
        self.kp_dp_rf_head = build_dp_kp_rf_head(cfg,input_shape)
      
    
    def _init_box_head(self, cfg, input_shape):
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.in_features = in_features
        self.pooler = box_pooler
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        box_predictor = FastRCNNOutputLayers(cfg, box_head.output_shape)
        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
        }


    def _forward_box(self, features: Dict[str, torch.Tensor], proposals: List[Instances]):
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.box_in_features]
        box_features = self.shared_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        del box_features

        if self.training:
            losses = self.box_predictor.losses(predictions, proposals)
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances
    
    
    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ):
        
        del images
        if self.training:
            assert targets, "'targets' argument is required during training"
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets


        # features_list = [features[f] for f in self.in_features]
        if self.training:
            #box_head
            losses = self._forward_box(features,proposals)
            
            #keypoint
            instances_kp,_ =select_foreground_proposals(proposals,self.num_classes)
            instances_kp = select_proposals_with_visible_keypoints(instances_kp)
            features_kp = [features[f] for f in self.in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances_kp]
            features_kp = self.pooler(features_kp,boxes)

            #densepose (features,proposals)
            features_list = [features[f] for f in self.in_features]
            proposals_dp, _ = select_foreground_proposals(proposals, self.num_classes)
            features_list, proposals_dp = self.densepose_data_filter(features_list, proposals_dp)
            if len(proposals_dp) > 0:
                proposal_boxes = [x.proposal_boxes for x in proposals_dp]

                if self.use_decoder:
                    features_list = [self.decoder(features_list)]

                features_dp = self.pooler(features_list, proposal_boxes)
            del targets, images

            instances_densepose_keypoint,losses = self.kp_dp_rf_head(features_kp,features_dp)
        return instances_densepose_keypoint,losses


 
    def forward_with_given_boxes(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ):
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (list[Instances]):
                the same `Instances` objects, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """

        instances = super().forward_with_given_boxes(features, instances)
        instances = self._forward_densepose(features, instances)
        return instances
    

