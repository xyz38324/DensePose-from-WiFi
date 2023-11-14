from .build import ROI_2_HEADS_REGISTRY
from typing import Dict, List, Optional
from torch import nn
import torch
from detectron2.structures import ImageList, Instances
from .customheads import build_keypoint_head
from detectron2.modeling.roi_heads import select_foreground_proposals
from densepose.modeling.roi_heads import Decoder
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import ROIHeads
from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.modeling.roi_heads import build_box_head
from detectron2.modeling.roi_heads.roi_heads import select_proposals_with_visible_keypoints
from detectron2.modeling import FastRCNNOutputLayers
from .customheads import build_keypoint_head
from densepose.modeling import (
    build_densepose_data_filter,
    build_densepose_embedder,
    build_densepose_head,
    build_densepose_losses,
    build_densepose_predictor,
    densepose_inference,
    
)


class WiFi_ROI_Head(ROIHeads):
    @configurable
    def __init__(
        self,
        cfg,
        input_shape,
        box_in_features: List[str],
        box_pooler: ROIPooler,
        box_head: nn.Module,
        box_predictor: nn.Module,
        keypoint_in_features: Optional[List[str]] = None,
        keypoint_pooler: Optional[ROIPooler] = None,
        keypoint_head: Optional[nn.Module] = None,
        train_on_pred_boxes: bool = False,
    ):
        super.__init__(cfg,input_shape)

        self.in_features = self.box_in_features = box_in_features
        self.box_pooler = box_pooler
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.keypoint_in_features = keypoint_in_features
        self.keypoint_pooler = keypoint_pooler
        self.keypoint_head = keypoint_head
        self._init_densepose_head(cfg,input_shape)
        self._init_keypoint_head(cfg,input_shape)

        self.train_on_pred_boxes = train_on_pred_boxes
    @classmethod
    def from_config(cls, cfg, input_shape):
        ret =  super().from_config(cfg)
        ret["train_on_pred_boxes"] = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        ret.update(cls._init_box_head(cfg, input_shape))
        ret.update(cls._init_keypoint_head(cfg, input_shape))

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
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

    @classmethod
    def _init_keypoint_head(cls, cfg, input_shape):
            if not cfg.MODEL.KEYPOINT_ON:
                return {}
            # fmt: off
            in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
            pooler_resolution = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION
            pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)  # noqa
            sampling_ratio    = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO
            pooler_type       = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_TYPE
            # fmt: on

            in_channels = [input_shape[f].channels for f in in_features][0]

            ret = {"keypoint_in_features": in_features}
            ret["keypoint_pooler"] = (
                ROIPooler(
                    output_size=pooler_resolution,
                    scales=pooler_scales,
                    sampling_ratio=sampling_ratio,
                    pooler_type=pooler_type,
                )
                if pooler_type
                else None
            )
            if pooler_type:
                shape = ShapeSpec(
                    channels=in_channels, width=pooler_resolution, height=pooler_resolution
                )
            else:
                shape = {f: input_shape[f] for f in in_features}
            ret["keypoint_head"] = build_keypoint_head(cfg, shape)
            return ret

    def _init_densepose_head(self, cfg, input_shape):
        # fmt: off
        self.densepose_on          = cfg.MODEL.DENSEPOSE_ON
        if not self.densepose_on:
            return
        self.densepose_data_filter = build_densepose_data_filter(cfg)
        dp_pooler_resolution       = cfg.MODEL.ROI_DENSEPOSE_HEAD.POOLER_RESOLUTION
        dp_pooler_sampling_ratio   = cfg.MODEL.ROI_DENSEPOSE_HEAD.POOLER_SAMPLING_RATIO
        dp_pooler_type             = cfg.MODEL.ROI_DENSEPOSE_HEAD.POOLER_TYPE
        self.use_decoder           = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECODER_ON
        # fmt: on
        if self.use_decoder:
            dp_pooler_scales = (1.0 / input_shape[self.in_features[0]].stride,)
        else:
            dp_pooler_scales = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        in_channels = [input_shape[f].channels for f in self.in_features][0]

        if self.use_decoder:
            self.decoder = Decoder(cfg, input_shape, self.in_features)

        self.densepose_pooler = ROIPooler(
            output_size=dp_pooler_resolution,
            scales=dp_pooler_scales,
            sampling_ratio=dp_pooler_sampling_ratio,
            pooler_type=dp_pooler_type,
        )
        self.densepose_head = build_densepose_head(cfg, in_channels)
        self.densepose_predictor = build_densepose_predictor(
            cfg, self.densepose_head.n_out_channels
        )
        self.densepose_losses = build_densepose_losses(cfg)
        self.embedder = build_densepose_embedder(cfg)

    def _forward_densepose(self, features: Dict[str, torch.Tensor], instances: List[Instances]):
        """
        Forward logic of the densepose prediction branch.

        Args:
            features (dict[str, Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            instances (list[Instances]): length `N` list of `Instances`. The i-th
                `Instances` contains instances for the i-th input image,
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "densepose" and return it.
        """
        if not self.densepose_on:
            return {} if self.training else instances

        features_list = [features[f] for f in self.in_features]
        if self.training:
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            features_list, proposals = self.densepose_data_filter(features_list, proposals)
            if len(proposals) > 0:
                proposal_boxes = [x.proposal_boxes for x in proposals]

                if self.use_decoder:
                    features_list = [self.decoder(features_list)]

                features_dp = self.densepose_pooler(features_list, proposal_boxes)
                densepose_head_outputs = self.densepose_head(features_dp)
                densepose_predictor_outputs = self.densepose_predictor(densepose_head_outputs)
                densepose_loss_dict = self.densepose_losses(
                    proposals, densepose_predictor_outputs, embedder=self.embedder
                )
                return densepose_predictor_outputs,densepose_loss_dict
        else:
            pred_boxes = [x.pred_boxes for x in instances]

            if self.use_decoder:
                features_list = [self.decoder(features_list)]

            features_dp = self.densepose_pooler(features_list, pred_boxes)
            if len(features_dp) > 0:
                densepose_head_outputs = self.densepose_head(features_dp)
                densepose_predictor_outputs = self.densepose_predictor(densepose_head_outputs)
            else:
                densepose_predictor_outputs = None

            densepose_inference(densepose_predictor_outputs, instances)
            return instances

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
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
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

        if self.training:

            instance_kp,loss_kp=self._forward_keypoint(features, proposals)
            del targets, images
            instance_dp,loss_dp = self._forward_densepose(features,proposals) 

            



            
            
        
        return instances, losses
      

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
    def _forward_keypoint(self, features: Dict[str, torch.Tensor], instances: List[Instances]):
        """
        Forward logic of the keypoint prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict keypoints.
                In training, they can be the proposals.
                In inference, they can be the boxes predicted by R-CNN box head.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_keypoints" and return it.
        """
        if not self.keypoint_on:
            return {} if self.training else instances

        if self.training:
            # head is only trained on positive proposals with >=1 visible keypoints.
            instances, _ = select_foreground_proposals(instances, self.num_classes)
            instances = select_proposals_with_visible_keypoints(instances)

        if self.keypoint_pooler is not None:
            features = [features[f] for f in self.keypoint_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.keypoint_pooler(features, boxes)
        else:
            features = {f: features[f] for f in self.keypoint_in_features}
        return self.keypoint_head(features, instances)


