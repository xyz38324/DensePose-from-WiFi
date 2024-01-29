from detectron2.modeling.roi_heads import KRCNNConvDeconvUpsampleHead
from detectron2.layers import Conv2d, ConvTranspose2d
from detectron2.modeling.roi_heads.keypoint_head import keypoint_rcnn_loss,keypoint_rcnn_inference
import torch.nn as nn
from densepose.modeling import build_densepose_head,build_densepose_losses,build_densepose_predictor,build_densepose_embedder
from densepose.modeling import densepose_inference
from detectron2.config import configurable
from .build import KP_DP_RF_HEAD_REGISTRY

@KP_DP_RF_HEAD_REGISTRY.register()
class Kp_Dp_Refinement_Head(KRCNNConvDeconvUpsampleHead):
    @configurable
    def __init__(
        self,
        *,
        input_shape,
        densepose_head,
        densepose_predictor,
        num_keypoints,
        conv_dims,
        densepose_loss,
        embedder,
        **kwargs
    ):
        # 首先调用父类的构造函数
        super().__init__(input_shape, num_keypoints=num_keypoints, conv_dims=conv_dims, **kwargs)

        # 添加您自己的初始化逻辑
        
        in_channels = input_shape.channels
        self.conv_layers_kp= nn.Sequential(
            Conv2d(in_channels, in_channels, 3, stride=1, padding=1),
            Conv2d(in_channels, in_channels, 3, stride=1, padding=1)
        )
        self.conv_layers_dp = nn.Sequential(
            Conv2d(in_channels, in_channels, 3, stride=1, padding=1),
            Conv2d(in_channels, in_channels, 3, stride=1, padding=1)
        )
        self.densepose_head=densepose_head
        self.densepose_predictor = densepose_predictor
        self.densepose_losses = densepose_loss
        self.embedder  = embedder
    @classmethod
    def from_config(cls, cfg, input_shape):
        
        in_channels = [input_shape[f].channels for f in self.in_features][0]
    # 提取所有需要的配置参数
        densepose_head = build_densepose_head(cfg,in_channels)
        densepose_predictor = build_densepose_predictor(cfg,densepose_head.n_out_channels)
        densepose_loss = build_densepose_losses(cfg)
        embedder = build_densepose_embedder(cfg)
        ret = {
            "input_shape": input_shape,
            "num_keypoints": cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS,
            "conv_dims": cfg.MODEL.ROI_KEYPOINT_HEAD.CONV_DIMS,
            "densepose_head": densepose_head,
            "densepose_predictor": densepose_predictor,
            "densepose_losses":densepose_loss,
            "embedder": embedder,
            # 其他需要的配置参数
        }
        return ret

    def forward(self,x,instances_keypoint_densepose):
        x_kp = self.layers(x)
        x_dp = self.densepose_head(x)

        merge  = x_dp + x_kp
        refinement_keypoint = self.conv_layers_kp(merge)
        refinement_densepose = self.conv_layers_dp(merge)

        
        if self.training:
            num_images = len(instances_keypoint_densepose)
            normalizer = (
                None if self.loss_normalizer == "visible" else num_images * self.loss_normalizer
            )
            keypoint_predictor_output = self.score_lowres(refinement_keypoint)

            densepose_predictor_output = self.densepose_predictor(refinement_densepose)

            densepose_loss_dict = self.densepose_losses(instances_keypoint_densepose,densepose_predictor_output,embedder=self.embedder)
            losses = keypoint_rcnn_loss(keypoint_predictor_output, instances_keypoint_densepose, normalizer=normalizer)* self.loss_weight
            losses.update(densepose_loss_dict)
            return losses
        else:

            if len(x)>0:
                densepose_predictor_output=self.densepose_predictor(refinement_densepose)
                keypoint_predictor_output = self.score_lowres(refinement_keypoint)
            else:
                densepose_predictor_output=None

            keypoint_rcnn_inference(keypoint_predictor_output, instances_keypoint_densepose)
            densepose_inference(densepose_predictor_output,instances_keypoint_densepose)
            return instances_keypoint_densepose

    def layers(self, x):

       
        # 重写layers方法以包含新的逻辑
        for layer in self.children():
            if isinstance(layer, ConvTranspose2d):
                break
            x = layer(x)
        return x
            
     

       
