from ..studentmodel import build_student_model
from ..teachermodel import build_teacher_model
import torch 
import torch.nn as nn
import torch.nn.functional as F
from detectron2.config import configurable
from .build import Combined_Model_REGISTRY
@Combined_Model_REGISTRY.register()
class CombinedModel(nn.Module):
    @configurable
    def __init__(self, teacher_model,student_model):
        super(CombinedModel, self).__init__()
        self.teacher_model = teacher_model
        self.student_model = student_model

        # 确保教师模型的参数不会更新
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    @classmethod
    def from_config(cls,cfg):
        teacher_model = build_teacher_model(cfg)
        student_model = build_student_model(cfg)
        return {"teacher_model":teacher_model,"student_model":student_model}

    def forward(self, x):
        # 教师模型的前向传播
        with torch.no_grad():  # 确保不计算梯度
            teacher_features = self.teacher_model(x)#features of fpn res2,3,4,5

        # 学生模型的前向传播
        student_loss,student_features = self.student_model(x)
        transfer_loss = self._calculate_transfer_learning_loss(teacher_features,student_features)
        student_loss.update(transfer_loss)

        # 返回两个模型的输出
        return student_loss
    def _calculate_transfer_learning_loss(teacher_features, student_features):
        loss = 0.0
        for key in ['P2', 'P3', 'P4', 'P5']:
            teacher_feature = teacher_features[key]
            student_feature = student_features[key]
            loss += F.mse_loss(student_feature, teacher_feature)

        return loss