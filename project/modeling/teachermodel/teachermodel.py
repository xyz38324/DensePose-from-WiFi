from detectron2.modeling.meta_arch import GeneralizedRCNN
from typing import Dict, List, Optional, Tuple
import torch
from .build import Teacher_Model_REGISTRY

@Teacher_Model_REGISTRY.register()
class TeacherModel(GeneralizedRCNN):
    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

       
        return features