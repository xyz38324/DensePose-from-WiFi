
# from densepose.engine import Trainer
from .customtrainer import CustomTrainer
import logging,torch
from detectron2.engine.defaults import create_ddp_model
from detectron2.engine import DefaultTrainer
from detectron2.utils import comm
from ..modeling import build_student_model

from ..modeling.studentmodel.studentmodel import Student_Model_REGISTRY,StudentModel
from ..modeling.mtn.mtn import ModalityTranslationNetwork
from ..dataloader.customdataloader import CustomDataset,DatasetMapper,custom_collate_fn
from torch.utils.data import DataLoader
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
# from densepose.data.dataset_mapper import DatasetMapper

class MyTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)  # 调用 TrainerBase 的构造函数
        
       
        self._trainer = CustomTrainer(cfg, self.model, self.data_loader, self.optimizer)


    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:

        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = build_student_model(cfg)
        
        # logger = logging.getLogger(__name__)
        # logger.info("Model:\n{}".format(model))

       
        return model

    @classmethod
    def build_train_loader(cls, cfg):
        images_dir = "/home/visier/mm_fi/MMFi_dataset/all_images"
        annotations ="/home/visier/mm_fi/MMFi_dataset/all_images/kp_dump_results/datasets_anno.json"
        result = utils.build_augmentation(cfg, True)
        random_rotation = T.RandomRotation( [0], expand=False, sample_style="choice"
        )

        result.append(random_rotation)
        mapper = DatasetMapper(cfg,result)
        dataset = CustomDataset(images_dir=images_dir, annotations=annotations,transform=mapper)
        
        # 创建DataLoader实例
        data_loader = DataLoader(dataset,shuffle=True, collate_fn=custom_collate_fn,num_workers=cfg.DATALOADER.NUM_WORKERS,batch_size=2)#cfg.SOLVER.IMS_PER_BATCH)
                
        return data_loader
    

 
