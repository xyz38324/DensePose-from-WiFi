from detectron2.engine import DefaultTrainer

class Trainer(DefaultTrainer):
    def build_evaluator():
        pass
    def build_train_loader(cls, cfg: CfgNode):
        pass 
    def build_optimizer(cls, cfg: CfgNode, model: nn.Module):
        pass