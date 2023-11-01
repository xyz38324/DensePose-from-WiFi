from  project.engine import Trainer
from detectron2.engine import default_argument_parser, launch,hooks,default_setup
from detectron2.config import get_cfg
from densepose import add_densepose_config

def setup(args):
    """
    Create a configuration object from args here.
    """
    cfg = get_cfg()
    add_densepose_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def main(args):
    cfg=set(args)
    if args.eval_only:
        # evaluation code here
        pass
    
   
    
    
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    return trainer.train()

if __name__ == "__main__":
    #trainer  = Trainer(cfg)
    print("XIAOYIZHUO")