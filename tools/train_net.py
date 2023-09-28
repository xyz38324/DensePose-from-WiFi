from engine import Trainer
from detectron2.engine import default_argument_parser, launch,hooks,default_setup
from detectron2.config import get_cfg
from densepose import add_densepose_config

def setup(args):
    # cfg = get_cfg()
    # add_densepose_config(cfg)
    # cfg.merge_from_file(args.config_file)
    # cfg.merge_from_list(args.opts)
    # cfg.freeze()
    # default_setup(cfg,args)
    # return cfg
    pass

def main(args):
    cfg=set(args)
    if args.eval_only:
        pass
    
   
    
    
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    return trainer.train()

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    cfg = setup(args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
        timeout=timeout,

    )