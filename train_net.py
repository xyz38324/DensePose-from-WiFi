from project.engine.trainer import MyTrainer
from datetime import timedelta

import detectron2.utils.comm as comm
from detectron2.config import get_cfg
from detectron2.engine import DEFAULT_TIMEOUT, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import verify_results
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger

from densepose import add_densepose_config
from project.engine.add_config import add_custom_config

from densepose.modeling.densepose_checkpoint import DensePoseCheckpointer
def setup(args):
    """
    Create a configuration object from args here.
    """
    cfg = get_cfg()
    add_densepose_config(cfg)
    add_custom_config(cfg)
    
  
    cfg.merge_from_file(args.config_file)
    # cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "densepose" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="densepose")
    return cfg


def main(args):
    cfg = setup(args)
    # disable strict kwargs checking: allow one to specify path handle
    # hints through kwargs, like timeout in DP evaluation
    PathManager.set_strict_kwargs_checking(False)

    if args.eval_only:
        model = MyTrainer.build_model(cfg)
        DensePoseCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = MyTrainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(MyTrainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    cfg = setup(args)
    timeout = (
        DEFAULT_TIMEOUT if cfg.DENSEPOSE_EVALUATION.DISTRIBUTED_INFERENCE else timedelta(hours=4)
    )
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
        timeout=timeout,
    )
