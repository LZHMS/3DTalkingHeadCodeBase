import argparse

from base import build_trainer
from config import CodeTalkerConfig, TrainerConfig
from trainers import StyleEncoderTrainer
import warnings
warnings.filterwarnings('ignore')
import wandb

def merge_args(assistant, args):
    if args.gpu:
        assistant.cfg.ENV.GPU = args.gpu

    if args.use_wandb:
        assistant.cfg.ENV.USE_WANDB = args.use_wandb

def main(args):
    assistant = TrainerConfig(args.config_file)
    # From command line arguments
    merge_args(assistant, args)
    # From optional input arguments
    assistant.cfg.merge_from_list(args.opts)
    
    assistant.cfg.freeze()
    assistant.print_info()

    # setup wandb
    if args.use_wandb:
        extra_config = {"NTXent_Temperature": assistant.cfg.LOSS.CONTRASTIVE.TEMPRATURE}
        assistant.setup_wandb(name=args.wandb_name,
                            notes=args.wandb_notes,
                            tags=args.wandb_tags.split(','),
                            extra_config=extra_config,
                            dir='output',
                            mode=args.wandb_mode)

    trainer = build_trainer(assistant)
    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config-file', type=str, default='config/codetalker/vocaset/stage1.yaml', help='path to config file'
    )
    parser.add_argument(
        '--gpu', type=str, default='0', help='gpu id to use'
    )
    parser.add_argument(
        '--no-train', type=bool, default=False, help='wether to train model'
    )
    parser.add_argument(
        '--eval-only', action='store_true', help='wether to train model'
    )

    # wandb config
    parser.add_argument(
        '--use-wandb', action='store_true', help='wether to use wandb for logging'
    )
    parser.add_argument(
        '--wandb-name', type=str, default='TrainingModel', help='the name of experinment'
    )
    parser.add_argument(
        '--wandb-notes', type=str, default='First Stage', help='the noting about the experinment'
    )
    parser.add_argument(
        '--wandb-tags', type=str, default="Codebase,Baseline", help='the tags about the experinment'
    )
    parser.add_argument(
        '--wandb-mode', type=str, default="online", help='the mode of wandb (online/offline)'
    )
    parser.add_argument('--debug', action='store_true', help='wether do debugging')
    parser.add_argument(
        'opts',
        default=None,
        nargs=argparse.REMAINDER,
        help='modify config options using the command-line'
    )
    args = parser.parse_args()
    if args.debug: 
        import debugpy
        debugpy.listen(6666)
        print("Waiting for debugger attach (rank 0)...")
        debugpy.wait_for_client()

    main(args)