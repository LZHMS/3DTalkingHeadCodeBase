import argparse
import warnings
warnings.filterwarnings('ignore')

from base import BaseConfig, build_trainer
from trainers import StyleEncoderTrainer, DiffPoseTalkTrainer, FlowMatchingTrainer

def merge_args(base_cfg, args):
    if args.gpu:
        base_cfg.cfg.ENV.GPU = args.gpu

def main(args):
    base_cfg = BaseConfig()
    base_cfg.cfg.merge_from_file(args.config_file)

    # From command line arguments
    merge_args(base_cfg, args)
    # From optional input arguments
    base_cfg.cfg.merge_from_list(args.opts)
    # frozen the trainer config
    base_cfg.cfg.freeze()
    base_cfg.print_info()

    trainer = build_trainer(base_cfg.cfg)
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