import argparse

from base.base_config import BaseConfig
from base.base_trainer import build_trainer
from trainers.toy_trainer import ToyTrainer
from trainers.diffposetalk_trainer import StyleEncoderTrainer, DiffPoseTalkTrainer

import warnings
warnings.filterwarnings('ignore')


def main(args):
    base_cfg = BaseConfig()
    base_cfg.cfg.merge_from_file(args.config_file)
    
    # From optional input arguments
    base_cfg.cfg.merge_from_list(args.opts)
    # frozen the trainer config
    base_cfg.cfg.freeze()

    trainer = build_trainer(base_cfg.cfg)
    if args.mode == "eval":
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
    elif args.mode == "analysis":
        trainer.dm.data_analysis()
    elif args.mode == "train":
        trainer.train()
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config-file', type=str, default='config/codetalker/vocaset/stage1.yaml', help='path to config file'
    )
    parser.add_argument(
        '--mode', type=str, choices=['train', 'eval', 'analysis'], 
        default='train', help='Operation mode: train, eval, or analysis'
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