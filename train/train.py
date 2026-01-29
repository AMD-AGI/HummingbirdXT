# Modifications Copyright(C)[2026] Advanced Micro Devices, Inc. All rights reserved.
import argparse
import os
from omegaconf import OmegaConf
import wandb

from trainer import Wan22ScoreDistillationTrainer

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_visualize", action="store_true")
    parser.add_argument("--logdir", type=str, default="", help="Path to the directory to save logs")
    parser.add_argument("--wandb-save-dir", type=str, default="", help="Path to the directory to save wandb logs")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--data_path", type=str, default=None, help="Path to the dataset")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode, no saving or visualization")
    return parser

def main():
    parser = get_arg_parser()
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)
    config.no_save = args.no_save
    config.no_visualize = args.no_visualize

    # get the filename of config_path
    config_name = os.path.basename(args.config_path).split(".")[0]
    config.config_name = config_name
    config.logdir = args.logdir
    config.wandb_save_dir = args.wandb_save_dir
    config.disable_wandb = args.disable_wandb
    config.data_path = args.data_path
    config.debug = args.debug

    trainer = Wan22ScoreDistillationTrainer(config)
    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    main()
