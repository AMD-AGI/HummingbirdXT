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
    parser.add_argument("--VSA_sparsity", type=float, default=0.0,
                        help="VSA sparsity ratio (0 = disabled / full attn, 0.8 = typical VSA)")
    parser.add_argument("--benchmark-warmup-steps", type=int, default=0,
                        help="Warmup outer training steps before collecting timing stats")
    parser.add_argument("--benchmark-steps", type=int, default=0,
                        help="Measured outer training steps before exiting")
    parser.add_argument("--benchmark-output", type=str, default="",
                        help="Optional path to save benchmark summary JSON")
    return parser

def main():
    parser = get_arg_parser()
    args = parser.parse_args()

    # Set VSA sparsity before model construction (env var controls arch, this controls ratio)
    if args.VSA_sparsity > 0:
        from wan22.modules.model import set_vsa_sparsity
        set_vsa_sparsity(args.VSA_sparsity)

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
    config.benchmark_warmup_steps = args.benchmark_warmup_steps
    config.benchmark_steps = args.benchmark_steps
    config.benchmark_output = args.benchmark_output

    if args.benchmark_steps > 0:
        config.no_save = True
        config.no_visualize = True
        os.environ["FASTVIDEO_TRACK_TRAINABLE_DIT_FORWARD"] = "1"

    trainer = Wan22ScoreDistillationTrainer(config)
    trainer.train()

    if not config.disable_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
