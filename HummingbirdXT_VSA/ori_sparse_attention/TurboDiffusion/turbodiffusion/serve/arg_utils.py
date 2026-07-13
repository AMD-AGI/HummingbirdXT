# download from https://github.com/thu-ml/TurboDiffusion.git

"""Argument parsing utilities for TurboDiffusion TUI server."""

import argparse
import sys

from imaginaire.utils import log
from rcm.datasets.utils import VIDEO_RES_SIZE_INFO


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for TUI server mode."""
    parser = argparse.ArgumentParser(
        description="TurboDiffusion TUI Server - Interactive video generation"
    )

    parser.add_argument("--mode", choices=["t2v", "i2v"], default="t2v",
                        help="Generation mode: t2v (text-to-video) or i2v (image-to-video)")

    # T2V model path
    parser.add_argument("--dit_path", type=str, default=None,
                        help="Path to DiT checkpoint (required for t2v mode)")

    # I2V model paths
    parser.add_argument("--high_noise_model_path", type=str, default=None,
                        help="Path to high-noise model (required for i2v mode)")
    parser.add_argument("--low_noise_model_path", type=str, default=None,
                        help="Path to low-noise model (required for i2v mode)")
    parser.add_argument("--boundary", type=float, default=0.9,
                        help="Timestep boundary for model switching (i2v only)")

    # Model configuration
    parser.add_argument("--model", choices=["Wan2.1-1.3B", "Wan2.1-14B", "Wan2.2-A14B"],
                        default=None, help="Model architecture (auto-detected from mode if not set)")
    parser.add_argument("--vae_path", type=str, default="checkpoints/Wan2.1_VAE.pth",
                        help="Path to the Wan2.1 VAE")
    parser.add_argument("--text_encoder_path", type=str,
                        default="checkpoints/models_t5_umt5-xxl-enc-bf16.pth",
                        help="Path to the umT5 text encoder")

    # Resolution
    parser.add_argument("--resolution", default=None, type=str,
                        help="Resolution (default: 480p for t2v, 720p for i2v)")
    parser.add_argument("--aspect_ratio", default="16:9", type=str,
                        help="Aspect ratio (width:height)")
    parser.add_argument("--adaptive_resolution", action="store_true",
                        help="Adapt resolution to input image aspect ratio (i2v only)")

    # Attention/quantization
    parser.add_argument("--attention_type", choices=["sla", "sagesla", "original"],
                        default="sagesla", help="Attention mechanism type")
    parser.add_argument("--sla_topk", type=float, default=0.1,
                        help="Top-k ratio for SLA/SageSLA attention")
    parser.add_argument("--quant_linear", action="store_true",
                        help="Use quantized linear layers")
    parser.add_argument("--default_norm", action="store_true",
                        help="Use default LayerNorm/RMSNorm (not optimized)")

    # Sampling options
    parser.add_argument("--ode", action="store_true",
                        help="Use ODE sampling (sharper but less robust, i2v only)")

    # Runtime-adjustable parameters
    parser.add_argument("--num_steps", type=int, choices=[1, 2, 3, 4], default=4,
                        help="Number of inference steps (1-4)")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of samples to generate")
    parser.add_argument("--num_frames", type=int, default=81,
                        help="Number of frames to generate")
    parser.add_argument("--sigma_max", type=float, default=None,
                        help="Initial sigma (default: 80 for t2v, 200 for i2v)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducibility")

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate arguments based on mode."""
    # Set mode-dependent defaults
    if args.model is None:
        args.model = "Wan2.1-1.3B" if args.mode == "t2v" else "Wan2.2-A14B"

    if args.resolution is None:
        args.resolution = "480p" if args.mode == "t2v" else "720p"

    if args.sigma_max is None:
        args.sigma_max = 80 if args.mode == "t2v" else 200

    # Validate mode-specific requirements
    if args.mode == "t2v":
        if args.dit_path is None:
            log.error("--dit_path is required for t2v mode")
            sys.exit(1)
    else:  # i2v
        if args.high_noise_model_path is None or args.low_noise_model_path is None:
            log.error("--high_noise_model_path and --low_noise_model_path are required for i2v mode")
            sys.exit(1)

    # Validate resolution
    if args.resolution not in VIDEO_RES_SIZE_INFO:
        log.error(f"Invalid resolution: {args.resolution}")
        log.info(f"Available: {list(VIDEO_RES_SIZE_INFO.keys())}")
        sys.exit(1)

    if args.aspect_ratio not in VIDEO_RES_SIZE_INFO[args.resolution]:
        log.error(f"Invalid aspect ratio: {args.aspect_ratio}")
        log.info(f"Available: {list(VIDEO_RES_SIZE_INFO[args.resolution].keys())}")
        sys.exit(1)
