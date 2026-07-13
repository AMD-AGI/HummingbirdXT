# download from https://github.com/thu-ml/TurboDiffusion.git

"""Runtime configuration utilities for TurboDiffusion TUI server."""

import argparse

# Runtime-adjustable parameters and their types/validators
RUNTIME_PARAMS = {
    "num_steps": {"type": int, "choices": [1, 2, 3, 4]},
    "num_samples": {"type": int, "min": 1},
    "num_frames": {"type": int, "min": 1},
    "sigma_max": {"type": float, "min": 0.1},
}

# Immutable launch-only parameters
LAUNCH_ONLY_PARAMS = [
    "mode", "model", "dit_path", "high_noise_model_path", "low_noise_model_path",
    "resolution", "aspect_ratio", "attention_type", "sla_topk",
    "quant_linear", "default_norm", "vae_path", "text_encoder_path",
    "boundary", "adaptive_resolution", "ode", "seed",
]


def format_config(args: argparse.Namespace, defaults: dict) -> str:
    """Format configuration as a string for display (with rich markup)."""
    lines = []
    lines.append("\n[bold cyan]Launch Configuration (immutable)[/bold cyan]")
    lines.append(f"  mode:            {args.mode}")
    lines.append(f"  model:           {args.model}")

    if args.mode == "t2v":
        lines.append(f"  dit_path:        {args.dit_path}")
    else:
        lines.append(f"  high_noise_model_path: {args.high_noise_model_path}")
        lines.append(f"  low_noise_model_path:  {args.low_noise_model_path}")
        lines.append(f"  boundary:        {args.boundary}")
        lines.append(f"  adaptive_resolution: {args.adaptive_resolution}")
        lines.append(f"  ode:             {args.ode}")

    lines.append(f"  resolution:      {args.resolution}")
    lines.append(f"  aspect_ratio:    {args.aspect_ratio}")
    lines.append(f"  attention_type:  {args.attention_type}")
    lines.append(f"  sla_topk:        {args.sla_topk}")
    lines.append(f"  quant_linear:    {args.quant_linear}")
    lines.append(f"  default_norm:    {args.default_norm}")
    lines.append(f"  seed:            {args.seed}")

    lines.append("\n[bold cyan]Runtime Configuration (adjustable)[/bold cyan]")
    for param in RUNTIME_PARAMS:
        val = getattr(args, param)
        default = defaults[param]
        marker = " [yellow]*[/yellow]" if val != default else ""
        lines.append(f"  {param}: {val}{marker}")

    return "\n".join(lines)


def set_runtime_param(args: argparse.Namespace, param: str, value: str) -> tuple[bool, str]:
    """Set a runtime parameter. Returns (success, message)."""
    if param not in RUNTIME_PARAMS:
        return False, f"'{param}' is not a runtime parameter. Available: {', '.join(RUNTIME_PARAMS.keys())}"

    spec = RUNTIME_PARAMS[param]
    try:
        typed_value = spec["type"](value)
    except ValueError:
        return False, f"Invalid value '{value}' for {param}"

    if "choices" in spec and typed_value not in spec["choices"]:
        return False, f"{param} must be one of {spec['choices']}"
    if "min" in spec and typed_value < spec["min"]:
        return False, f"{param} must be >= {spec['min']}"

    setattr(args, param, typed_value)
    return True, f"{param} = {typed_value}"
