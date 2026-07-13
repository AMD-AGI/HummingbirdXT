# TurboDiffusion TUI Server

Interactive text-based interface for video generation. Loads models once and keeps them GPU-resident for multiple generations.

## Quick Start

```bash
# T2V mode (text-to-video)
PYTHONPATH=turbodiffusion python -m turbodiffusion.serve \
  --mode t2v \
  --dit_path checkpoints/TurboWan2.1-T2V-1.3B-480P.pth

# I2V mode (image-to-video)
PYTHONPATH=turbodiffusion python -m turbodiffusion.serve \
  --mode i2v \
  --high_noise_model_path checkpoints/TurboWan2.2-I2V-14B-720P-high.pth \
  --low_noise_model_path checkpoints/TurboWan2.2-I2V-14B-720P-low.pth
```

## Launch Methods

```bash
# 1. Python module
PYTHONPATH=turbodiffusion python -m turbodiffusion.serve [args]

# 2. Installed CLI (after pip install -e .)
PYTHONPATH=turbodiffusion turbodiffusion-serve [args]

# 3. Via existing inference scripts with --serve flag
PYTHONPATH=turbodiffusion python turbodiffusion/inference/wan2.1_t2v_infer.py --serve [args]
PYTHONPATH=turbodiffusion python turbodiffusion/inference/wan2.2_i2v_infer.py --serve [args]
```

## Arguments

### Mode Selection
| Argument | Description |
|----------|-------------|
| `--mode` | `t2v` (text-to-video) or `i2v` (image-to-video). Default: `t2v` |

### Model Paths
| Argument | Description |
|----------|-------------|
| `--dit_path` | DiT checkpoint path (required for T2V) |
| `--high_noise_model_path` | High-noise model path (required for I2V) |
| `--low_noise_model_path` | Low-noise model path (required for I2V) |
| `--vae_path` | VAE checkpoint. Default: `checkpoints/Wan2.1_VAE.pth` |
| `--text_encoder_path` | Text encoder. Default: `checkpoints/models_t5_umt5-xxl-enc-bf16.pth` |

### Model Configuration
| Argument | Description |
|----------|-------------|
| `--model` | Architecture: `Wan2.1-1.3B`, `Wan2.1-14B`, `Wan2.2-A14B`. Auto-detected from mode |
| `--resolution` | Video resolution: `480p`, `720p`, etc. Default: `480p` (T2V), `720p` (I2V) |
| `--aspect_ratio` | Aspect ratio. Default: `16:9` |
| `--attention_type` | `sagesla` (default), `sla`, or `original` |
| `--sla_topk` | Top-k ratio for SLA attention. Default: `0.1` |
| `--quant_linear` | Enable quantized linear layers |
| `--default_norm` | Use default (non-optimized) normalization |

### Sampling Options
| Argument | Description |
|----------|-------------|
| `--num_steps` | Inference steps (1-4). Default: `4` |
| `--num_samples` | Samples per generation. Default: `1` |
| `--num_frames` | Frames to generate. Default: `81` |
| `--sigma_max` | Initial sigma. Default: `80` (T2V), `200` (I2V) |
| `--seed` | Random seed. Default: `0` |

### I2V-Specific
| Argument | Description |
|----------|-------------|
| `--boundary` | Timestep boundary for model switching. Default: `0.9` |
| `--adaptive_resolution` | Adapt resolution to input image aspect ratio |
| `--ode` | Use ODE sampling (sharper but less robust) |

## Interactive Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/show` | Display current configuration |
| `/set <param> <value>` | Set a runtime parameter |
| `/reset` | Reset runtime parameters to defaults |
| `/quit` | Exit the server |

### Runtime Parameters (adjustable with `/set`)
- `num_steps` - Inference steps (1-4)
- `num_samples` - Samples per generation
- `num_frames` - Frames to generate
- `sigma_max` - Initial sigma value

## Usage Flow

1. **Enter prompt**: Type your text prompt and press Enter
2. **Image path** (I2V only): Enter path to input image
3. **Output path**: Enter output path or press Enter for default
4. **Generation**: Video is generated and saved

End a line with `\` to continue on the next line (bash-style).

## Example Session

```
┌──────────────────────────────────────────┐
│ TurboDiffusion TUI Server                │
│ Mode: T2V (text-to-video)                │
│ Model: Wan2.1-1.3B | Resolution: 480p    │
└──────────────────────────────────────────┘
Type /help for commands. Use \ for newline in prompts.

> A cat sitting on a windowsill \
... watching the rain fall outside
output [output/generated_video.mp4]:
Generating video...
Done: output/generated_video.mp4

> /set num_steps 2
num_steps = 2

> /quit
Goodbye!
```
