# -*- coding: utf-8 -*-
import os
import sys
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
import clip
from argparse import ArgumentParser
from omegaconf import OmegaConf

# Schedulers
#from diffusers import FlowMatchEulerDiscreteScheduler
current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.dist import set_multi_gpus_devices, shard_model
from videox_fun.models import (
    AutoencoderKLWan, AutoencoderKLWan3_8, WanT5EncoderModel, AutoTokenizer, Wan2_2Transformer3DModel, Wan22Model
)
from videox_fun.models.cache_utils import get_teacache_coefficients
from videox_fun.pipeline import Wan2_2TI2VPipeline, FlowShiftScheduler
from videox_fun.utils.fp8_optimization import (
    convert_model_weight_to_float8, replace_parameters_by_name, convert_weight_dtype_wrapper
)
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.utils import filter_kwargs, get_image_to_video_latent, save_videos_grid
from inference_realesrgan_video import run_sr
from nsfw import detect_nsfw_theme
from autoencoder_kl_turbo_vaed_ours_wan22 import ours_decode, load_vae_for_videox_fun

import torchvision.transforms.functional as TF
from PIL import Image


# ---------------------------
# Parsing
# ---------------------------
def build_parser() -> ArgumentParser:
    p = ArgumentParser()
    p.add_argument('--prompt_file', type=str, default='prompts_test_0617_movgen.txt')
    p.add_argument('--i2v_prompt_file', type=str, default='prompts_test_0617_movgen.txt')
    p.add_argument('--outdir', type=str, default="vis_results_10/wan2.2_distilled/")
    p.add_argument('--lora_path', type=str, default=None)
    p.add_argument('--lora_high_path', type=str, default=None)
    p.add_argument('--lora_weight', type=float, default=0.55)
    p.add_argument('--lora_high_weight', type=float, default=0.55)
    p.add_argument('--num_skip_start_steps', type=float, default=2)
    p.add_argument('--enable_teacache', action='store_true', default=False)
    p.add_argument('--nsfw_detection', action='store_true', default=False)
    p.add_argument('--seed', type=int, default=6666)
    p.add_argument('--H', type=int, default=704)
    p.add_argument('--W', type=int, default=1280)
    p.add_argument('--video_length', type=int, default=121)
    p.add_argument('--fps', type=int, default=16)
    p.add_argument('--shift', type=float, default=5)
    # p.add_argument('--use_14b', action='store_true') # will support in future
    p.add_argument('--st', type=int, default=0)
    p.add_argument('--ed', type=int, default=100000)
    # p.add_argument('--with_neg', action='store_true', default=False)
    p.add_argument('--infer_steps', default=50, type=int)
    p.add_argument('--model_path', default="models/wan2.2_5b_dmd/", type=str)
    p.add_argument('--timesteps', default=None, type=str)
    p.add_argument('--cfg', type=float, default=5.0)
    p.add_argument('--sr_scale', type=int, default=4)
    p.add_argument('--valid_image_path', default=None, type=str)
    p.add_argument('--sr_model_path', type=str, default="/home/takisobe/takashi/Hummingbird_XT_22/VSR/models/sr.pth")
    p.add_argument('--is_sr', action='store_true', default=False)
    # VAE
    p.add_argument('--vae_model', type=str, default=None, help="compressed VAE ckpt for ours_decode")
    p.add_argument('--t_block_size', type=int, default=-1)
    p.add_argument('--t_stride', type=int, default=-1)
    return p


# ---------------------------
# Utilities
# ---------------------------
SENSITIVE_WORDS = {
    "violent", "nudity", "pornography", "sexual_intercourse", "child_exploitation",
    "sexual_solicitation", "violence_gore", "self_harm", "harassment",
    "hate", "intolerance", "drugs", "alcohol", "tobacco",
    "weapons", "gambling", "controversial"
}

def is_sensitive_word(word: str) -> bool:
    return word in SENSITIVE_WORDS

def safe_slug(s: str, max_len: int = 80) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^\w\-]+", "_", s, flags=re.UNICODE).strip("_")
    s = s or "output"
    return s[:max_len]

def load_prompts(txt: Path) -> List[str]:
    assert txt.suffix == ".txt" and txt.is_file(), f"Prompt file invalid: {txt}"
    out: List[str] = []
    with txt.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(line)
    return out

def parse_timesteps(timesteps: Optional[str], expect_steps: int) -> Optional[List[int]]:
    if timesteps is None or timesteps == "None":
        return None
    values = [int(v) for v in timesteps.split(",") if v.strip() != ""]
    assert len(values) == expect_steps, f"timesteps length mismatch: {len(values)} vs {expect_steps}"
    return values

def round_video_length_for_vae(video_length: int, compression_ratio: int) -> Tuple[int, int]:
    """Return (rounded_video_length, latent_frames)."""
    if video_length == 1:
        return 1, 1
    rounded = int((video_length - 1) // compression_ratio * compression_ratio) + 1
    latent = (rounded - 1) // compression_ratio + 1
    return rounded, latent

def ensure_outdir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def pick_negative_prompt(use_chinese_strong: bool) -> str:
    if use_chinese_strong:
        return ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，"
                "低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"
                "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")
    return "cartoon,anime"


# ---------------------------
# Model / Pipeline setup
# ---------------------------
def build_components(args):
    # Device / multi-GPU
    ulysses_degree, ring_degree = 1, 1
    device = set_multi_gpus_devices(ulysses_degree, ring_degree)

    # Load config
    config_path = Path("config/wan2.2/wan_civitai_5b.yaml")
    config = OmegaConf.load(str(config_path))
    boundary = config['transformer_additional_kwargs'].get('boundary', 0.875)
    weight_dtype = torch.bfloat16  # fallback to fp16 if card doesn't support bfloat16 (keep behavior)


    # Transformers
    model_root = Path(args.model_path)
    low_sub = config['transformer_additional_kwargs'].get('transformer_low_noise_model_subpath', 'transformer')
    high_sub = config['transformer_additional_kwargs'].get('transformer_high_noise_model_subpath', 'transformer')
    comb_type = config['transformer_additional_kwargs'].get('transformer_combination_type', 'single')

    # I2V folder mode?
    i2v_folder: Optional[Path] = None
    if args.valid_image_path is not None and Path(args.valid_image_path).is_dir():
        mode = 'i2v'
        transformer = Wan22Model.from_pretrained(str(model_root / low_sub))
        if args.infer_steps == 1:
            args.timesteps = "1000"
        elif args.infer_steps == 2:
            args.timesteps = "1000,750"
        elif args.infer_steps == 3:
            args.timesteps = "1000,750,500"
        elif args.infer_steps == 4:
            args.timesteps = "1000,750,500,250"

        scheduler = FlowShiftScheduler(
            shift=args.shift, sigma_min=0.0, extra_one_step=True
        )
        shift = args.shift

        print("Using Flow_Shift for I2V.")
        #print("shift:", shift)
            
    else:
        mode = 't2v'
        transformer = Wan22Model.from_pretrained(str(model_root / low_sub))
        if args.infer_steps == 1:
            args.timesteps = "1000"
        elif args.infer_steps == 2:
            args.timesteps = "1000,750"
        elif args.infer_steps == 3:
            args.timesteps = "1000,750,500"
        elif args.infer_steps == 4:
            args.timesteps = "1000,750,500,250"

        scheduler = FlowShiftScheduler(
            shift=args.shift, sigma_min=0.0, extra_one_step=True
        )
        shift = args.shift

        print("Using Flow_Shift for T2V.")
    
    # print(comb_type)
    transformer_2 = None
    if comb_type == "moe":
        transformer_2 = Wan2_2Transformer3DModel.from_pretrained(
            str(model_root / high_sub),
            transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
            low_cpu_mem_usage=True, torch_dtype=weight_dtype,
        )

    # VAE (base)
    ChosenAE = {
        "AutoencoderKLWan": AutoencoderKLWan,
        "AutoencoderKLWan3_8": AutoencoderKLWan3_8,
    }[config['vae_kwargs'].get('vae_type', 'AutoencoderKLWan')]
    vae = ChosenAE.from_pretrained(
        str(model_root / config['vae_kwargs'].get('vae_subpath', 'vae')),
        additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
    ).to(weight_dtype)

    # Compressed VAE for final decode
    assert args.vae_model and Path(args.vae_model).exists(), f"vae_model not found: {args.vae_model}"
    compressed_vae = load_vae_for_videox_fun(args.vae_model)
    if args.t_block_size > 0:
        compressed_vae.use_framewise_decoding = True
        compressed_vae.tile_sample_min_num_frames = args.t_block_size
        compressed_vae.tile_sample_stride_num_frames = args.t_block_size if args.t_stride <= 0 else args.t_stride
    else:
        compressed_vae.use_framewise_decoding = False

    # compressed_vae = vae

    # Tokenizer / Text encoder
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_root / config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
    )
    text_encoder = WanT5EncoderModel.from_pretrained(
        str(model_root / config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
        additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
        low_cpu_mem_usage=True, torch_dtype=weight_dtype,
    )

    # Pipeline
    pipeline = Wan2_2TI2VPipeline(
        transformer=transformer,
        transformer_2=transformer_2,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
    ).to(device=device)

    return dict(
        mode = mode,
        device=device,
        weight_dtype=weight_dtype,
        config=config,
        boundary=boundary,
        transformer=transformer,
        transformer_2=transformer_2,
        vae=vae,
        compressed_vae=compressed_vae,
        pipeline=pipeline,
        shift=shift,
    )


def configure_memory_offload(pipeline, transformer, transformer_2, weight_dtype, device, mode: Optional[str] = None):
    """
    mode in [None, 'sequential_cpu_offload', 'model_cpu_offload_and_qfloat8',
             'model_cpu_offload', 'model_full_load_and_qfloat8']
    """
    if mode == "sequential_cpu_offload":
        replace_parameters_by_name(transformer, ["modulation"], device=device)
        transformer.freqs = transformer.freqs.to(device=device)
        if transformer_2 is not None:
            replace_parameters_by_name(transformer_2, ["modulation"], device=device)
            transformer_2.freqs = transformer_2.freqs.to(device=device)
        pipeline.enable_sequential_cpu_offload(device=device)
    elif mode == "model_cpu_offload_and_qfloat8":
        convert_model_weight_to_float8(transformer, exclude_module_name=["modulation"], device=device)
        convert_weight_dtype_wrapper(transformer, weight_dtype)
        if transformer_2 is not None:
            convert_model_weight_to_float8(transformer_2, exclude_module_name=["modulation"], device=device)
            convert_weight_dtype_wrapper(transformer_2, weight_dtype)
        pipeline.enable_model_cpu_offload(device=device)
    elif mode == "model_cpu_offload":
        pipeline.enable_model_cpu_offload(device=device)
    elif mode == "model_full_load_and_qfloat8":
        convert_model_weight_to_float8(transformer, exclude_module_name=["modulation"], device=device)
        convert_weight_dtype_wrapper(transformer, weight_dtype)
        if transformer_2 is not None:
            convert_model_weight_to_float8(transformer_2, exclude_module_name=["modulation"], device=device)
            convert_weight_dtype_wrapper(transformer_2, weight_dtype)
        pipeline.to(device=device)
    else:
        pipeline.to(device=device)


def maybe_enable_teacache(pipeline, model_name: str, enable: bool,
                          num_inference_steps: int, threshold: float, skip_steps: float, offload: bool):
    if not enable:
        return
    coeff = get_teacache_coefficients(model_name)
    if coeff is None:
        return
    print(f"Enable TeaCache with threshold {threshold} and skip the first {skip_steps} steps.")
    pipeline.transformer.enable_teacache(coeff, num_inference_steps, threshold,
                                         num_skip_start_steps=skip_steps, offload=offload)
    if pipeline.transformer_2 is not None:
        pipeline.transformer_2.share_teacache(transformer=pipeline.transformer)


def maybe_enable_cfg_skip(pipeline, cfg_skip_ratio: float, num_inference_steps: int):
    if cfg_skip_ratio is None:
        return
    print(f"Enable cfg_skip_ratio {cfg_skip_ratio}.")
    pipeline.transformer.enable_cfg_skip(cfg_skip_ratio, num_inference_steps)
    if pipeline.transformer_2 is not None:
        pipeline.transformer_2.share_cfg_skip(transformer=pipeline.transformer)


# ---------------------------
# Generation
# ---------------------------
def generate_one_video(
    mode, 
    pipeline,
    compressed_vae,
    negative_prompt,
    prompt: str,
    sample_size: Tuple[int, int],
    video_length: int,
    boundary: float,
    shift: float,
    num_inference_steps: int,
    guidance_scale: float,
    generator: torch.Generator,
    timesteps: Optional[List[int]],
    validation_image_path: Optional[Path],
    enable_riflex: bool,
    riflex_k: int,
    device: torch.device,
    is_sr: bool,
    sr_model_path: Optional[Path],
    sr_scale: int,
    outpath: Path,
    fps: int = 24,
    seed: int = 0,
) -> None:
    """
    Unified path for both T2V and I2V-start (if validation_image_path is given).
    """
    # VAEs need alignment between num frames and compression_ratio
    vae = pipeline.vae
    vlen_aligned, latent_frames = round_video_length_for_vae(video_length, vae.config.temporal_compression_ratio)

    # Riflex
    if enable_riflex:
        pipeline.transformer.enable_riflex(k=riflex_k, L_test=latent_frames)
        if pipeline.transformer_2 is not None:
            pipeline.transformer_2.enable_riflex(k=riflex_k, L_test=latent_frames)

    # Timesteps
    if timesteps is not None:
        assert len(timesteps) == num_inference_steps, f"{len(timesteps)=}, {num_inference_steps=}"

    # Input initialization (T2V vs I2V-start)
    if validation_image_path is not None:
        img = Image.open(validation_image_path).convert("RGB")
        img = img.resize((sample_size[1], sample_size[0]), Image.LANCZOS)
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to("cuda").unsqueeze(1).to(dtype=torch.bfloat16)
        #img = TF.to_tensor(img).to("cuda").unsqueeze(1).to(dtype=torch.bfloat16)
        input_video_mask = None
    else:
        img, input_video_mask = None, None

    with torch.no_grad():
        latents = pipeline(
            mode,
            prompt,
            num_frames=vlen_aligned,
            negative_prompt=negative_prompt,
            height=sample_size[0],
            width=sample_size[1],
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            boundary=boundary,
            img=img,
            mask_video=input_video_mask,
            shift=shift,
            input_timesteps=timesteps,
            output_type='latent',   # decode later with compressed VAE
            seed = seed,
        ).videos

    # decode -> optional SR -> save
    decoded = ours_decode(pipeline, compressed_vae, latents)  # [B, C, T, H, W], 0..1, torch
    if is_sr:
        assert sr_model_path is not None and sr_model_path.exists(), f"SR model not found: {sr_model_path}"
        decoded = run_sr(decoded, str(sr_model_path), sr_scale)

    # to numpy -> back to torch for save helper
    decoded_np = decoded.cpu().float().numpy()
    decoded = torch.from_numpy(decoded_np)

    ensure_outdir(outpath)
    if vlen_aligned == 1:
        # Save image (RGB) — ensure channel order (C,T,H,W) -> (H,W,C)
        img = decoded[0, :, 0]  # C,H,W
        img = img.permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy()
        Image.fromarray(img).save(str(outpath))
    else:
        save_videos_grid(decoded, str(outpath), fps=fps)

    print("video saving to:", str(outpath))


def run():
    torch.use_deterministic_algorithms(True)
    args = build_parser().parse_args()
    print(args)
    if args.infer_steps > 4:
        raise ValueError(
            f"Invalid infer_steps={args.infer_steps}. "
            "For turbo model, input infer_steps should be less than or equal to 4."
        )

    # Prepare outdir
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Prompts
    if args.valid_image_path is None:
        prompt_file = Path(args.prompt_file)
    else:
        prompt_file = Path(args.i2v_prompt_file)
    prompts = load_prompts(prompt_file)[args.st: args.ed]

    # # Neg prompts
    # neg_prompt = pick_negative_prompt(use_chinese_strong=args.with_neg)
    # negative_prompts = [neg_prompt] * len(prompts)

    # Core components
    comps = build_components(args)
    mode = comps["mode"]
    pipeline = comps["pipeline"]
    compressed_vae = comps["compressed_vae"]
    device = comps["device"]
    boundary = comps["boundary"]
    transformer = comps["transformer"]
    transformer_2 = comps["transformer_2"]
    weight_dtype = comps["weight_dtype"]
    shift = comps["shift"] 

    # (Optional) memory modes — keep default None to match original behavior
    GPU_MEMORY_MODE = None
    configure_memory_offload(pipeline, transformer, transformer_2, weight_dtype, device, GPU_MEMORY_MODE)

    # NSFW model (optional)
    nsfw_model = None
    if args.nsfw_detection:
        nsfw_model, _ = clip.load("ViT-B/32", device=device)

    # I2V folder mode
    i2v_folder: Optional[Path] = None
    if args.valid_image_path is not None and Path(args.valid_image_path).is_dir():
        i2v_folder = Path(args.valid_image_path)
        pngs = sorted([p for p in i2v_folder.iterdir() if p.suffix.lower() == ".png"],
                      key=lambda p: int(p.stem))
        assert len(pngs) == len(prompts), f"图片数量 {len(pngs)} 与 prompt 数量 {len(prompts)} 不匹配"
        print(f"Detected I2V mode: {len(pngs)} images, {len(prompts)} prompts")
        image_iter: Iterable[Optional[Path]] = pngs
    else:
        print(f"Detected T2V mode: {len(prompts)} prompts")
        image_iter = [None] * len(prompts)

    # LoRA merge/unmerge around generation
    merged = False
    try:
        if args.lora_path:
            pipeline = merge_lora(pipeline, args.lora_path, args.lora_weight, device=device, dtype=weight_dtype)
            if transformer_2 is not None and args.lora_high_path:
                pipeline = merge_lora(
                    pipeline, args.lora_high_path, args.lora_high_weight,
                    device=device, dtype=weight_dtype, sub_transformer_name="transformer_2"
                )
            merged = True

        # Common params
        sample_size = (args.H, args.W)  # H, W
        video_length = args.video_length
        fps = args.fps

        # Iterate
        for prompt, img_path in zip(prompts, image_iter):
            # NSFW gate (prompt-based)
            if nsfw_model is not None:
                label, prob, tag = detect_nsfw_theme(nsfw_model, device, prompt, threshold=0.80)
                if is_sensitive_word(tag):
                    print(f"❌ NSFW Detected: {tag} ({prob*100:.2f}%). Stop Video Generation")
                    continue
                else:
                    print(f"✅ Safe: {label}. Do Video Generation!")
            else:
                print("without using NSFW detection mode!")

            slug = safe_slug(prompt)
            outname = f"{slug}.mp4"
            outpath = outdir / outname

            tsteps = parse_timesteps(args.timesteps, args.infer_steps) if args.timesteps else None

            print("shift:", shift)

            # For T2V, allow optional single image start (validation_image_start)
            validation_image_path = None
            if img_path is not None:
                validation_image_path = img_path
            elif args.valid_image_path and Path(args.valid_image_path).is_file():
                validation_image_path = Path(args.valid_image_path)

            # RNG
            # print(shift)
            generator = torch.Generator(device="cpu").manual_seed(args.seed)
            generate_one_video(
                mode,
                pipeline, compressed_vae,
                prompt=prompt,
                negative_prompt=None, # CFG free model
                sample_size=sample_size,
                video_length=video_length,
                boundary=boundary,
                shift=shift,
                num_inference_steps=args.infer_steps,
                guidance_scale=args.cfg,
                generator=generator,
                timesteps=tsteps,
                validation_image_path=validation_image_path,
                enable_riflex=False,
                riflex_k=6,
                device=device,
                is_sr=args.is_sr,
                sr_model_path=Path(args.sr_model_path) if args.sr_model_path else None,
                sr_scale=args.sr_scale,
                outpath=outpath,
                fps=fps,
                seed=args.seed,
            )

    finally:
        # Clean LoRA even on exceptions
        if merged and args.lora_path:
            pipeline = unmerge_lora(pipeline, args.lora_path, args.lora_weight, device=device, dtype=weight_dtype)
            if transformer_2 is not None and args.lora_high_path:
                pipeline = unmerge_lora(
                    pipeline, args.lora_high_path, args.lora_high_weight,
                    device=device, dtype=weight_dtype, sub_transformer_name="transformer_2"
                )


if __name__ == "__main__":
    run()

