# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.

import torch
from safetensors.torch import load_file, save_file
import argparse
import os

def clean_key(k: str):

    k = k.replace("model.", "")
    k = k.replace("_fsdp_wrapped_module.", "")
    return k

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file1", type=str, default="models/hm_2.2_5B/diffusion_pytorch_model_i2v_ori.safetensors", help="Path to diffusion_pytorch_model.safetensors (base)")
    parser.add_argument("--file2", type=str, default="000151_model.pt", help="Path to model.pt (source)")
    parser.add_argument("--out", type=str, default="./diffusion_pytorch_model.safetensors")
    args = parser.parse_args()

    assert os.path.exists(args.file1), f"File1 not found: {args.file1}"
    assert os.path.exists(args.file2), f"File2 not found: {args.file2}"


    print(f"📦 Loading base weights from {args.file1}")
    base = load_file(args.file1)


    print(f"📦 Loading source weights from {args.file2}")
    src = torch.load(args.file2, map_location="cpu")
    if isinstance(src, dict) and "state_dict" in src:
        src = src["state_dict"]


    cleaned_src = {}
    for k, v in src.items():
        new_k = clean_key(k)
        cleaned_src[new_k] = v


    updated = {}
    replaced = 0
    missing = 0
    for k, v in base.items():
        if k in cleaned_src:
            updated[k] = cleaned_src[k].detach().cpu()
            replaced += 1
        else:
            updated[k] = v
            missing += 1

    print(f"✅ Replaced {replaced} tensors from file2")
    print(f"⚠️  Missing {missing} tensors (kept original)")


    print(f"💾 Saving to {args.out}")
    save_file(updated, args.out)
    print("🎉 Done!")

if __name__ == "__main__":
    main()

