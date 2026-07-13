# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.

import torch
from safetensors.torch import load_file, save_file
import argparse
import os

def clean_key(k: str):

    k = k.replace("model.", "")
    k = k.replace("_fsdp_wrapped_module.", "")
    return k


def load_state_dict(path: str):
    if path.endswith(".safetensors"):
        return load_file(path)

    state_dict = torch.load(path, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    return state_dict

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
    src = load_state_dict(args.file2)


    cleaned_src = {}
    for k, v in src.items():
        new_k = clean_key(k)
        if isinstance(v, torch.Tensor):
            cleaned_src[new_k] = v.detach().cpu()


    updated = dict(base)
    replaced = 0
    added = 0
    skipped = 0
    for k, v in cleaned_src.items():
        if k in base:
            if base[k].shape != v.shape:
                skipped += 1
                continue
            updated[k] = v
            replaced += 1
        else:
            updated[k] = v
            added += 1

    print(f"✅ Replaced {replaced} tensors from file2")
    print(f"➕ Added {added} new tensors from file2")
    print(f"⚠️  Skipped {skipped} tensors because of shape mismatch")


    print(f"💾 Saving to {args.out}")
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    save_file(updated, args.out)
    print("🎉 Done!")

if __name__ == "__main__":
    main()
