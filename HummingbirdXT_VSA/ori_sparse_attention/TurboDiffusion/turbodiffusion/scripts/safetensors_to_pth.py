# download from https://github.com/thu-ml/TurboDiffusion.git

import json
import torch
import argparse
import os
from safetensors import safe_open


def merge_safetensors_to_pt(model_dir, output_path, prefix=None):
    index_file = os.path.join(model_dir, "diffusion_pytorch_model.safetensors.index.json")

    if not os.path.exists(index_file):
        print(f"Error: Index file not found at: {index_file}")
        return

    print(f"Reading index file: {index_file}")
    with open(index_file, "r") as f:
        index = json.load(f)

    weight_map = index.get("weight_map")
    if not weight_map:
        print(f"Error: 'weight_map' not found in the index file.")
        return

    merged_state_dict = {}
    shard_cache = {}

    shards_to_process = sorted(list(set(weight_map.values())))
    print(f"Found {len(shards_to_process)} shard files to merge.")

    for param_name, shard_filename in weight_map.items():
        if shard_filename not in shard_cache:
            shard_path = os.path.join(model_dir, shard_filename)
            print(f"  - Loading shard: {shard_path}")
            shard_cache[shard_filename] = safe_open(shard_path, framework="pt", device="cpu")

        try:
            tensor = shard_cache[shard_filename].get_tensor(param_name)
            # Convert Conv3d to Linear
            if "patch_embedding.weight" in param_name:
                tensor = tensor.reshape(tensor.shape[0], -1)
            # Convert to bf16 by default
            merged_state_dict[param_name] = tensor.to(torch.bfloat16)
        except Exception as e:
            print(f"Error: Failed to get tensor '{param_name}' from {shard_filename}: {e}")
            return

    if prefix:
        print(f"\nAdding prefix '{prefix}' to all {len(merged_state_dict)} keys...")
        prefixed_state_dict = {f"{prefix}{k}": v for k, v in merged_state_dict.items()}
        merged_state_dict = prefixed_state_dict

    print(f"All tensors have been merged. Saving to: {output_path}")
    try:
        torch.save(merged_state_dict, output_path)
        print(f"Success! Merged PyTorch checkpoint saved to {output_path}")
    except Exception as e:
        print(f"Error: Failed to save .pth file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge sharded .safetensors model files into a single .pth file.")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to the input directory containing 'diffusion_pytorch_model.safetensors.index.json' and all shard files.",
    )
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output .pth file to be created (including the filename).")
    parser.add_argument("--prefix", type=str, default=None, help="Optional prefix to add to all keys in the state dictionary. E.g., 'net.'")
    args = parser.parse_args()

    merge_safetensors_to_pt(args.model_dir, args.output_path, args.prefix)
