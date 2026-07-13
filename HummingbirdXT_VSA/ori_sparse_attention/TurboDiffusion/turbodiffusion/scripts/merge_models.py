# download from https://github.com/thu-ml/TurboDiffusion.git

import torch
import argparse
import sys


def merge_weights(base_path, diff_base_path, diff_target_path, output_path, w):
    print(f"Loading models...")
    print(f"Base: {base_path}")
    print(f"Diff Base: {diff_base_path}")
    print(f"Diff Target: {diff_target_path}")

    try:
        base_sd = torch.load(base_path, map_location="cpu")
        diff_base_sd = torch.load(diff_base_path, map_location="cpu")
        diff_target_sd = torch.load(diff_target_path, map_location="cpu")
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while loading: {e}")
        sys.exit(1)

    merged_sd = {}

    print("\nStarting merge operation...")
    print(f"Formula: Result = Base + {w} * (Diff_Target - Diff_Base)")
    print("-" * 50)

    for key, base_tensor in base_sd.items():
        if not isinstance(base_tensor, torch.Tensor):
            print(f"[WARNING] Key '{key}' is not Tensor.")
            merged_sd[key] = base_tensor
            continue

        if key in diff_base_sd and key in diff_target_sd:
            d_base_tensor = diff_base_sd[key]
            d_target_tensor = diff_target_sd[key]

            if base_tensor.shape != d_base_tensor.shape or base_tensor.shape != d_target_tensor.shape:
                print(f"[WARNING] Shape mismatch for key '{key}'. Keeping Base tensor.")
                merged_sd[key] = base_tensor
                continue

            with torch.no_grad():
                res = base_tensor.float() + w * (d_target_tensor.float() - d_base_tensor.float())

            merged_sd[key] = res.to(base_tensor.dtype)
        else:
            print(f"[INFO] Key '{key}' missing in diff models. Keeping Base tensor.")
            merged_sd[key] = base_tensor

    for key, target_tensor in diff_target_sd.items():
        if key not in merged_sd:
            print(f"[INFO] Key '{key}' missing in base models. Keeping Target tensor.")
            merged_sd[key] = target_tensor

    print("-" * 50)
    print(f"Saving merged model to {output_path}...")
    torch.save(merged_sd, output_path)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge 3 PyTorch models using vector arithmetic.")

    parser.add_argument("--base", type=str, required=True, help="Path to the Base model .pt file")
    parser.add_argument("--diff_base", type=str, required=True, help="Path to the Diff Base model .pt file")
    parser.add_argument("--diff_target", type=str, required=True, help="Path to the Diff Target model .pt file")
    parser.add_argument("--w", type=float, default=1.0, help="Weight factor (w). Default is 1.0")
    parser.add_argument("--output", type=str, default="merged_model.pt", help="Path to save the output .pt file")

    args = parser.parse_args()

    merge_weights(base_path=args.base, diff_base_path=args.diff_base, diff_target_path=args.diff_target, output_path=args.output, w=args.w)
