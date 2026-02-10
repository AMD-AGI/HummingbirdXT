# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
ROCR_VISIBLE_DEVICES=7 python inference.py \
    --config_path configs/wan21_dmd_vsink1.yaml \
    --output_folder videos/wan21 \
    --checkpoint_path checkpoints/wan21_model.pt \
    --data_path prompts/MovieGenVideoBench_extended.txt \
    --num_output_frames 81 \
    --model_name Wan2.1-T2V-1.3B \
    --use_ema