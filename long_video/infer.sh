# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ROCR_VISIBLE_DEVICES=0 python inference.py \
    --config_path configs/wan_22_dmd_vsink1_5B.yaml \
    --output_folder videos/wan22_dmd_vsink1 \
    --checkpoint_path checkpoint/wan_22_vsink1_5B_seq_880/model.pt \
    --data_path prompts/MovieGenVideoBench_extended.txt \
    --num_output_frames 81 \
    --use_ema