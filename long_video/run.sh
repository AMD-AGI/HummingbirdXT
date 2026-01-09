# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.

ROCR_VISIBLE_DEVICES=0 python inference.py \
    --config_path configs/self_forcing_dmd_vsink1.yaml \
    --output_folder videos/self_forcing_dmd_vsink1 \
    --checkpoint_path path/to/your/pt/checkpoint.pt \
    --data_path prompts/MovieGenVideoBench_extended.txt \
    --num_output_frames 81 \
    --save_with_index \
    --use_ema