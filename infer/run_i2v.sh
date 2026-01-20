#!/bin/bash
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
ROCR_VISIBLE_DEVICES=0 python3 examples/wan2.2/predict_ti2v_single.py \
  --model_path models/hm_2.2_5B/ \
  --outdir output \
  --seed 123 --shift 10 --st 0 --ed 20 \
  --H 704 --W 1280 --video_length 121 --fps 24 \
  --i2v_prompt_file prompt.txt \
  --infer_steps 3 \
  --cfg 1.0 \
  --sr_model_path ./models/sr/sr_v3.pth \
  --valid_image_path image_file \
  --vae_model ./models/vae/wan22_v1_tiling_16_12 --t_block_size 16 --t_stride 12 
  #--is_sr 
  #--nsfw_detection \
