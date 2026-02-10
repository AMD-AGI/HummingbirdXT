# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ROCR_VISIBLE_DEVICES=0,1,4,5,6,7 torchrun --nproc_per_node=6 \
  --master_addr=127.0.0.1 \
  --master_port=29290 \
  train.py \
  --config_path configs/wan_22_dmd_frame1_5B.yaml \
  --logdir checkpoint/wan_22_frame1_5B_seq_880