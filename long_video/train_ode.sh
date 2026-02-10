# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ROCR_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 \
    --master_addr=127.0.0.1 \
    --master_port=29630 \
    train.py \
    --config_path configs/ODE_5B.yaml \
    --no_visualize
