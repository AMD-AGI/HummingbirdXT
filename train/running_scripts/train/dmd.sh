#!/bin/bash
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
set -e
NODE_RANK=${NODE_RANK:-0}

NNODES=${NNODES:-${WORLD_SIZE:-2}}

NPROC_PER_NODE=${NPROC_PER_NODE:-8}

MASTER_ADDR=${MASTER_ADDR:-$(hostname -i | awk '{print $1}')}
MASTER_PORT=${MASTER_PORT:-29500}

echo "======================"
echo " NODE_RANK       = ${NODE_RANK}"
echo " NNODES          = ${NNODES}"
echo " NPROC_PER_NODE  = ${NPROC_PER_NODE}"
echo " MASTER_ADDR     = ${MASTER_ADDR}"
echo " MASTER_PORT     = ${MASTER_PORT}"
echo "======================"



CONFIG_PATH="configs/self_forcing_wan22_dmd_openvid_human2.yaml"
LOGDIR="model_save_path"
DATA_PATH="data/DMD_train_data.csv"
WANDB_DIR="./wandb"


torchrun \
  --nnodes=${NNODES} \
  --node_rank=${NODE_RANK} \
  --nproc_per_node=${NPROC_PER_NODE} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  train.py \
    --config_path "${CONFIG_PATH}" \
    --logdir "${LOGDIR}" \
    --data_path "${DATA_PATH}" \
    --no_visualize \
    --wandb-save-dir "${WANDB_DIR}"
