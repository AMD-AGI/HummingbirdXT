#!/bin/bash
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
set -e
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
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



# Attention backends:
#   FLASH_ATTN               -> full attention
#   VIDEO_SPARSE_ATTN_ORIG   -> original FastVideo VSA
#   SPARSE_LINEAR_ATTN_ORIG  -> original TurboDiffusion SLA
#   VIDEO_SPARSE_ATTN        -> current VSA path in this repo
#   SPARSE_LINEAR_ATTN       -> current SLA path in this repo
export FASTVIDEO_ATTENTION_BACKEND=${FASTVIDEO_ATTENTION_BACKEND:-SPARSE_LINEAR_ATTN}
export FASTVIDEO_VSA_OPT_KERNEL_DIR=${FASTVIDEO_VSA_OPT_KERNEL_DIR:-"${REPO_ROOT}/amd_VSA/FastVideo/fastvideo-kernel/python"}
export FASTVIDEO_VSA_ORIG_KERNEL_DIR=${FASTVIDEO_VSA_ORIG_KERNEL_DIR:-"${REPO_ROOT}/local_vsa_kernels/original"}
# VSA num_stages profile:
#   1 -> training profile (current stable mixed schedule)
#   0 -> testing profile (force all AMD VSA num_stages to [1, 2])
export FASTVIDEO_TRAIN_VSA=${FASTVIDEO_TRAIN_VSA:-1}
# DMD teacher/fake attention role override:
#   auto               -> keep the repo default role split for each backend,auto 保持当前仓库默认行为
# FLASH_ATTN 和 SPARSE_LINEAR_ATTN_ORIG 下，teacher/fake -> dense
# 其他 backend 下，teacher/fake -> vsa_zero_sparsity
#   dense              -> force teacher and fake_score to dense flash attention
#   vsa_zero_sparsity  -> force teacher and fake_score to VSA with sparsity=0.0
export FASTVIDEO_DMD_SCORE_ATTN_MODE=${FASTVIDEO_DMD_SCORE_ATTN_MODE:-dense}

CONFIG_PATH=${CONFIG_PATH:-"configs/self_forcing_wan22_dmd_openvid_human2.yaml"}
LOGDIR=${LOGDIR:-"model_save_${FASTVIDEO_ATTENTION_BACKEND}_bs2"}
DATA_PATH=${DATA_PATH:-"data/DMD_train_data_refine.csv"}
VSA_SPARSITY=${VSA_SPARSITY:-0.8}
BENCHMARK_WARMUP_STEPS=${BENCHMARK_WARMUP_STEPS:-0}
BENCHMARK_STEPS=${BENCHMARK_STEPS:-0}
BENCHMARK_OUTPUT=${BENCHMARK_OUTPUT:-}

echo " FASTVIDEO_ATTENTION_BACKEND = ${FASTVIDEO_ATTENTION_BACKEND}"
echo " FASTVIDEO_VSA_OPT_KERNEL_DIR = ${FASTVIDEO_VSA_OPT_KERNEL_DIR}"
echo " FASTVIDEO_VSA_ORIG_KERNEL_DIR = ${FASTVIDEO_VSA_ORIG_KERNEL_DIR}"
echo " FASTVIDEO_TRAIN_VSA         = ${FASTVIDEO_TRAIN_VSA}"
echo " FASTVIDEO_DMD_SCORE_ATTN_MODE = ${FASTVIDEO_DMD_SCORE_ATTN_MODE}"
echo " VSA_SPARSITY                = ${VSA_SPARSITY}"
echo " BENCHMARK_WARMUP_STEPS      = ${BENCHMARK_WARMUP_STEPS}"
echo " BENCHMARK_STEPS             = ${BENCHMARK_STEPS}"
echo " BENCHMARK_OUTPUT            = ${BENCHMARK_OUTPUT}"

TRAIN_ARGS=(
  --config_path "${CONFIG_PATH}"
  --logdir "${LOGDIR}"
  --data_path "${DATA_PATH}"
  # --disable-wandb
  --VSA_sparsity "${VSA_SPARSITY}"
)

if [ "${BENCHMARK_STEPS}" -gt 0 ]; then
  TRAIN_ARGS+=(
    --benchmark-warmup-steps "${BENCHMARK_WARMUP_STEPS}"
    --benchmark-steps "${BENCHMARK_STEPS}"
    --no_save
    --no_visualize
  )
  if [ -n "${BENCHMARK_OUTPUT}" ]; then
    TRAIN_ARGS+=(--benchmark-output "${BENCHMARK_OUTPUT}")
  fi
fi


torchrun \
  --nnodes=${NNODES} \
  --node_rank=${NODE_RANK} \
  --nproc_per_node=${NPROC_PER_NODE} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  train.py \
    "${TRAIN_ARGS[@]}"
