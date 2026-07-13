#!/bin/bash
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${SCRIPT_DIR}"

PYTHON_BIN=${PYTHON_BIN:-python3}
ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-0}
SEED=${SEED:-123}
SHIFT=${SHIFT:-12}
ST=${ST:-0}
ED=${ED:-20}
HEIGHT=${HEIGHT:-704}
WIDTH=${WIDTH:-1280}
VIDEO_LENGTH=${VIDEO_LENGTH:-121}
FPS=${FPS:-24}
INFER_STEPS=${INFER_STEPS:-3}
CFG_SCALE=${CFG_SCALE:-1.0}
SR_MODEL_PATH=${SR_MODEL_PATH:-./models/sr/sr_v3.pth}
I2V_PROMPT_FILE=${I2V_PROMPT_FILE:-}
VALID_IMAGE_PATH=${VALID_IMAGE_PATH:-}
VAE_MODEL=${VAE_MODEL:-./models/vae/wan22_v1_tiling_16_12}
T_BLOCK_SIZE=${T_BLOCK_SIZE:-16}
T_STRIDE=${T_STRIDE:-12}

# Attention backends:
#   FLASH_ATTN               -> full attention
#   VIDEO_SPARSE_ATTN_ORIG   -> original FastVideo VSA
#   SPARSE_LINEAR_ATTN_ORIG  -> original TurboDiffusion SLA
#   VIDEO_SPARSE_ATTN        -> current VSA path in this repo
#   SPARSE_LINEAR_ATTN       -> current SLA path in this repo
export FASTVIDEO_ATTENTION_BACKEND=${FASTVIDEO_ATTENTION_BACKEND:-VIDEO_SPARSE_ATTN}
MODEL_PATH=${MODEL_PATH:-models/hm_2.2_5B/diffusion_pytorch_model_VIDEO_SPARSE_ATTN.safetensors}
OUTDIR=${OUTDIR:-output_VSA_bs2_new3}
#if [ -z "${MODEL_PATH:-}" ]; then
#  case "${FASTVIDEO_ATTENTION_BACKEND}" in
#    VIDEO_SPARSE_ATTN_ORIG)
#      MODEL_PATH="models/hm_2.2_5B/diffusion_pytorch_model_VIDEO_SPARSE_ATTN_ORIG.safetensors"
#      ;;
#    VIDEO_SPARSE_ATTN)
#      MODEL_PATH="models/hm_2.2_5B/diffusion_pytorch_model_VSA.safetensors"
#      ;;
#    *)
#      MODEL_PATH="models/hm_2.2_5B/diffusion_pytorch_model_i2v_ori.safetensors"
#      ;;
#  esac
#fi

export FASTVIDEO_ATTN_SPARSITY=${FASTVIDEO_ATTN_SPARSITY:-${VSA_SPARSITY:-0.8}}
export FASTVIDEO_VSA_OPT_KERNEL_DIR=${FASTVIDEO_VSA_OPT_KERNEL_DIR:-"${REPO_ROOT}/amd_VSA/FastVideo/fastvideo-kernel/python"}
export FASTVIDEO_VSA_ORIG_KERNEL_DIR=${FASTVIDEO_VSA_ORIG_KERNEL_DIR:-"${REPO_ROOT}/local_vsa_kernels/original"}
# AMD VSA optimization switch:
#   0 -> enable AMD-optimized VSA kernel path
#   1 -> disable AMD optimization and fall back to the non-optimized path
export FASTVIDEO_DISABLE_AMD_VSA_OPT=${FASTVIDEO_DISABLE_AMD_VSA_OPT:-0}
# VSA num_stages profile:
#   1 -> training profile (current stable mixed schedule)
#   0 -> testing profile (force all AMD VSA num_stages to [1, 2])
export FASTVIDEO_TRAIN_VSA=${FASTVIDEO_TRAIN_VSA:-0}

CMD=(
  "${PYTHON_BIN}" examples/wan2.2/predict_ti2v_single.py
  --model_path "${MODEL_PATH}"
  --outdir "${OUTDIR}"
  --seed "${SEED}"
  --shift "${SHIFT}"
  --st "${ST}"
  --ed "${ED}"
  --H "${HEIGHT}"
  --W "${WIDTH}"
  --video_length "${VIDEO_LENGTH}"
  --fps "${FPS}"
  --i2v_prompt_file "${I2V_PROMPT_FILE}"
  --infer_steps "${INFER_STEPS}"
  --cfg "${CFG_SCALE}"
  --sr_model_path "${SR_MODEL_PATH}"
  --valid_image_path "${VALID_IMAGE_PATH}"
  --vae_model "${VAE_MODEL}"
  --t_block_size "${T_BLOCK_SIZE}"
  --t_stride "${T_STRIDE}"
)

ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES}" "${CMD[@]}"
