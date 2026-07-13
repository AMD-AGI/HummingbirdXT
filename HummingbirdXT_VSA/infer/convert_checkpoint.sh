#!/bin/bash
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

PYTHON_BIN=${PYTHON_BIN:-python3}
INPUT_CHECKPOINT=${1:-${INPUT_CHECKPOINT:-model_4000.pt}}
INTERMEDIATE_CHECKPOINT=${2:-${INTERMEDIATE_CHECKPOINT:-model.pt}}
OUTPUT_CHECKPOINT=${3:-${OUTPUT_CHECKPOINT:-./diffusion_pytorch_model.safetensors}}
BASE_CHECKPOINT=${BASE_CHECKPOINT:-"${SCRIPT_DIR}/models/hm_2.2_5B/diffusion_pytorch_model_i2v_ori.safetensors"}

INTERMEDIATE_DIR=$(dirname "${INTERMEDIATE_CHECKPOINT}")
OUTPUT_DIR=$(dirname "${OUTPUT_CHECKPOINT}")
mkdir -p "${INTERMEDIATE_DIR}" "${OUTPUT_DIR}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/convert_checkpoint.py" \
  --input-checkpoint "${INPUT_CHECKPOINT}" \
  --output-checkpoint "${INTERMEDIATE_CHECKPOINT}" \
  --ema \
  --to-bf16

"${PYTHON_BIN}" "${SCRIPT_DIR}/merge_weights.py" \
  --file1 "${BASE_CHECKPOINT}" \
  --file2 "${INTERMEDIATE_CHECKPOINT}" \
  --out "${OUTPUT_CHECKPOINT}"
