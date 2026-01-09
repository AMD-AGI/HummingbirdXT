# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
python convert_checkpoint.py --input-checkpoint model_4000.pt --output-checkpoint model.pt --ema --to-bf16
python merge_weights.py
