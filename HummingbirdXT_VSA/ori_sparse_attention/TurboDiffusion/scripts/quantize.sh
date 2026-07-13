# download from https://github.com/thu-ml/TurboDiffusion.git

export PYTHONPATH=turbodiffusion

# python turbodiffusion/inference/modify_model.py \
#     --input_path checkpoints/1.3B_ckpts/2_1_480P/merge_rcm_format.pth \
#     --output_path checkpoints/modified/TurboWan2.1-T2V-1.3B-480P.pth \
#     --model Wan2.1-1.3B \
#     --attention_type sla

# python turbodiffusion/inference/modify_model.py \
#     --input_path checkpoints/1.3B_ckpts/2_1_480P/merge_rcm_format.pth \
#     --output_path checkpoints/modified/TurboWan2.1-T2V-1.3B-480P-quant.pth \
#     --model Wan2.1-1.3B \
#     --attention_type sla \
#     --quant_linear

# python turbodiffusion/inference/modify_model.py \
#     --input_path checkpoints/14B_ckpts/2_1_480P/merge_rcm_format.pth \
#     --output_path checkpoints/modified/TurboWan2.1-T2V-14B-480P.pth \
#     --model Wan2.1-14B \
#     --attention_type sla

# python turbodiffusion/inference/modify_model.py \
#     --input_path checkpoints/14B_ckpts/2_1_480P/merge_rcm_format.pth \
#     --output_path checkpoints/modified/TurboWan2.1-T2V-14B-480P-quant.pth \
#     --model Wan2.1-14B \
#     --attention_type sla \
#     --quant_linear

# python turbodiffusion/inference/modify_model.py \
#     --input_path checkpoints/14B_ckpts/2_1_720P/merge_rcm_format.pth \
#     --output_path checkpoints/modified/TurboWan2.1-T2V-14B-720P.pth \
#     --model Wan2.1-14B \
#     --attention_type sla

# python turbodiffusion/inference/modify_model.py \
#     --input_path checkpoints/14B_ckpts/2_1_720P/merge_rcm_format.pth \
#     --output_path checkpoints/modified/TurboWan2.1-T2V-14B-720P-quant.pth \
#     --model Wan2.1-14B \
#     --attention_type sla \
#     --quant_linear

python turbodiffusion/inference/modify_model.py \
    --input_path checkpoints/14B_ckpts/2_2_720P/merged_low_noise.pth \
    --output_path checkpoints/modified/TurboWan2.2-I2V-A14B-low-720P.pth \
    --model Wan2.2-A14B \
    --attention_type sla

python turbodiffusion/inference/modify_model.py \
    --input_path checkpoints/14B_ckpts/2_2_720P/merged_low_noise.pth \
    --output_path checkpoints/modified/TurboWan2.2-I2V-A14B-low-720P-quant.pth \
    --model Wan2.2-A14B \
    --attention_type sla \
    --quant_linear

python turbodiffusion/inference/modify_model.py \
    --input_path checkpoints/14B_ckpts/2_2_720P/merged_high_noise.pth \
    --output_path checkpoints/modified/TurboWan2.2-I2V-A14B-high-720P.pth \
    --model Wan2.2-A14B \
    --attention_type sla

python turbodiffusion/inference/modify_model.py \
    --input_path checkpoints/14B_ckpts/2_2_720P/merged_high_noise.pth \
    --output_path checkpoints/modified/TurboWan2.2-I2V-A14B-high-720P-quant.pth \
    --model Wan2.2-A14B \
    --attention_type sla \
    --quant_linear
