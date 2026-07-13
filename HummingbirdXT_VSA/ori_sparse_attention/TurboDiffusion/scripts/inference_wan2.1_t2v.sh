# download from https://github.com/thu-ml/TurboDiffusion.git

export PYTHONPATH=turbodiffusion

# Arguments:
# --dit_path            Path to the finetuned TurboDiffusion checkpoint
# --model               Model to use: Wan2.1-1.3B or Wan2.1-14B (default: Wan2.1-1.3B)
# --num_samples         Number of videos to generate (default: 1)
# --num_steps           Sampling steps, 1–4 (default: 4)
# --sigma_max           Initial sigma for rCM (default: 80); larger choices (e.g., 1600) reduce diversity but may enhance quality
# --vae_path            Path to Wan2.1 VAE (default: checkpoints/Wan2.1_VAE.pth)
# --text_encoder_path   Path to umT5 text encoder (default: checkpoints/models_t5_umt5-xxl-enc-bf16.pth)
# --num_frames          Number of frames to generate (default: 77)
# --prompt              Text prompt for video generation
# --resolution          Output resolution: "480p" or "720p" (default: 480p)
# --aspect_ratio        Aspect ratio in W:H format (default: 16:9)
# --seed                Random seed for reproducibility (default: 0)
# --save_path           Output file path including extension (default: output/generated_video.mp4)
# --attention_type      Attention module to use: original, sla or sagesla (default: sagesla)
# --sla_topk            Top-k ratio for SLA/SageSLA attention (default: 0.1), we recommend using 0.15 for better video quality
# --quant_linear        Enable quantization for linear layers, pass this if using a quantized checkpoint
# --default_norm        Use the original LayerNorm and RMSNorm of Wan models

python turbodiffusion/inference/wan2.1_t2v_infer.py \
    --model Wan2.1-1.3B \
    --dit_path checkpoints/TurboWan2.1-T2V-1.3B-480P-quant.pth \
    --resolution 480p \
    --prompt "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about." \
    --num_samples 1 \
    --num_steps 4 \
    --quant_linear \
    --attention_type sagesla \
    --sla_topk 0.1