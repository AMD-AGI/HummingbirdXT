# download from https://github.com/thu-ml/TurboDiffusion.git

export PYTHONPATH=turbodiffusion

# --image_path              Path to the input image
# --high_noise_model_path   Path to the high noise TurboDiffusion checkpoint
# --low_noise_model_path    Path to the high noise TurboDiffusion checkpoint
# --boundary                Timestep boundary for switching from high to low noise model (default: 0.9)
# --model                   Model to use: Wan2.2-A14B (default: Wan2.2-A14B)
# --num_samples             Number of videos to generate (default: 1)
# --num_steps               Sampling steps, 1–4 (default: 4)
# --sigma_max               Initial sigma for rCM (default: 200); larger choices (e.g., 1600) reduce diversity but may enhance quality
# --vae_path                Path to Wan2.2 VAE (default: checkpoints/Wan2.2_VAE.pth)
# --text_encoder_path       Path to umT5 text encoder (default: checkpoints/models_t5_umt5-xxl-enc-bf16.pth)
# --num_frames              Number of frames to generate (default: 77)
# --prompt                  Text prompt for video generation
# --resolution              Output resolution: "480p" or "720p" (default: 720p)
# --aspect_ratio            Aspect ratio in W:H format (default: 16:9)
# --adaptive_resolution     Enable adaptive resolution based on input image size
# --ode                     Use ODE for sampling (sharper but less robust than SDE)
# --seed                    Random seed for reproducibility (default: 0)
# --save_path               Output file path including extension (default: output/generated_video.mp4)
# --attention_type          Attention module to use: original, sla or sagesla (default: sagesla)
# --sla_topk                Top-k ratio for SLA/SageSLA attention (default: 0.1), we recommend using 0.15 for better video quality
# --quant_linear            Enable quantization for linear layers, pass this if using a quantized checkpoint
# --default_norm            Use the original LayerNorm and RMSNorm of Wan models

python turbodiffusion/inference/wan2.2_i2v_infer.py \
    --model Wan2.2-A14B \
    --low_noise_model_path checkpoints/TurboWan2.2-I2V-A14B-low-720P-quant.pth \
    --high_noise_model_path checkpoints/TurboWan2.2-I2V-A14B-high-720P-quant.pth \
    --resolution 720p \
    --adaptive_resolution \
    --image_path assets/i2v_inputs/i2v_input_0.jpg \
    --prompt "POV selfie video, ultra-messy and extremely fast. A white cat in sunglasses stands on a surfboard with a neutral look when the board suddenly whips sideways, throwing cat and camera into the water; the frame dives sharply downward, swallowed by violent bursts of bubbles, spinning turbulence, and smeared water streaks as the camera sinks. Shadows thicken, pressure ripples distort the edges, and loose bubbles rush upward past the lens, showing the camera is still sinking. Then the cat kicks upward with explosive speed, dragging the view through churning bubbles and rapidly brightening water as sunlight floods back in; the camera races upward, water streaming off the lens, and finally breaks the surface in a sudden blast of light and spray, snapping back into a crooked, frantic selfie as the cat resurfaces." \
    --num_samples 1 \
    --num_steps 4 \
    --quant_linear \
    --attention_type sagesla \
    --sla_topk 0.1 \
    --ode