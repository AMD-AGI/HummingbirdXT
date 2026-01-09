
# Hummingbird-XT: Efficient DiT-based Text-to-Video Models on AMD GPUs

This repository presents an **efficient acceleration pipeline for Diffusion Transformer (DiT) based video generation models**, optimized for **AMD client-grade GPUs**, including **Navi48 dGPUs** and **Strix Halo iGPUs**.

Built upon this pipeline, we introduce **Hummingbird-XT**, a new family of DiT-based text-to-video models derived from **Wan2.2-5B**, achieving high-quality video generation with significantly reduced inference cost.

---

## 🔍 Overview

Current text-to-video diffusion models achieve impressive visual quality but remain computationally expensive, particularly on client-grade hardware.
This project aims to bridge the gap between state-of-the-art video generation and practical deployment on AMD client-grade platforms.

Our approach consists of two core components:

1. **Step Distillation for DiT Models**  
   We train DiT models using *step distillation*, enabling them to mimic the diffusion trajectories of the original teacher model with **only a few denoising steps**, dramatically reducing inference latency.

2. **Lightweight VAE Decoder Optimization**  
   We redesign and optimize the **VAE decoder** to reduce both **computational overhead** and **memory consumption**, making high-resolution video decoding feasible on client GPUs.

To address artifacts such as **temporal ghosting** and **motion discontinuity** introduced by aggressive step reduction, we further construct a **curated, re-captioned text–video dataset** covering diverse motion patterns and visual scenarios.

---

## 🚀 Models

### Hummingbird-XT
- DiT-based text-to-video model built upon **Wan2.2-5B**
- Optimized for **few-step inference**
- Designed for **efficient deployment on AMD GPUs**
- Maintains competitive visual quality compared to full-step baselines

### Hummingbird-XTX (Long Video Extension)
- Extends Hummingbird-XT to support **efficient long video generation**
- Improves temporal consistency across extended sequences
- Suitable for long-form generation scenarios


## 🧪 Training & Implementation

This section describes the environment setup, training pipeline, and inference workflow used in this project.

---

### Environment Setup

#### Option 1: Conda Environment

Create a conda environment and install the required dependencies:

```bash
conda create -n HM-XT python=3.10 -y
conda activate HM-XT
pip install -r requirements.txt
pip install flash-attn
```
**Option 2: Docker (Recommended)**
We recommend using our pre-built Docker image for better reproducibility:


```
panisobe/dmd_flash_image_2
Pull the image from Docker Hub and start the container according to your environment configuration.
```

**DMD Training Pipeline**
```
cd train
```
Step 1: Download the Teacher Model
We use Wan2.2-TI2V-5B as the teacher model for step distillation.

```bash
pip install "huggingface_hub[hf_transfer]"
HF_HUB_ENABLE_HF_TRANSFER=1 --local-dir wan_models/Wan2.2-TI2V-5B
```
Step 2: Prepare Training Datasets
We train our models on a mixture of large-scale video datasets, including: MagicData,OpenVid,HumanVid

Please update the dataset root paths in the corresponding CSV files to match your local storage layout.

Step 3: Launch Training
Start the step distillation training using the provided script:

```bash
bash running_scripts/train/dmd.sh
```
The training pipeline demonstrates stable loss convergence across all models.
> Training configuration
> GPUs: 16 × AMD MI325,Iterations: 4000,Training time: ~48 hours.



**Inference Pipeline**
Step 1: Convert the distilled model to the inference-compatible format:
```
cd infer
bash convert_model.sh # Note: change the model path
```

Step 2: Run inference for TI2V generation modes:
inference-compatible format:
```
cd infer
bash convert_model.sh # change the model path to your path
bash run_i2v.sh # Image-to-Video (I2V)
bash run_t2v.sh # Image-to-Video (T2V)
```