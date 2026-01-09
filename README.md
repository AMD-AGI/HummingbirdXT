<div align="center">
  <br>
  <br>
  <h1>Bridging the Last Mile: Deploying Hummingbird-XT for Efficient Video Generation on AMD Consumer-Grade Platforms</h1>
<a href='https://huggingface.co/amd/HummingbirdXT'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>
<a href='https://rocm.blogs.amd.com/artificial-intelligence/hummingbirdxt/README.html'><img src='https://img.shields.io/badge/Technical-Blog-red'></a> 
</div>

This repository presents an **efficient acceleration pipeline for Diffusion Transformer (DiT) based video generation models**, optimized for **AMD client-grade GPUs**, including **Navi48 dGPUs** and **Strix Halo iGPUs**.

Built upon this pipeline, we introduce **Hummingbird-XT**, a new family of DiT-based text-to-video models derived from **Wan2.2-5B**, achieving high-quality video generation with significantly reduced inference cost.


<table style="width:100%; table-layout:fixed; border-collapse:collapse;">
  <thead>
    <tr>
      <th style="width:40%; text-align:center;">Caption</th>
      <th style="width:60%; text-align:center;">Video</th>
    </tr>
  </thead>

  <!-- Row 1 -->
  <tr>
    <td style="vertical-align:top; padding:12px;">
      <details>
        <summary style="cursor:pointer; font-weight:600;">
          Text Prompt (click to expand)
        </summary>
        <div style="
          max-height:260px;
          overflow:hidden;
          margin-top:8px;
          line-height:1.55;
          text-align:justify;
        ">
The young East Asian man with short black hair, fair skin, and monolid eyes looks ahead. A young East Asian woman with long black hair and fair skin turns to smile warmly at him.  The background is blurred, focusing on their shared gaze. Realistic cinematic style.
        </div>
      </details>
    </td>
    <td style="padding:12px;">
      <video src="https://github.com/user-attachments/assets/97beef02-ed76-4635-8b36-a296c227cab1"
             controls
             muted
             loop
             style="width:100%; border-radius:8px;">
      </video>
    </td>
  </tr>

  <!-- Row 2 -->
  <tr>
    <td style="vertical-align:top; padding:12px;">
      <details>
        <summary style="cursor:pointer; font-weight:600;">
          Text Prompt (click to expand)
        </summary>
        <div style="
          max-height:260px;
          overflow:hidden;
          margin-top:8px;
          line-height:1.55;
          text-align:justify;
        ">
          A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about.
        </div>
      </details>
    </td>
    <td style="padding:12px;">
      <video src="https://github.com/user-attachments/assets/6698d25f-e839-4acd-b5cd-af8f325d37fc"
             controls
             muted
             loop
             style="width:100%; border-radius:8px;">
      </video>
    </td>
  </tr>

  <!-- Row 3 -->
  <tr>
    <td style="vertical-align:top; padding:12px;">
      <details>
        <summary style="cursor:pointer; font-weight:600;">
          Text Prompt (click to expand)
        </summary>
        <div style="
          max-height:260px;
          overflow:hidden;
          margin-top:8px;
          line-height:1.55;
          text-align:justify;
        ">
          Animated scene features a close-up of a short fluffy monster kneeling beside a melting red candle. The art style is 3D and realistic, with a focus on lighting and texture. The mood of the painting is one of wonder and curiosity, as the monster gazes at the flame with wide eyes and open mouth. Its pose and expression convey a sense of innocence and playfulness, as if it is exploring the world around it for the first time. The use of warm colors and dramatic lighting further enhances the cozy atmosphere of the image.
        </div>
      </details>
    </td>
    <td style="padding:12px;">
      <video src="https://github.com/user-attachments/assets/064c242a-4ee5-429e-9a9b-9b12df076c96"
             controls
             muted
             loop
             style="width:100%; border-radius:8px;">
      </video>
    </td>
  </tr>
</table>

<p align="center">Hummingbird-XT Text-to-Video Showcases</p>


<table style="width:100%; table-layout:fixed; border-collapse:collapse;">
  <thead>
    <tr>
      <th style="width:40%; text-align:center;">Caption</th>
      <th style="width:60%; text-align:center;">Video</th>
    </tr>
  </thead>

  <!-- Row 1 -->
  <tr>
    <td style="vertical-align:top; padding:12px;">
      <details>
        <summary style="cursor:pointer; font-weight:600;">
          Text Prompt (click to expand)
        </summary>
        <div style="
          max-height:260px;
          overflow:hidden;
          margin-top:8px;
          line-height:1.55;
          text-align:justify;
        ">
          A back-view close-up focusing on the runner’s feet striking the track.
          Only subtle movement occurs—his steps land firmly, kicking a small
          amount of dust or rubber granules. The camera stays low and straight-on
          behind him, following smoothly with minimal shake. The sunlight is
          bright, with long shadows stretching forward.
        </div>
      </details>
    </td>
    <td style="padding:12px;">
      <video src="https://github.com/user-attachments/assets/d01d9fe7-bebe-4f0b-902a-3e913d93df1d"
             controls
             muted
             loop
             style="width:100%; border-radius:8px;">
      </video>
    </td>
  </tr>

  <!-- Row 2 -->
  <tr>
    <td style="vertical-align:top; padding:12px;">
      <details>
        <summary style="cursor:pointer; font-weight:600;">
          Text Prompt (click to expand)
        </summary>
        <div style="
          max-height:260px;
          overflow:hidden;
          margin-top:8px;
          line-height:1.55;
          text-align:justify;
        ">
          A graceful woman stands under a majestic sandstone arch, forming a
          small heart shape with her fingers close to the camera while smiling
          warmly and radiating joy. Behind her, a smooth and elegant fountain
          rises gracefully, its water reflecting the warm, inviting courtyard
          walls in a mirror-like fashion.
        </div>
      </details>
    </td>
    <td style="padding:12px;">
      <video src="https://github.com/user-attachments/assets/d4197430-13e7-46d9-b2a9-80df0aee491d"
             controls
             muted
             loop
             style="width:100%; border-radius:8px;">
      </video>
    </td>
  </tr>

  <!-- Row 3 -->
  <tr>
    <td style="vertical-align:top; padding:12px;">
      <details>
        <summary style="cursor:pointer; font-weight:600;">
          Text Prompt (click to expand)
        </summary>
        <div style="
          max-height:260px;
          overflow:hidden;
          margin-top:8px;
          line-height:1.55;
          text-align:justify;
        ">
          舞台上，一名男子弹奏着一把由闪电构成的电吉他。随着音乐渐强，
          火花在他周围噼啪作响。突然，耀眼的光芒转为暗红色，他的双眼
          发出幽光，黑色的翅膀从背后羽化而出。他的皮肤变得黝黑，闪电
          缠绕着他的身体，他化身为一个恶魔，伫立在翻滚的烟雾和雷鸣之中。
        </div>
      </details>
    </td>
    <td style="padding:12px;">
      <video src="https://github.com/user-attachments/assets/7fc77ead-cc5e-4a98-b678-21ed80b91e8c"
             controls
             muted
             loop
             style="width:100%; border-radius:8px;">
      </video>
    </td>
  </tr>
</table>


<p align="center">Hummingbird-XT Image-to-Video Showcases</p>






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
**Long Video Generation**

```
cd long_video
bsah run.sh
```