<div align="center">
  <br>
  <br>
  <h1>Bridging the Last Mile: Deploying Hummingbird-XT for Efficient Video Generation on AMD Consumer-Grade Platforms</h1>
<a href='https://huggingface.co/amd/HummingbirdXT'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>
<a href='https://rocm.blogs.amd.com/artificial-intelligence/hummingbirdxt/README.html'><img src='https://img.shields.io/badge/Technical-Blog-red'></a> 
</div>

## 🔍 Overview
This repository presents an efficient acceleration pipeline for Diffusion Transformer (DiT) based video generation models, optimized for **AMD client-grade GPUs**, including **Navi48 dGPUs** and **Strix Halo iGPUs**.

Built upon this pipeline, we introduce **Hummingbird-XT**, a new family of DiT-based text-to-video models derived from **Wan2.2-5B**, achieving high-quality video generation with significantly reduced inference cost.Additionally, to further extend the length of generated videos, we introduce Hummingbird-XTX, an efficient autoregressive model for long-video generation based on **Wan-2.1-1.3B**, which is capable of generating long videos.

<p align="center"><strong>Hummingbird-XT Text-to-Video Showcases</strong></p>
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


<p align="center"><strong>Hummingbird-XT Image-to-Video Showcases</strong></p>
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


<p align="center"><strong>Hummingbird-XTX 20s videos Showcases</strong></p>
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
Several giant wooly mammoths approach treading through a snowy meadow, their long wooly fur lightly blows in the wind as they walk, snow covered trees and dramatic snow capped mountains in the distance, mid afternoon light with wispy clouds and a sun high in the distance creates a warm glow, the low camera view is stunning capturing the large furry mammal with beautiful photography, depth of field.
        </div>
      </details>
    </td>
    <td style="padding:12px;">
      <video src="https://github.com/user-attachments/assets/8dcab976-6ac6-419e-82f9-71a0b8d8fe7e"
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
           A movie trailer featuring the adventures of the 30 year old space man wearing a red wool knitted motorcycle helmet, blue sky, salt desert, cinematic style, shot on 35mm film, vivid colors.
        </div>
      </details>
    </td>
    <td style="padding:12px;">
      <video src="https://github.com/user-attachments/assets/6c1e0e92-8521-4402-8652-35b87efac7ed"
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
      A cinematic wide portrait of a man with his face lit by the glow of a TV.
        </div>
      </details>
    </td>
    <td style="padding:12px;">
      <video src="https://github.com/user-attachments/assets/49465535-34b5-49f4-925c-ca3379b92dc1"
             controls
             muted
             loop
             style="width:100%; border-radius:8px;">
      </video>
    </td>
  </tr>
</table>

---

## 📝 News
- __[2026.01.09]__: 🔥🔥Release the full code and pre-trained weight of HummingbirdXT.!
- __[2026.01.08]__: 🔥🔥Release our Blog: [Bridging the Last Mile: Deploying Hummingbird-XT for Efficient Video Generation on AMD Consumer-Grade Platforms](https://rocm.blogs.amd.com/artificial-intelligence/hummingbirdxt/README.html) !

---

## 🧬 Models

**Hummingbird-XT**
- DiT-based text-to-video model built upon **Wan2.2-5B**
- Optimized for **few-step inference**
- Designed for **efficient deployment on AMD GPUs**
- Maintains competitive visual quality compared to full-step baselines

**Hummingbird-XTX** (Long Video Extension)
- Extends Hummingbird-XT to support **efficient long video generation**
- Improves temporal consistency across extended sequences
- Suitable for long-form generation scenarios
---

## ⚙️ Installation
Clone this Repo:
```bash
git clone https://github.com/AMD-AGI/HummingbirdXT.git
cd HummingbirdXT
```
### Option 1: Conda Environment
```bash
conda create -n hummingbirdxt python=3.10
conda activate hummingbirdxt
pip install -r requirements.txt
```
For rocm flash-attn, you can install it by this [link](https://github.com/ROCm/flash-attention).
```
git clone https://github.com/ROCm/flash-attention.git
cd flash-attention
python setup.py install
```
### Option 2: Docker
You can download our pre-built Docker image for better reproducibility:
```bash
docker pull panisobe/dmd_flash_image_2:latest
```
You can use `docker run` to run the image. For example:
```bash
docker run -it \
  --shm-size=900g \
  --name hm \
  --network host \
  --ipc=host \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  -e HSA_OVERRIDE_GFX_VERSION=11.0.0 \
  -v /home:/home \
  panisobe/dmd_flash_image_2_release:latest
```
---

## 🚀 Getting Started for Video Generation
You can download the weights for all our models from our models' huggingface: [amd/HummingbirdXT](https://huggingface.co/amd/HummingbirdXT/tree/main).

### HummingbirdXT Video Generation

```bash
cd infer
bash run_t2v.sh # for text-to-video  task
bash run_i2v.sh # for image-to-video task
```

### HummingbirdXTX Long Video Generation
```bash
cd long_video
bash run.sh
```
---


## 🧪 Training & Implementation


### HummingbirdXT Training
First you need to enter the train folder:
```
cd train
```
**Step 1: Download the Teacher Model**
We use Wan2.2-TI2V-5B as the teacher model for step distillation.

```bash
pip install "huggingface_hub[hf_transfer]"
HF_HUB_ENABLE_HF_TRANSFER=1 --local-dir wan_models/Wan2.2-TI2V-5B
```
**Step 2: Prepare Training Datasets**
We train our models on a mixture of large-scale video datasets, including: MagicData,OpenVid,HumanVid

Please update the dataset root paths in the corresponding CSV files to match your local storage layout. You can download the csv file from our models' huggingface: [amd/HummingbirdXT](https://huggingface.co/amd/HummingbirdXT/tree/main).

**Step 3: Launch Training**
Start the step distillation training using the provided script:

```bash
bash running_scripts/train/dmd.sh
```
The training pipeline demonstrates stable loss convergence across all models.
> Reference Training Configuration:
> GPUs: 16 × AMD MI325, Iterations: 4000, Training time: ~48 hours.

### HummingbirdXTX Training

**Step 1: ODE Initialization(Optional)**
```bash
cd long_video
bash train_ode.sh
```
Or you can directly download our trained ODE initialization weights from our models' huggingface: [amd/HummingbirdXT](https://huggingface.co/amd/HummingbirdXT/tree/main), for the second stage of training.

**Step 2: DMD Training**
```bash
bash train_dmd.sh
```
---
## 📊 Experimental Results

<p align="center"><strong>Table 1. Quantitative results for the text-to-video task on VBench.</strong></p>

| Model                     | Quality Score ↑ | Semantic Score ↑ | Total Score ↑ |
|---------------------------|-----------------|------------------|-------------|
| Wan-2.2-5B-T2V w/o recap  | 82.75           | 68.38            | 79.88       |
| Wan-2.2-5B-T2V with recap | 83.99           | 77.04            | 82.60       |
| Ours-T2V w/o recap        | 84.07           | 54.75            | 78.20       |
| Ours-T2V with recap       | 85.71           | 72.33            | 83.03       |


<p align="center"><strong>Table 2. Quantitative results for the image-to-video task on VBench.</strong></p>

| Model                     | Video-Image Subject Consistency ↑ | Video-Image Background Consistency ↑ | Quality Score ↑ |
|---------------------------|-----------------------------------|--------------------------------------|-----------------|
| Wan-2.2-5B-I2V w/o recap   | 97.89                             | 99.04                                | 81.43           |
| Wan-2.2-5B-I2V with recap  | 97.63                             | 98.95                                | 81.06           |
| Ours-I2V w/o recap        | 98.46                             | 98.91                                | 80.01           |
| Ours-I2V with recap       | 98.42                             | 98.99                                | 80.57           |

<p align="center"><strong>Table 3. Runtime for generating a 121-frame video at 704×1280 resolution on server-grade (AMD Instinct™ MI300X and AMD Instinct™ MI325X GPU) and client-grade (Strix Halo and Navi48).</strong></p>

| Model          | MI300X | MI325X | Strix Halo iGPU | Navi48 dGPU |
|----------------|-------|-------|------------------|--------------|
| Wan-2.2-5B | 193.4s | 153.9s | 15000s           | OOM          |
| Ours | 6.5s  | 3.8s  | 460s             | 36.4s        |

<p align="center"><strong>Table 4. Performance and efficiency comparison of different VAE decoders on AMD Instinct™ MI300X GPU.</strong></p>

| Model      | LPIPS ↓ | PSNR ↑ | SSIM ↑ | RunTime ↓ | Memory ↓ |
|------------|---------|--------|--------|-----------|----------|
| Wan-2.2 VAE| 0.0141  | 35.979 | 0.9598 | 31.34s    | 11.37G   |
| TAEW2.2    | 0.0575  | 29.599 | 0.8953 | 0.14s     | 1.35G    |
| Ours VAE   | 0.0260  | 34.635 | 0.9483 | 2.29s     | 2.71G    |

<p align="center"><strong>Table 5. Quantitative results for long video generation on three benchmarks.</strong></p>

| Model          | FPS ↑ | Flicker Metric ↓ | DOVER ↑ | VBench Quality ↑ | VBench Semantic ↑ | VBench Total ↑ |
|----------------|-------------------|------------------|---------|----------------|-----------------|--------------|
| Self-Forcing    | 19.28             | 0.1010           | 84.37   | 81.99          | 80.09           | 81.61        |
| Causvid         | 18.24             | 0.0972           | 82.77   | 81.96          | 77.02           | 80.97        |
| LongLive       | 21.32             | 0.0947           | 84.07   | 82.86          | 81.61           | 82.61        |
| RollingForcing | 19.57             | 0.0928           | 85.16   | 82.94          | 80.61           | 82.47        |
| Ours           | 26.38             | 0.0946           | 84.55   | 83.42          | 79.22           | 82.58        |

---

## 🤗Additional Resources

Huggingface model cards: [AMD-HummingbirdXT](https://huggingface.co/amd/HummingbirdXT)

Full training code: [AMD-AIG-AIMA/HummingbirdXT](https://github.com/AMD-AGI/HummingbirdXT)

Related work on diffusion models by the AMD team:

- [AMD Hummingbird-0.9B: An Efficient Text-to-Video Diffusion Model with 4-Step Inferencing](https://rocm.blogs.amd.com/artificial-intelligence/image-to-video/README.html)
- [AMD Hummingbird Image to Video: A Lightweight Feedback-Driven Model for Efficient Image-to-Video Generation](https://rocm.blogs.amd.com/artificial-intelligence/image-to-video/README.html)

Please refer to the following resources to get started with training on AMD ROCm™ software:

- Use the [public PyTorch ROCm Docker images](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/pytorch-training.html) that enable optimized training performance out-of-the-box
- [PyTorch Fully Sharded Data Parallel (FSDP) on AMD GPUs with ROCm — ROCm Blogs](https://rocm.blogs.amd.com/artificial-intelligence/fsdp-training-pytorch/README.html)
- [Accelerating Large Language Models with Flash Attention on AMD GPUs — ROCm Blogs](https://rocm.blogs.amd.com/artificial-intelligence/flash-attention/README.html)
---

## ❤️ Acknowledgement

Our codebase builds on [Wan 2.1](https://github.com/Wan-Video/Wan2.1), [Wan 2.2](https://github.com/Wan-Video/Wan2.2), [Self-Forcing](https://github.com/guandeh17/Self-Forcing), [VideoX-Fun
](https://github.com/aigc-apps/VideoX-Fun).Thanks the authors for sharing their awesome codebases!

---

## 📋 Citations
Feel free to cite our Hummingbird models and give us a star⭐, if you find our work helpful. Thank you.