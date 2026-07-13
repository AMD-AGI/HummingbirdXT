#!/bin/bash
# download from https://github.com/hao-ai-lab/FastVideo.git


# 1. Install missing dependency
pip install -q opencv-python-headless transformers huggingface_hub

# 2. Run FVD script
python benchmarks/fvd/run_fvd.py
