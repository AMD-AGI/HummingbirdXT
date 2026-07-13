# download from https://github.com/hao-ai-lab/FastVideo.git

from fastvideo.configs.sample.base import SamplingParam
from fastvideo.configs.sample.hunyuangamecraft import (
    HunyuanGameCraftSamplingParam,
    HunyuanGameCraft65FrameSamplingParam,
    HunyuanGameCraft129FrameSamplingParam,
)

__all__ = [
    "SamplingParam",
    "HunyuanGameCraftSamplingParam",
    "HunyuanGameCraft65FrameSamplingParam",
    "HunyuanGameCraft129FrameSamplingParam",
]
