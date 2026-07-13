# download from https://github.com/hao-ai-lab/FastVideo.git

from fastvideo.pipelines.basic.turbodiffusion.turbodiffusion_pipeline import (
    TurboDiffusionPipeline, )
from fastvideo.pipelines.basic.turbodiffusion.turbodiffusion_i2v_pipeline import (
    TurboDiffusionI2VPipeline, )

__all__ = ["TurboDiffusionPipeline", "TurboDiffusionI2VPipeline"]
