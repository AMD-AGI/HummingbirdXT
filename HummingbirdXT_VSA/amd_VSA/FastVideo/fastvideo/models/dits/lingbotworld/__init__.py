# download from https://github.com/hao-ai-lab/FastVideo.git

from .model import LingBotWorldTransformer3DModel

__all__ = [
    "LingBotWorldTransformer3DModel",
]

# Entry point for model registry
EntryClass = [LingBotWorldTransformer3DModel]
