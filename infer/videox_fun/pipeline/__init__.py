# Modifications Copyright(C)[2026] Advanced Micro Devices, Inc. All rights reserved.
from .pipeline_wan2_2_ti2v import Wan2_2TI2VPipeline, FlowShiftScheduler
import importlib.util

if importlib.util.find_spec("paifuser") is not None:
    # --------------------------------------------------------------- #
    #   Sparse Attention
    # --------------------------------------------------------------- #
    from paifuser.ops import sparse_reset


    Wan2_2TI2VPipeline.__call__ = sparse_reset(Wan2_2TI2VPipeline.__call__)
