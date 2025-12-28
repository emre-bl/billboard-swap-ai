"""
Billboard Swap AI - Main Package

4-Stage Pipeline for automated billboard replacement in videos:
1. Segmentation - Detect and segment billboards
2. Tracking - Track billboard across video frames
3. Replacement - Warp replacement image onto billboard
4. Post-processing - Edge blending and temporal smoothing
"""

from .config import PipelineConfig, TrainingConfig, EvaluationConfig
from .pipeline import BillboardReplacementPipeline, FrameResult

__all__ = [
    "PipelineConfig",
    "TrainingConfig", 
    "EvaluationConfig",
    "BillboardReplacementPipeline",
    "FrameResult",
]
