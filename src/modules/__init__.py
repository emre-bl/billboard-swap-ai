"""
Billboard Swap AI - Modules

Provides:
- segmentation: YOLOv8-seg, Mask R-CNN, GroundedSAM, SAM2
- tracking: SAM2, OpticalFlow, Cutie
- replacement: Perspective transform, Kalman smoothing, edge blending
- evaluation: IoU, mAP, precision, recall
"""

from .segmentation import (
    BaseSegmenter,
    YOLOSegmenter,
    MaskRCNNSegmenter,
    GroundedSAMSegmenter,
    SAM2Segmenter,
    create_segmenter,
)

from .tracking import (
    BaseTracker,
    SAM2Tracker,
    OpticalFlowTracker,
    CutieTracker,
    create_tracker,
)

from .replacement import (
    ReplacementEngine,
    KalmanCornerSmoother,
    EMACornerSmoother,
    CornerSmoother,
    apply_warp,
    order_points,
)

from .evaluation import (
    calculate_iou,
    calculate_precision_recall,
    calculate_ap,
    calculate_map,
    EvaluationMetrics,
    DetectionResult,
)

__all__ = [
    # Segmentation
    "BaseSegmenter",
    "YOLOSegmenter", 
    "MaskRCNNSegmenter",
    "GroundedSAMSegmenter",
    "SAM2Segmenter",
    "create_segmenter",
    # Tracking
    "BaseTracker",
    "SAM2Tracker",
    "OpticalFlowTracker",
    "CutieTracker",
    "create_tracker",
    # Replacement
    "ReplacementEngine",
    "KalmanCornerSmoother",
    "EMACornerSmoother",
    "CornerSmoother",
    "apply_warp",
    "order_points",
    # Evaluation
    "calculate_iou",
    "calculate_precision_recall",
    "calculate_ap",
    "calculate_map",
    "EvaluationMetrics",
    "DetectionResult",
]
