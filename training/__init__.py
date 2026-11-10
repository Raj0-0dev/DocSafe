"""
DocSafe: Training & Optimization Engine
===========================================
"""

from docsafe.training.loss import (
    FocalLoss,
    DiceLoss,
    IoULoss,
    CompoundTamperLoss,
)
from docsafe.training.trainer import TamperDetectorTrainer
from docsafe.training.metrics import (
    compute_pixel_metrics,
    compute_image_level_decision,
    TamperMetricsEvaluator,
)

__all__ = [
    "FocalLoss",
    "DiceLoss",
    "IoULoss",
    "CompoundTamperLoss",
    "TamperDetectorTrainer",
    "compute_pixel_metrics",
    "compute_image_level_decision",
    "TamperMetricsEvaluator",
]
