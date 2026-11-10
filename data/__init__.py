"""
DocSafe Data Engine
======================
Multi-domain dataset loading, synthetic tamper attacks, channel distortions,
and PyTorch DataLoaders.
"""

from docsafe.data.dataset import (
    ImageLoader,
    TamperDetectionDataset,
    create_dataloader,
)
from docsafe.data.attacks import TamperSynthesizer
from docsafe.data.distortions import ChannelDistortions

__all__ = [
    "ImageLoader",
    "TamperDetectionDataset",
    "create_dataloader",
    "TamperSynthesizer",
    "ChannelDistortions",
]
