"""
DocSafe: Self-Supervised Data Engine
======================================
Module 2.1: Multi-Domain Image Dataset Loader & Preprocessing Pipeline.
Ingests camera photographs and document images with color-space normalization,
robust format decoding, and aspect-ratio preserving transformations.
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional, Union

import cv2
import numpy as np


class ImageLoader:
    """
    Submodule 2.1: High-reliability multi-domain image loader.
    Discovers, decodes, validates, and normalizes camera photographs
    and scanned document images across various formats.
    """

    SUPPORTED_EXTENSIONS = {
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"
    }

    def __init__(
        self,
        target_size: Optional[Tuple[int, int]] = (256, 256),
        normalize: bool = True,
        preserve_aspect_ratio: bool = False,
    ):
        """
        Parameters:
        -----------
        target_size : tuple of int, optional
            (height, width) for resizing/cropping. If None, original resolution is kept.
        normalize : bool, optional
            If True, converts pixels from [0, 255] uint8 to [0.0, 1.0] float32.
        preserve_aspect_ratio : bool, optional
            If True, resizes with letterbox padding rather than stretching.
        """
        self.target_size = target_size
        self.normalize = normalize
        self.preserve_aspect_ratio = preserve_aspect_ratio

    @classmethod
    def discover_image_files(cls, directory_path: Union[str, Path]) -> List[Path]:
        """
        Recursively discovers all valid image files in a given directory,
        deduplicating and filtering out corrupt or zero-byte files.
        """
        dir_p = Path(directory_path)
        if not dir_p.exists() or not dir_p.is_dir():
            return []

        discovered = []
        seen_names = set()

        for ext in cls.SUPPORTED_EXTENSIONS:
            for f in dir_p.rglob(f"*{ext}"):
                if f.is_file() and f.stat().st_size > 0:
                    real_path = str(f.resolve())
                    if real_path not in seen_names:
                        seen_names.add(real_path)
                        discovered.append(f)

            # Case-insensitive uppercase search for Windows/Linux
            for f in dir_p.rglob(f"*{ext.upper()}"):
                if f.is_file() and f.stat().st_size > 0:
                    real_path = str(f.resolve())
                    if real_path not in seen_names:
                        seen_names.add(real_path)
                        discovered.append(f)

        return sorted(discovered)

    def load_image(self, file_path: Union[str, Path]) -> np.ndarray:
        """
        Loads an image from disk, converts to RGB, and applies requested sizing and normalization.

        Returns:
        --------
        image : np.ndarray
            RGB array of shape (H, W, 3).
            If normalize=True, dtype is float32 with values in [0.0, 1.0].
            If normalize=False, dtype is uint8 with values in [0, 255].
        """
        path_str = str(Path(file_path).resolve())
        # Decode image using OpenCV
        bgr = cv2.imread(path_str, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Failed to read or decode image file: {path_str}")

        # Convert BGR (OpenCV default) to RGB (Deep Learning standard)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Apply target sizing if requested
        if self.target_size is not None:
            rgb = self.resize_image(rgb, self.target_size, self.preserve_aspect_ratio)

        # Apply normalization
        if self.normalize:
            return rgb.astype(np.float32) / 255.0
        return rgb

    @staticmethod
    def resize_image(
        img: np.ndarray,
        target_size: Tuple[int, int],
        preserve_aspect_ratio: bool = False,
    ) -> np.ndarray:
        """
        Resizes an image to target_size (target_h, target_w).
        """
        target_h, target_w = target_size
        h, w = img.shape[:2]

        if (h, w) == (target_h, target_w):
            return img

        if not preserve_aspect_ratio:
            return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

        # Aspect-ratio preserving resize with letterbox padding
        scale = min(target_w / float(w), target_h / float(h))
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Create padded canvas (neutral gray 128 for unnormalized images)
        canvas = np.full((target_h, target_w, 3), 128, dtype=img.dtype)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

        return canvas

    @staticmethod
    def create_synthetic_sample(
        height: int = 256, width: int = 256, sample_type: str = "document"
    ) -> np.ndarray:
        """
        Fallback synthetic image generator for testing and standalone validation.
        Produces either a synthetic document (text lines, receipt header)
        or a synthetic natural scene (gradients, geometric shapes).
        """
        canvas = np.full((height, width, 3), 245, dtype=np.uint8)

        if sample_type == "document":
            # Draw synthetic receipt text lines
            cv2.putText(canvas, "RECEIPT INVOICE", (int(width * 0.15), 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 2)
            cv2.line(canvas, (10, 45), (width - 10, 45), (180, 180, 180), 1)
            for y in range(70, height - 30, 25):
                cv2.putText(canvas, f"Item Ref {y}: $ {y*0.25:.2f}", (15, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1)
            cv2.putText(canvas, "TOTAL: $ 14.50", (15, height - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2)
        else:
            # Synthetic natural photography (color gradients and circular shapes)
            for y in range(height):
                r = int(120 + 80 * (y / height))
                b = int(220 - 100 * (y / height))
                canvas[y, :] = [r, 160, b]
            cv2.circle(canvas, (width // 2, height // 2), min(width, height) // 4, (255, 200, 50), -1)

        return canvas


import torch
from torch.utils.data import Dataset, DataLoader
from docsafe.core.lattice import generate_carrier_lattice, inject_carrier_lattice
from docsafe.data.attacks import TamperSynthesizer
from docsafe.data.distortions import ChannelDistortions


class TamperDetectionDataset(Dataset):
    """
    Submodule 2.4: PyTorch self-supervised dataset.
    Binds the multi-domain loader, carrier injector, tamper synthesizer,
    and channel distortions into an efficient streaming PyTorch Dataset.
    """

    def __init__(
        self,
        image_paths: Optional[List[Union[str, Path]]] = None,
        image_dir: Optional[Union[str, Path]] = None,
        secret_key: str = "DocSafeKey2026",
        target_size: Tuple[int, int] = (256, 256),
        period: int = 8,
        delta: float = 1.0,
        enable_distortions: bool = True,
        negative_sample_prob: float = 0.15,
        virtual_length: Optional[int] = None,
    ):
        """
        Parameters:
        -----------
        image_paths : list of paths, optional
            Explicit list of image file paths.
        image_dir : str or Path, optional
            Directory to discover image files from.
        secret_key : str
            Cryptographic key for carrier lattice generation.
        target_size : tuple of int
            (H, W) for image tensors.
        period : int
            Lattice period in pixels (default 8).
        delta : float
            Carrier amplitude (default 1.0).
        enable_distortions : bool
            Whether to apply benign WhatsApp JPEG compression and sensor noise.
        negative_sample_prob : float
            Fraction of clean, untouched samples.
        virtual_length : int, optional
            If set, loops or truncates the dataset to a fixed number of samples per epoch.
        """
        self.secret_key = secret_key
        self.target_size = target_size
        self.period = period
        self.delta = delta
        self.enable_distortions = enable_distortions
        self.virtual_length = virtual_length

        self.loader = ImageLoader(target_size=target_size, normalize=True)
        self.synthesizer = TamperSynthesizer(negative_sample_prob=negative_sample_prob)
        self.distorter = ChannelDistortions() if enable_distortions else None

        # Discover or assign files
        if image_paths is not None:
            self.files = [Path(p) for p in image_paths]
        elif image_dir is not None:
            self.files = ImageLoader.discover_image_files(image_dir)
        else:
            self.files = []

        # Precompute expected single-channel carrier tensor for the target dimensions
        h, w = target_size
        raw_carrier = generate_carrier_lattice(
            height=h, width=w, secret_key=secret_key, period=period, channels=1
        ) # (H, W, 1)
        # Convert to torch tensor (1, H, W)
        self.carrier_tensor = torch.from_numpy(raw_carrier.transpose(2, 0, 1)).float()

    def __len__(self) -> int:
        if self.virtual_length is not None:
            return self.virtual_length
        return max(1, len(self.files))

    def __getitem__(self, idx: int) -> dict:
        h, w = self.target_size

        # Load real image or generate fallback synthetic sample
        if self.files:
            file_idx = idx % len(self.files)
            try:
                base_img_f32 = self.loader.load_image(self.files[file_idx])
            except Exception:
                base_img_f32 = self.loader.create_synthetic_sample(h, w).astype(np.float32) / 255.0
        else:
            base_img_f32 = self.loader.create_synthetic_sample(h, w).astype(np.float32) / 255.0

        # Convert to uint8 for lattice injection
        base_uint8 = np.clip(base_img_f32 * 255.0, 0, 255).astype(np.uint8)

        # 1. Inject carrier lattice
        protected_uint8, _ = inject_carrier_lattice(
            image=base_uint8,
            secret_key=self.secret_key,
            period=self.period,
            delta=self.delta,
        )

        # 2. Apply synthetic tampering (attack)
        tampered_uint8, mask_f32, meta = self.synthesizer.generate_tampered_pair(
            protected_uint8, attack_type="random"
        )

        # 3. Apply benign channel distortions (WhatsApp JPEG, noise)
        if self.enable_distortions and self.distorter is not None:
            # Channel distortions affect the image, but do NOT change the ground truth mask
            final_uint8 = self.distorter.apply_channel_pipeline(tampered_uint8)
        else:
            final_uint8 = tampered_uint8

        # 4. Convert to PyTorch float32 tensors with channel-first ordering (C, H, W)
        img_f32 = final_uint8.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(img_f32.transpose(2, 0, 1)).float() # (3, H, W)
        mask_tensor = torch.from_numpy(mask_f32[None, :, :]).float()        # (1, H, W)

        return {
            "image": image_tensor,
            "carrier": self.carrier_tensor,
            "mask": mask_tensor,
            "meta": meta,
        }


def collate_tamper_batch(batch: List[dict]) -> dict:
    """
    Custom collator for TamperDetectionDataset:
    Stacks image, carrier, and mask tensors into batches (B, C, H, W)
    while preserving variable-length metadata dictionaries in a list.
    """
    images = torch.stack([item["image"] for item in batch], dim=0)
    carriers = torch.stack([item["carrier"] for item in batch], dim=0)
    masks = torch.stack([item["mask"] for item in batch], dim=0)
    metas = [item["meta"] for item in batch]

    return {
        "image": images,
        "carrier": carriers,
        "mask": masks,
        "meta": metas,
    }


def create_dataloader(
    dataset: Union[Dataset, str, Path],
    secret_key: str = "DocSafeKey2026",
    batch_size: int = 8,
    shuffle: bool = True,
    target_size: Tuple[int, int] = (256, 256),
    num_workers: int = 0,
    enable_distortions: bool = True,
) -> DataLoader:
    """
    Submodule 2.4: Factory function to create a ready-to-train PyTorch DataLoader.
    """
    if isinstance(dataset, (str, Path)):
        ds = TamperDetectionDataset(
            image_dir=dataset,
            secret_key=secret_key,
            target_size=target_size,
            enable_distortions=enable_distortions,
        )
    else:
        ds = dataset

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_tamper_batch,
        drop_last=True,
    )

