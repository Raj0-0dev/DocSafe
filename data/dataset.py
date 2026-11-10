import os
from pathlib import Path
from typing import List, Tuple, Optional, Union

import cv2
import numpy as np

class ImageLoader:

    SUPPORTED_EXTENSIONS = {
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"
    }

    def __init__(
        self,
        target_size: Optional[Tuple[int, int]] = (256, 256),
        normalize: bool = True,
        preserve_aspect_ratio: bool = False,
    ):
        self.target_size = target_size
        self.normalize = normalize
        self.preserve_aspect_ratio = preserve_aspect_ratio

    @classmethod
    def discover_image_files(cls, directory_path: Union[str, Path]) -> List[Path]:
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

            for f in dir_p.rglob(f"*{ext.upper()}"):
                if f.is_file() and f.stat().st_size > 0:
                    real_path = str(f.resolve())
                    if real_path not in seen_names:
                        seen_names.add(real_path)
                        discovered.append(f)

        return sorted(discovered)

    def load_image(self, file_path: Union[str, Path]) -> np.ndarray:
        path_str = str(Path(file_path).resolve())
        bgr = cv2.imread(path_str, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Failed to read or decode image file: {path_str}")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if self.target_size is not None:
            rgb = self.resize_image(rgb, self.target_size, self.preserve_aspect_ratio)

        if self.normalize:
            return rgb.astype(np.float32) / 255.0
        return rgb

    @staticmethod
    def resize_image(
        img: np.ndarray,
        target_size: Tuple[int, int],
        preserve_aspect_ratio: bool = False,
    ) -> np.ndarray:
        target_h, target_w = target_size
        h, w = img.shape[:2]

        if (h, w) == (target_h, target_w):
            return img

        if not preserve_aspect_ratio:
            return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

        scale = min(target_w / float(w), target_h / float(h))
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        canvas = np.full((target_h, target_w, 3), 128, dtype=img.dtype)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

        return canvas

    @staticmethod
    def create_synthetic_sample(
        height: int = 256, width: int = 256, sample_type: str = "document"
    ) -> np.ndarray:
        canvas = np.full((height, width, 3), 245, dtype=np.uint8)

        if sample_type == "document":
            cv2.putText(canvas, "RECEIPT INVOICE", (int(width * 0.15), 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 2)
            cv2.line(canvas, (10, 45), (width - 10, 45), (180, 180, 180), 1)
            for y in range(70, height - 30, 25):
                cv2.putText(canvas, f"Item Ref {y}: $ {y*0.25:.2f}", (15, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1)
            cv2.putText(canvas, "TOTAL: $ 14.50", (15, height - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2)
        else:
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
        self.secret_key = secret_key
        self.target_size = target_size
        self.period = period
        self.delta = delta
        self.enable_distortions = enable_distortions
        self.virtual_length = virtual_length

        self.loader = ImageLoader(target_size=target_size, normalize=True)
        self.synthesizer = TamperSynthesizer(negative_sample_prob=negative_sample_prob)
        self.distorter = ChannelDistortions() if enable_distortions else None

        if image_paths is not None:
            self.files = [Path(p) for p in image_paths]
        elif image_dir is not None:
            self.files = ImageLoader.discover_image_files(image_dir)
        else:
            self.files = []

        h, w = target_size
        raw_carrier = generate_carrier_lattice(
            height=h, width=w, secret_key=secret_key, period=period, channels=1
        )
        self.carrier_tensor = torch.from_numpy(raw_carrier.transpose(2, 0, 1)).float()

    def __len__(self) -> int:
        if self.virtual_length is not None:
            return self.virtual_length
        return max(1, len(self.files))

    def __getitem__(self, idx: int) -> dict:
        h, w = self.target_size

        if self.files:
            file_idx = idx % len(self.files)
            try:
                base_img_f32 = self.loader.load_image(self.files[file_idx])
            except Exception:
                base_img_f32 = self.loader.create_synthetic_sample(h, w).astype(np.float32) / 255.0
        else:
            base_img_f32 = self.loader.create_synthetic_sample(h, w).astype(np.float32) / 255.0

        base_uint8 = np.clip(base_img_f32 * 255.0, 0, 255).astype(np.uint8)

        protected_uint8, _ = inject_carrier_lattice(
            image=base_uint8,
            secret_key=self.secret_key,
            period=self.period,
            delta=self.delta,
        )

        tampered_uint8, mask_f32, meta = self.synthesizer.generate_tampered_pair(
            protected_uint8, attack_type="random"
        )

        if self.enable_distortions and self.distorter is not None:
            final_uint8 = self.distorter.apply_channel_pipeline(tampered_uint8)
        else:
            final_uint8 = tampered_uint8

        img_f32 = final_uint8.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(img_f32.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask_f32[None, :, :]).float()

        return {
            "image": image_tensor,
            "carrier": self.carrier_tensor,
            "mask": mask_tensor,
            "meta": meta,
        }

def collate_tamper_batch(batch: List[dict]) -> dict:
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

