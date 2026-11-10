import hashlib
import cv2
import numpy as np
from typing import Tuple, Union

def derive_key_seed(secret_key: str) -> int:
    digest = hashlib.sha256(secret_key.encode('utf-8')).digest()
    seed = int.from_bytes(digest[:8], byteorder='big') % (2**31 - 1)
    return seed

def generate_carrier_lattice(
    height: int,
    width: int,
    secret_key: str,
    period: int = 8,
    channels: int = 3,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    seed = derive_key_seed(secret_key)
    rng = np.random.RandomState(seed)

    phi_x = rng.uniform(0.0, 2.0 * np.pi)
    phi_y = rng.uniform(0.0, 2.0 * np.pi)

    omega = 2.0 * np.pi / float(period)
    x = np.arange(width, dtype=np.float32)
    y = np.arange(height, dtype=np.float32)

    sin_x = np.sin(omega * x + phi_x)
    cos_y = np.cos(omega * y + phi_y)
    harmonic = np.outer(cos_y, sin_x)

    block_size = max(4, period * 2)
    bh = (height + block_size - 1) // block_size
    bw = (width + block_size - 1) // block_size
    block_signs = rng.choice([-1.0, 1.0], size=(bh, bw)).astype(np.float32)

    mod_grid = np.repeat(np.repeat(block_signs, block_size, axis=0)[:height], block_size, axis=1)[:, :width]

    bipolar_lattice = np.where(harmonic * mod_grid >= 0.0, 1.0, -1.0).astype(dtype)

    if channels == 1:
        return bipolar_lattice[:, :, None]
    elif channels == 3:
        return np.repeat(bipolar_lattice[:, :, None], 3, axis=2)
    else:
        raise ValueError(f"Unsupported channel count: {channels}. Supported: 1 or 3.")

def calculate_invisibility_metrics(
    original_img: np.ndarray, protected_img: np.ndarray
) -> dict:
    orig_f = original_img.astype(np.float64)
    prot_f = protected_img.astype(np.float64)

    mse = np.mean((orig_f - prot_f) ** 2)

    if mse == 0:
        psnr = float("inf")
    else:
        psnr = 10.0 * np.log10((255.0 ** 2) / mse)

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    mu1 = np.mean(orig_f)
    mu2 = np.mean(prot_f)
    sigma1_sq = np.var(orig_f)
    sigma2_sq = np.var(prot_f)
    sigma12 = np.mean((orig_f - mu1) * (prot_f - mu2))

    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2)
    )

    return {
        "mse": float(mse),
        "psnr_db": float(psnr),
        "ssim": float(ssim),
        "max_abs_diff": int(np.max(np.abs(orig_f - prot_f))),
    }

def inject_carrier_lattice(
    image: np.ndarray,
    secret_key: str,
    period: int = 8,
    delta: float = 1.0,
    chroma_mode: bool = False,
) -> Tuple[np.ndarray, dict]:
    if image.dtype != np.uint8:
        raise ValueError(f"Input image must be uint8, got {image.dtype}")

    is_grayscale = (image.ndim == 2) or (image.ndim == 3 and image.shape[2] == 1)
    h, w = image.shape[:2]

    if chroma_mode and not is_grayscale:
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        lattice_1ch = generate_carrier_lattice(
            height=h,
            width=w,
            secret_key=secret_key,
            period=period,
            channels=1,
            dtype=np.float32,
        ).squeeze()
        effective_delta = float(delta)
        ycrcb[:, :, 2] = np.clip(ycrcb[:, :, 2] + effective_delta * lattice_1ch, 0.0, 255.0)
        protected_uint8 = cv2.cvtColor(np.round(ycrcb).astype(np.uint8), cv2.COLOR_YCrCb2BGR)
        metrics = calculate_invisibility_metrics(image, protected_uint8)
        return protected_uint8, metrics

    channels = 1 if is_grayscale else 3

    lattice = generate_carrier_lattice(
        height=h,
        width=w,
        secret_key=secret_key,
        period=period,
        channels=channels,
        dtype=np.float32,
    )

    if is_grayscale and image.ndim == 2:
        img_reshaped = image[:, :, None].astype(np.float32)
    else:
        img_reshaped = image.astype(np.float32)

    protected_float = np.clip(img_reshaped + delta * lattice, 0.0, 255.0)
    protected_uint8 = np.round(protected_float).astype(np.uint8)

    if is_grayscale and image.ndim == 2:
        protected_uint8 = protected_uint8.squeeze(-1)

    metrics = calculate_invisibility_metrics(image, protected_uint8)

    return protected_uint8, metrics

def protect_image(
    input_path: str,
    secret_key: str,
    output_path: str,
    period: int = 8,
    delta: float = 1.0,
) -> dict:
    import os
    import cv2
    from pathlib import Path

    in_p = Path(input_path)
    if not in_p.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    img_bgr = cv2.imread(str(in_p))
    if img_bgr is None:
        raise ValueError(f"Failed to decode image from: {input_path}")

    protected_bgr, metrics = inject_carrier_lattice(
        image=img_bgr,
        secret_key=secret_key,
        period=period,
        delta=delta,
    )

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    ext = out_p.suffix.lower()
    if ext in [".jpg", ".jpeg"]:
        cv2.imwrite(str(out_p), protected_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 98])
    elif ext == ".png":
        cv2.imwrite(str(out_p), protected_bgr, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
    else:
        cv2.imwrite(str(out_p), protected_bgr)

    summary = {
        "status": "success",
        "input_file": in_p.name,
        "output_file": out_p.name,
        "resolution": f"{img_bgr.shape[1]}x{img_bgr.shape[0]}",
        "channels": img_bgr.shape[2] if img_bgr.ndim == 3 else 1,
        "period": period,
        "delta": delta,
        "metrics": metrics,
    }
    return summary

def protect_batch(
    input_dir: str,
    secret_key: str,
    output_dir: str,
    period: int = 8,
    delta: float = 1.0,
    max_images: int = None,
) -> list:
    import json
    from pathlib import Path
    from tqdm import tqdm

    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif"]
    files = []
    for pat in patterns:
        files.extend(list(in_dir.glob(pat)))
        files.extend(list(in_dir.glob(pat.upper())))

    seen = set()
    unique_files = [f for f in files if str(f) not in seen and not seen.add(str(f))]

    if max_images is not None:
        unique_files = unique_files[:max_images]

    manifest = []
    for f in tqdm(unique_files, desc="Protecting Batch", unit="img"):
        try:
            out_file = out_dir / f"protected_{f.stem}.png"
            summary = protect_image(
                input_path=str(f),
                secret_key=secret_key,
                output_path=str(out_file),
                period=period,
                delta=delta,
            )
            manifest.append(summary)
        except Exception as e:
            continue

    with open(out_dir / "protection_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest
