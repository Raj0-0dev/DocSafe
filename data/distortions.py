import random
from typing import Tuple, Optional, Union

import cv2
import numpy as np

class ChannelDistortions:

    def __init__(
        self,
        jpeg_prob: float = 0.70,
        jpeg_quality_range: Tuple[int, int] = (70, 92),
        sensor_noise_prob: float = 0.50,
        sensor_noise_sigma: Tuple[float, float] = (0.001, 0.004),
        photometric_prob: float = 0.40,
        seed: Optional[int] = None,
    ):
        self.jpeg_prob = jpeg_prob
        self.jpeg_quality_range = jpeg_quality_range
        self.sensor_noise_prob = sensor_noise_prob
        self.sensor_noise_sigma = sensor_noise_sigma
        self.photometric_prob = photometric_prob
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def apply_jpeg_compression(
        self,
        img: np.ndarray,
        quality: Optional[int] = None,
    ) -> np.ndarray:
        is_float = np.issubdtype(img.dtype, np.floating)
        if is_float:
            img_uint8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        else:
            img_uint8 = img.copy()

        if quality is None:
            quality = random.randint(self.jpeg_quality_range[0], self.jpeg_quality_range[1])

        bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)

        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        success, encoded_buf = cv2.imencode(".jpg", bgr, encode_params)
        if not success:
            return img.copy()

        decompressed_bgr = cv2.imdecode(encoded_buf, cv2.IMREAD_COLOR)
        decompressed_rgb = cv2.cvtColor(decompressed_bgr, cv2.COLOR_BGR2RGB)

        if is_float:
            return decompressed_rgb.astype(np.float32) / 255.0
        return decompressed_rgb

    def apply_sensor_noise(self, img: np.ndarray) -> np.ndarray:
        is_float = np.issubdtype(img.dtype, np.floating)
        sigma = random.uniform(self.sensor_noise_sigma[0], self.sensor_noise_sigma[1])

        if is_float:
            noise = np.random.normal(0.0, sigma, img.shape).astype(np.float32)
            noisy = np.clip(img + noise, 0.0, 1.0)
            return noisy
        else:
            noise_uint8 = np.random.normal(0.0, sigma * 255.0, img.shape).astype(np.float32)
            noisy = np.clip(img.astype(np.float32) + noise_uint8, 0, 255).astype(np.uint8)
            return noisy

    def apply_photometric_jitter(self, img: np.ndarray) -> np.ndarray:
        is_float = np.issubdtype(img.dtype, np.floating)
        alpha = random.uniform(0.97, 1.03)
        beta = random.uniform(-0.02, 0.02)

        if is_float:
            jittered = np.clip(img * alpha + beta, 0.0, 1.0)
            return jittered.astype(np.float32)
        else:
            jittered = np.clip(img.astype(np.float32) * alpha + (beta * 255.0), 0, 255)
            return jittered.astype(np.uint8)

    def apply_channel_pipeline(self, img: np.ndarray) -> np.ndarray:
        out = img.copy()

        if random.random() < self.photometric_prob:
            out = self.apply_photometric_jitter(out)

        if random.random() < self.sensor_noise_prob:
            out = self.apply_sensor_noise(out)

        if random.random() < self.jpeg_prob:
            out = self.apply_jpeg_compression(out)

        return out

