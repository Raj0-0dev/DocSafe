"""
DocSafe: Self-Supervised Data Engine
======================================
Module 2.3: Benign Channel Distortions & Noise Simulator.
Simulates real-world transmission channels (WhatsApp / Social Media JPEG compression,
camera sensor readout noise, and subtle photometric jitter) during training.
Ensures the Deep Learning verifier learns semi-fragility: tolerating benign compression
while remaining hypersensitive to malicious semantic tampering.
"""

import random
from typing import Tuple, Optional, Union

import cv2
import numpy as np


class ChannelDistortions:
    """
    Submodule 2.3: Simulates benign channel transmissions and physical camera noise.
    """

    def __init__(
        self,
        jpeg_prob: float = 0.70,
        jpeg_quality_range: Tuple[int, int] = (70, 92),
        sensor_noise_prob: float = 0.50,
        sensor_noise_sigma: Tuple[float, float] = (0.001, 0.004),
        photometric_prob: float = 0.40,
        seed: Optional[int] = None,
    ):
        """
        Parameters:
        -----------
        jpeg_prob : float
            Probability of applying JPEG compression simulation.
        jpeg_quality_range : tuple of (int, int)
            Range of JPEG quality factors Q (e.g. 70-92, typical for WhatsApp/email).
        sensor_noise_prob : float
            Probability of injecting subtle camera sensor noise.
        sensor_noise_sigma : tuple of (float, float)
            Standard deviation range for Gaussian sensor noise (normalized to [0, 1]).
        photometric_prob : float
            Probability of subtle brightness/contrast jitter.
        """
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
        """
        Simulates in-memory lossy JPEG compression (e.g. WhatsApp / web re-encoding).

        Parameters:
        -----------
        img : np.ndarray
            Input image array (H, W, 3), float32 [0.0, 1.0] or uint8 [0, 255].
        quality : int, optional
            JPEG quality factor (1-100). If None, randomly sampled from quality_range.

        Returns:
        --------
        compressed_img : np.ndarray
            Re-encoded image with identical shape and dtype as input.
        """
        is_float = np.issubdtype(img.dtype, np.floating)
        if is_float:
            img_uint8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        else:
            img_uint8 = img.copy()

        if quality is None:
            quality = random.randint(self.jpeg_quality_range[0], self.jpeg_quality_range[1])

        # Convert RGB to BGR for OpenCV encoder
        bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)

        # In-memory JPEG encoding and decoding
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
        """
        Simulates subtle digital camera sensor readout noise (Gaussian noise).
        """
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
        """
        Simulates subtle brightness and contrast variations (+/- 3%).
        """
        is_float = np.issubdtype(img.dtype, np.floating)
        alpha = random.uniform(0.97, 1.03) # Contrast factor
        beta = random.uniform(-0.02, 0.02) # Brightness shift

        if is_float:
            jittered = np.clip(img * alpha + beta, 0.0, 1.0)
            return jittered.astype(np.float32)
        else:
            jittered = np.clip(img.astype(np.float32) * alpha + (beta * 255.0), 0, 255)
            return jittered.astype(np.uint8)

    def apply_channel_pipeline(self, img: np.ndarray) -> np.ndarray:
        """
        Applies stochastic combination of benign channel distortions.
        """
        out = img.copy()

        # 1. Subtle Photometric Jitter
        if random.random() < self.photometric_prob:
            out = self.apply_photometric_jitter(out)

        # 2. Camera Sensor Readout Noise
        if random.random() < self.sensor_noise_prob:
            out = self.apply_sensor_noise(out)

        # 3. WhatsApp / Social Media JPEG Compression
        if random.random() < self.jpeg_prob:
            out = self.apply_jpeg_compression(out)

        return out

