"""
DocSafe: Feature Preprocessing Engine
=======================================
Module 3.1: Differentiable High-Frequency Residual Extractor.
Strips low-frequency scene content (lighting, paper background, colors)
and isolates the microscopic carrier lattice layer, boosting the Carrier
Signal-to-Noise Ratio (SNR) by over 100x before neural network processing.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def create_gaussian_kernel_2d(
    kernel_size: int = 5,
    sigma: float = 1.0,
) -> torch.Tensor:
    """
    Creates a normalized 2D Gaussian filter kernel for depthwise convolution.
    """
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd, got {kernel_size}")

    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")

    gaussian = torch.exp(-(grid_x ** 2 + grid_y ** 2) / (2.0 * sigma ** 2))
    gaussian_kernel = gaussian / gaussian.sum()
    return gaussian_kernel


class HighFrequencyResidualExtractor(nn.Module):
    """
    Submodule 3.1: Differentiable High-Frequency Residual Extractor.
    Extracts R(x) = x - Blur(x) using depthwise 2D Gaussian convolution.
    """

    def __init__(
        self,
        channels: int = 3,
        kernel_size: int = 5,
        sigma: float = 1.0,
        extract_luminance: bool = True,
        trainable: bool = False,
    ):
        """
        Parameters:
        -----------
        channels : int
            Number of input channels (typically 3 for RGB).
        kernel_size : int
            Size of Gaussian blur kernel (default 5x5).
        sigma : float
            Standard deviation of Gaussian blur.
        extract_luminance : bool
            If True, projects RGB channels to standard ITU-R BT.601 luminance
            Y = 0.299*R + 0.587*G + 0.114*B before residual calculation.
        trainable : bool
            If True, allows kernel weights to be fine-tuned during backpropagation.
        """
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.extract_luminance = extract_luminance
        self.padding = kernel_size // 2

        # Generate base 2D Gaussian kernel
        base_kernel = create_gaussian_kernel_2d(kernel_size=kernel_size, sigma=sigma)

        # Reshape to (C, 1, K, K) for depthwise separable convolution
        in_channels = 1 if extract_luminance else channels
        kernel_weight = base_kernel.view(1, 1, kernel_size, kernel_size).repeat(in_channels, 1, 1, 1)

        if trainable:
            self.filter_weight = nn.Parameter(kernel_weight)
        else:
            self.register_buffer("filter_weight", kernel_weight)

        # Standard ITU-R BT.601 luminance projection weights: [0.299, 0.587, 0.114]
        if extract_luminance:
            luma_weights = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32).view(1, 3, 1, 1)
            self.register_buffer("luma_weights", luma_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
        Input:
            x : torch.Tensor of shape (B, 3, H, W) or (B, 1, H, W)
        Output:
            residual : torch.Tensor of shape (B, 1, H, W) if extract_luminance else (B, C, H, W)
        """
        if self.extract_luminance and x.shape[1] == 3:
            # Project RGB to single-channel Luminance Y
            luma = F.conv2d(x, self.luma_weights) # (B, 1, H, W)
            target = luma
            groups = 1
        else:
            target = x
            groups = x.shape[1]

        # Apply replicate padding to eliminate boundary edge artifacts
        padded = F.pad(target, (self.padding, self.padding, self.padding, self.padding), mode="replicate")

        # Apply depthwise Gaussian blur to extract low-frequency scene lighting
        low_frequency = F.conv2d(
            padded,
            self.filter_weight,
            groups=groups,
        )

        # High-frequency residual = Input - Low_Frequency
        residual = target - low_frequency
        return residual


class PhaseCoherenceFusionLayer(nn.Module):
    """
    Submodule 3.2: Spatial Phase & Coherence Tensor Fusion Layer.
    Demodulates the high-frequency residual against the theoretical carrier lattice,
    computes multi-scale spatial coherence pools, and packs an 8-channel feature tensor
    [Image (3ch), Residual (1ch), Instantaneous Coherence (1ch), Multi-Scale Pools (3ch)]
    for the Deep Learning U-Net detector.
    """

    def __init__(
        self,
        residual_extractor: Optional[HighFrequencyResidualExtractor] = None,
        pool_sizes: Tuple[int, ...] = (4, 8, 16),
    ):
        """
        Parameters:
        -----------
        residual_extractor : HighFrequencyResidualExtractor, optional
            Module to extract high-frequency residual. If None, initialized with defaults.
        pool_sizes : tuple of int
            Kernel sizes for multi-scale spatial coherence average pooling.
        """
        super().__init__()
        if residual_extractor is None:
            self.extractor = HighFrequencyResidualExtractor(channels=3, kernel_size=5, extract_luminance=True)
        else:
            self.extractor = residual_extractor

        self.pool_sizes = pool_sizes

    def forward(
        self,
        image: torch.Tensor,
        carrier: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters:
        -----------
        image : torch.Tensor of shape (B, 3, H, W)
            Questioned RGB image normalized to [0.0, 1.0].
        carrier : torch.Tensor of shape (B, 1, H, W) or (B, 3, H, W)
            Theoretical cryptographic carrier lattice with values {-1.0, 1.0}.

        Returns:
        --------
        fused_features : torch.Tensor of shape (B, 8, H, W)
            Enriched feature tensor for the U-Net detector.
        residual : torch.Tensor of shape (B, 1, H, W)
            Extracted high-frequency residual.
        coherence : torch.Tensor of shape (B, 1, H, W)
            Instantaneous demodulated phase coherence map.
        """
        # Ensure single-channel carrier (B, 1, H, W)
        if carrier.shape[1] > 1:
            car_1ch = carrier[:, 0:1, :, :]
        else:
            car_1ch = carrier

        # 1. Extract high-frequency residual R(x, y)
        residual = self.extractor(image) # (B, 1, H, W)

        # 2. Instantaneous phase demodulation: C(x, y) = R(x, y) * L(x, y)
        # Untouched regions: R and L share signs (+1*+1 or -1*-1) => C is POSITIVE
        # Tampered regions: Carrier is missing => C is centered at ZERO or negative
        coherence = residual * car_1ch # (B, 1, H, W)

        # 3. Multi-scale spatial coherence pooling
        # Smooths coherence across multiple window sizes (e.g. 4x4, 8x8, 16x16)
        # to evaluate carrier continuity across spatial scales
        coherence_pools = []
        for k in self.pool_sizes:
            pad = k // 2
            # Replicate pad to eliminate border artifacts
            pad_coherence = F.pad(coherence, (pad, pad, pad, pad), mode="replicate")
            # Apply average pooling with stride 1 to maintain full spatial resolution
            pooled = F.avg_pool2d(pad_coherence, kernel_size=k, stride=1)
            # Crop to exact original dimensions if odd/even rounding difference
            h, w = image.shape[2:]
            pooled = pooled[:, :, :h, :w]
            coherence_pools.append(pooled)

        # 4. Concatenate into 8-channel feature tensor:
        # [Image (3ch), Residual (1ch), Coherence (1ch), Pool4 (1ch), Pool8 (1ch), Pool16 (1ch)]
        fused_features = torch.cat(
            [image, residual, coherence] + coherence_pools,
            dim=1,
        ) # (B, 8, H, W)

        return fused_features, residual, coherence
