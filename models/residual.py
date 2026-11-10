import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

def create_gaussian_kernel_2d(
    kernel_size: int = 5,
    sigma: float = 1.0,
) -> torch.Tensor:
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd, got {kernel_size}")

    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")

    gaussian = torch.exp(-(grid_x ** 2 + grid_y ** 2) / (2.0 * sigma ** 2))
    gaussian_kernel = gaussian / gaussian.sum()
    return gaussian_kernel

class HighFrequencyResidualExtractor(nn.Module):

    def __init__(
        self,
        channels: int = 3,
        kernel_size: int = 5,
        sigma: float = 1.0,
        extract_luminance: bool = True,
        trainable: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.extract_luminance = extract_luminance
        self.padding = kernel_size // 2

        base_kernel = create_gaussian_kernel_2d(kernel_size=kernel_size, sigma=sigma)

        in_channels = 1 if extract_luminance else channels
        kernel_weight = base_kernel.view(1, 1, kernel_size, kernel_size).repeat(in_channels, 1, 1, 1)

        if trainable:
            self.filter_weight = nn.Parameter(kernel_weight)
        else:
            self.register_buffer("filter_weight", kernel_weight)

        if extract_luminance:
            luma_weights = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32).view(1, 3, 1, 1)
            self.register_buffer("luma_weights", luma_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.extract_luminance and x.shape[1] == 3:
            luma = F.conv2d(x, self.luma_weights)
            target = luma
            groups = 1
        else:
            target = x
            groups = x.shape[1]

        padded = F.pad(target, (self.padding, self.padding, self.padding, self.padding), mode="replicate")

        low_frequency = F.conv2d(
            padded,
            self.filter_weight,
            groups=groups,
        )

        residual = target - low_frequency
        return residual

class PhaseCoherenceFusionLayer(nn.Module):

    def __init__(
        self,
        residual_extractor: Optional[HighFrequencyResidualExtractor] = None,
        pool_sizes: Tuple[int, ...] = (4, 8, 16),
    ):
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
        if carrier.shape[1] > 1:
            car_1ch = carrier[:, 0:1, :, :]
        else:
            car_1ch = carrier

        residual = self.extractor(image)

        coherence = residual * car_1ch

        coherence_pools = []
        for k in self.pool_sizes:
            pad = k // 2
            pad_coherence = F.pad(coherence, (pad, pad, pad, pad), mode="replicate")
            pooled = F.avg_pool2d(pad_coherence, kernel_size=k, stride=1)
            h, w = image.shape[2:]
            pooled = pooled[:, :, :h, :w]
            coherence_pools.append(pooled)

        fused_features = torch.cat(
            [image, residual, coherence] + coherence_pools,
            dim=1,
        )

        return fused_features, residual, coherence
