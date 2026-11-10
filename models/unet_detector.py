"""
DocSafe: Deep Learning Tamper Segmentation Network
=====================================================
Module 4.1: Multi-Scale Hierarchical Encoder Backbone.
Extracts multi-resolution latent feature representations (from sub-character
stroke precision at 1/2 resolution to global scene context at 1/16 resolution)
using parameter-efficient ConvNeXt-style residual depthwise separable blocks.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNeXtBlock(nn.Module):
    """
    Submodule 4.1: Modern Residual Depthwise Separable Convolution Block.
    Combines 7x7 depthwise convolution, GroupNorm, GELU activation,
    and 1x1 pointwise expansion/projection with an identity skip connection.
    """

    def __init__(self, dim: int, drop_path: float = 0.0):
        super().__init__()
        # Depthwise 7x7 convolution with large spatial receptive field
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        # GroupNorm with 8 groups for stable small-batch training (CPU & GPU)
        self.norm = nn.GroupNorm(num_groups=min(8, dim), num_channels=dim)
        # Pointwise projection: inverted bottleneck expansion (dim -> 4*dim -> dim)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.dwconv(x)
        out = self.norm(out)
        out = self.pwconv1(out)
        out = self.act(out)
        out = self.pwconv2(out)
        return residual + out


class DownsampleStage(nn.Module):
    """
    Submodule 4.1: Strided downsampling layer between hierarchical stages.
    Reduces spatial resolution by 2x and doubles feature channels.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=min(8, in_dim), num_channels=in_dim)
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return self.conv(x)


class TamperEncoderBackbone(nn.Module):
    """
    Submodule 4.1: 4-Stage Hierarchical Multi-Scale Encoder Backbone.

    Inputs:
        x : (B, in_channels, H, W)
            Typically 8 channels from PhaseCoherenceFusionLayer
            [Image (3), Residual (1), Coherence (1), Multi-Scale Pools (3)].

    Outputs:
        Tuple of 4 multi-scale feature maps for the U-Net Feature Pyramid:
            - E1: (B, 32, H/2, W/2)   -> Microscopic sub-character strokes
            - E2: (B, 64, H/4, W/4)   -> Character and digit boundaries
            - E3: (B, 128, H/8, W/8)  -> Word and line-item context
            - E4: (B, 256, H/16, W/16)-> Global document & scene semantics
    """

    def __init__(
        self,
        in_channels: int = 8,
        dims: Tuple[int, ...] = (32, 64, 128, 256),
        depths: Tuple[int, ...] = (2, 2, 2, 2),
    ):
        super().__init__()
        self.dims = dims

        # Initial stem: projects 8-channel input to 1/2 spatial resolution
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=min(8, dims[0]), num_channels=dims[0]),
            nn.GELU(),
        )

        # Stage 1: (B, 32, H/2, W/2)
        self.stage1 = nn.Sequential(*[ConvNeXtBlock(dims[0]) for _ in range(depths[0])])

        # Stage 2: (B, 64, H/4, W/4)
        self.down1 = DownsampleStage(dims[0], dims[1])
        self.stage2 = nn.Sequential(*[ConvNeXtBlock(dims[1]) for _ in range(depths[1])])

        # Stage 3: (B, 128, H/8, W/8)
        self.down2 = DownsampleStage(dims[1], dims[2])
        self.stage3 = nn.Sequential(*[ConvNeXtBlock(dims[2]) for _ in range(depths[2])])

        # Stage 4: (B, 256, H/16, W/16)
        self.down3 = DownsampleStage(dims[2], dims[3])
        self.stage4 = nn.Sequential(*[ConvNeXtBlock(dims[3]) for _ in range(depths[3])])

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass extracting hierarchical feature representations.
        """
        # Stem -> Stage 1 (1/2 resolution)
        e1 = self.stem(x)
        e1 = self.stage1(e1)

        # Downsample -> Stage 2 (1/4 resolution)
        e2 = self.down1(e1)
        e2 = self.stage2(e2)

        # Downsample -> Stage 3 (1/8 resolution)
        e3 = self.down2(e2)
        e3 = self.stage3(e3)

        # Downsample -> Stage 4 (1/16 resolution)
        e4 = self.down3(e3)
        e4 = self.stage4(e4)

        return e1, e2, e3, e4


class DecoderStage(nn.Module):
    """
    Submodule 4.2: U-Net Decoder stage with lateral skip connection fusion.
    Bilinearly upsamples deeper features by 2x, concatenates with lateral encoder
    skip connections, and refines representations using residual convolution blocks.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        # 1x1 projection for upsampled deep features
        self.up_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
            nn.GELU(),
        )

        # Refinement block for fused (upsampled + skip) features
        fused_channels = out_channels + skip_channels
        self.refine = nn.Sequential(
            nn.Conv2d(fused_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
            nn.GELU(),
        )

    def forward(self, deep_features: torch.Tensor, skip_features: torch.Tensor) -> torch.Tensor:
        # Robust interpolation matching exact spatial shape of skip_features
        up = F.interpolate(deep_features, size=skip_features.shape[-2:], mode="bilinear", align_corners=False)
        up = self.up_conv(up)

        # Lateral skip concatenation
        fused = torch.cat([up, skip_features], dim=1)
        return self.refine(fused)


class TamperDecoderFPN(nn.Module):
    """
    Submodule 4.2: Feature Pyramid Network (FPN) Decoder for U-Net.
    Progressively upsamples and fuses (E4, E3, E2, E1) back to full image resolution (H, W).

    Inputs:
        (e1, e2, e3, e4) from TamperEncoderBackbone:
            - e1: (B, 32, H/2, W/2)
            - e2: (B, 64, H/4, W/4)
            - e3: (B, 128, H/8, W/8)
            - e4: (B, 256, H/16, W/16)

    Output:
        Full-resolution feature tensor: (B, out_dim=32, H, W)
    """

    def __init__(
        self,
        encoder_dims: Tuple[int, ...] = (32, 64, 128, 256),
        out_dim: int = 32,
    ):
        super().__init__()
        e1_dim, e2_dim, e3_dim, e4_dim = encoder_dims

        # Stage 4 -> 3 (1/16 to 1/8)
        self.dec_stage3 = DecoderStage(in_channels=e4_dim, skip_channels=e3_dim, out_channels=e3_dim)

        # Stage 3 -> 2 (1/8 to 1/4)
        self.dec_stage2 = DecoderStage(in_channels=e3_dim, skip_channels=e2_dim, out_channels=e2_dim)

        # Stage 2 -> 1 (1/4 to 1/2)
        self.dec_stage1 = DecoderStage(in_channels=e2_dim, skip_channels=e1_dim, out_channels=e1_dim)

        # Final Stage: 1/2 to Full Resolution (H, W)
        self.final_conv = nn.Sequential(
            nn.Conv2d(e1_dim, out_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(8, out_dim), num_channels=out_dim),
            nn.GELU(),
        )

    def forward(
        self,
        e1: torch.Tensor,
        e2: torch.Tensor,
        e3: torch.Tensor,
        e4: torch.Tensor,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        # Fuse E4 (1/16) + E3 (1/8) -> D3 (1/8)
        d3 = self.dec_stage3(e4, e3)

        # Fuse D3 (1/8) + E2 (1/4) -> D2 (1/4)
        d2 = self.dec_stage2(d3, e2)

        # Fuse D2 (1/4) + E1 (1/2) -> D1 (1/2)
        d1 = self.dec_stage1(d2, e1)

        # Final upsample to target full resolution (H, W)
        if target_size is not None:
            d1_up = F.interpolate(d1, size=target_size, mode="bilinear", align_corners=False)
        else:
            d1_up = F.interpolate(d1, scale_factor=2.0, mode="bilinear", align_corners=False)

        out_full = self.final_conv(d1_up)
        return out_full


class SpatialAttentionGate(nn.Module):
    """
    Submodule 4.3: Spatial Attention Gate.
    Calculates channel-wise statistics (mean and max pooling across feature channels),
    applies 7x7 spatial convolution, and outputs a spatial attention weight map
    alpha in [0.0, 1.0] to suppress background texture noise and focus on carrier anomalies.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        # Input has 2 channels: (ChannelMax + ChannelAvg)
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Spatial channel statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        stats = torch.cat([avg_out, max_out], dim=1)

        # 7x7 spatial attention coefficient map
        attention_map = self.sigmoid(self.conv(stats))
        # Gated feature modulation
        out = x * attention_map
        return out, attention_map


class SegmentationHead(nn.Module):
    """
    Submodule 4.3: Dense Pixel-Wise Sigmoid Segmentation Head.
    Projects refined feature representations into a single-channel
    calibrated tamper probability mask M_hat in [0.0, 1.0]^(1 x H x W).
    """

    def __init__(self, in_channels: int = 32, mid_channels: int = 16):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(8, mid_channels), num_channels=mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


from docsafe.models.residual import PhaseCoherenceFusionLayer


class DocSafeUNetDetector(nn.Module):
    """
    Submodule 4.3: Unified End-to-End Deep Learning Tamper Detector.
    Integrates the Phase Coherence Fusion Layer (Module 3), Multi-Scale
    Encoder Backbone (Submodule 4.1), Feature Pyramid Decoder (Submodule 4.2),
    Spatial Attention Gate, and Dense Segmentation Head (Submodule 4.3).

    Single Unified Interface:
        Input:
            image   : (B, 3, H, W) - Questioned RGB image in [0.0, 1.0]
            carrier : (B, 1, H, W) - Expected carrier lattice in {-1.0, 1.0}
        Output:
            prob_mask : (B, 1, H, W) - Tamper probability mask in [0.0, 1.0]
    """

    def __init__(
        self,
        in_fused_channels: int = 8,
        encoder_dims: Tuple[int, ...] = (32, 64, 128, 256),
        decoder_out_dim: int = 32,
    ):
        super().__init__()
        # Module 3: Phase coherence feature fusion
        self.coherence_fusion = PhaseCoherenceFusionLayer()

        # Submodule 4.1: Multi-scale hierarchical encoder backbone
        self.encoder = TamperEncoderBackbone(
            in_channels=in_fused_channels,
            dims=encoder_dims,
        )

        # Submodule 4.2: Feature Pyramid Network decoder
        self.decoder = TamperDecoderFPN(
            encoder_dims=encoder_dims,
            out_dim=decoder_out_dim,
        )

        # Submodule 4.3: Spatial Attention Gate & Dense Segmentation Head
        self.attention = SpatialAttentionGate(kernel_size=7)
        self.seg_head = SegmentationHead(in_channels=decoder_out_dim, mid_channels=16)

    def forward(
        self,
        image: torch.Tensor,
        carrier: torch.Tensor,
        return_intermediates: bool = False,
    ) -> torch.Tensor:
        # 1. Feature Preprocessing & Phase Coherence Fusion (B, 8, H, W)
        fused_features, residual, coherence = self.coherence_fusion(image, carrier)

        # 2. Multi-Scale Hierarchical Feature Extraction
        e1, e2, e3, e4 = self.encoder(fused_features)

        # 3. Feature Pyramid Decoder Reconstruction (B, 32, H, W)
        d_full = self.decoder(e1, e2, e3, e4, target_size=(image.shape[-2], image.shape[-1]))

        # 4. Spatial Attention Gating
        gated_features, att_map = self.attention(d_full)

        # 5. Dense Pixel-Wise Sigmoid Classification Head (B, 1, H, W)
        tamper_prob_mask = self.seg_head(gated_features)

        if return_intermediates:
            return tamper_prob_mask, {
                "residual": residual,
                "coherence": coherence,
                "attention_map": att_map,
            }
        return tamper_prob_mask

