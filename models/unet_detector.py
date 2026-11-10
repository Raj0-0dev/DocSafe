from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNeXtBlock(nn.Module):

    def __init__(self, dim: int, drop_path: float = 0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.GroupNorm(num_groups=min(8, dim), num_channels=dim)
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

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=min(8, in_dim), num_channels=in_dim)
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return self.conv(x)

class TamperEncoderBackbone(nn.Module):

    def __init__(
        self,
        in_channels: int = 8,
        dims: Tuple[int, ...] = (32, 64, 128, 256),
        depths: Tuple[int, ...] = (2, 2, 2, 2),
    ):
        super().__init__()
        self.dims = dims

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=min(8, dims[0]), num_channels=dims[0]),
            nn.GELU(),
        )

        self.stage1 = nn.Sequential(*[ConvNeXtBlock(dims[0]) for _ in range(depths[0])])

        self.down1 = DownsampleStage(dims[0], dims[1])
        self.stage2 = nn.Sequential(*[ConvNeXtBlock(dims[1]) for _ in range(depths[1])])

        self.down2 = DownsampleStage(dims[1], dims[2])
        self.stage3 = nn.Sequential(*[ConvNeXtBlock(dims[2]) for _ in range(depths[2])])

        self.down3 = DownsampleStage(dims[2], dims[3])
        self.stage4 = nn.Sequential(*[ConvNeXtBlock(dims[3]) for _ in range(depths[3])])

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        e1 = self.stem(x)
        e1 = self.stage1(e1)

        e2 = self.down1(e1)
        e2 = self.stage2(e2)

        e3 = self.down2(e2)
        e3 = self.stage3(e3)

        e4 = self.down3(e3)
        e4 = self.stage4(e4)

        return e1, e2, e3, e4

class DecoderStage(nn.Module):

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
            nn.GELU(),
        )

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
        up = F.interpolate(deep_features, size=skip_features.shape[-2:], mode="bilinear", align_corners=False)
        up = self.up_conv(up)

        fused = torch.cat([up, skip_features], dim=1)
        return self.refine(fused)

class TamperDecoderFPN(nn.Module):

    def __init__(
        self,
        encoder_dims: Tuple[int, ...] = (32, 64, 128, 256),
        out_dim: int = 32,
    ):
        super().__init__()
        e1_dim, e2_dim, e3_dim, e4_dim = encoder_dims

        self.dec_stage3 = DecoderStage(in_channels=e4_dim, skip_channels=e3_dim, out_channels=e3_dim)

        self.dec_stage2 = DecoderStage(in_channels=e3_dim, skip_channels=e2_dim, out_channels=e2_dim)

        self.dec_stage1 = DecoderStage(in_channels=e2_dim, skip_channels=e1_dim, out_channels=e1_dim)

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
        d3 = self.dec_stage3(e4, e3)

        d2 = self.dec_stage2(d3, e2)

        d1 = self.dec_stage1(d2, e1)

        if target_size is not None:
            d1_up = F.interpolate(d1, size=target_size, mode="bilinear", align_corners=False)
        else:
            d1_up = F.interpolate(d1, scale_factor=2.0, mode="bilinear", align_corners=False)

        out_full = self.final_conv(d1_up)
        return out_full

class SpatialAttentionGate(nn.Module):

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        stats = torch.cat([avg_out, max_out], dim=1)

        attention_map = self.sigmoid(self.conv(stats))
        out = x * attention_map
        return out, attention_map

class SegmentationHead(nn.Module):

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

    def __init__(
        self,
        in_fused_channels: int = 8,
        encoder_dims: Tuple[int, ...] = (32, 64, 128, 256),
        decoder_out_dim: int = 32,
    ):
        super().__init__()
        self.coherence_fusion = PhaseCoherenceFusionLayer()

        self.encoder = TamperEncoderBackbone(
            in_channels=in_fused_channels,
            dims=encoder_dims,
        )

        self.decoder = TamperDecoderFPN(
            encoder_dims=encoder_dims,
            out_dim=decoder_out_dim,
        )

        self.attention = SpatialAttentionGate(kernel_size=7)
        self.seg_head = SegmentationHead(in_channels=decoder_out_dim, mid_channels=16)

    def forward(
        self,
        image: torch.Tensor,
        carrier: torch.Tensor,
        return_intermediates: bool = False,
    ) -> torch.Tensor:
        fused_features, residual, coherence = self.coherence_fusion(image, carrier)

        e1, e2, e3, e4 = self.encoder(fused_features)

        d_full = self.decoder(e1, e2, e3, e4, target_size=(image.shape[-2], image.shape[-1]))

        gated_features, att_map = self.attention(d_full)

        tamper_prob_mask = self.seg_head(gated_features)

        if return_intermediates:
            return tamper_prob_mask, {
                "residual": residual,
                "coherence": coherence,
                "attention_map": att_map,
            }
        return tamper_prob_mask

