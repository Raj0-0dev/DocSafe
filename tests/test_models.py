"""
Unit Test Suite for Module 3.1: Differentiable High-Frequency Residual Extractor
================================================================================
"""

import unittest
import torch
import numpy as np

from docsafe.models.residual import HighFrequencyResidualExtractor
from docsafe.core import inject_carrier_lattice, generate_carrier_lattice


class TestResidualExtractor(unittest.TestCase):

    def test_extractor_shape_and_dtype(self):
        """Submodule 3.1: Test tensor dimensions and data type preservation."""
        extractor = HighFrequencyResidualExtractor(channels=3, kernel_size=5, extract_luminance=True)

        batch_x = torch.rand(4, 3, 128, 128, dtype=torch.float32)
        residual = extractor(batch_x)

        self.assertEqual(residual.shape, (4, 1, 128, 128))
        self.assertEqual(residual.dtype, torch.float32)

    def test_flat_surface_zero_residual(self):
        """Submodule 3.1: Test that low-frequency uniform surfaces yield near-zero residuals."""
        extractor = HighFrequencyResidualExtractor(channels=3, kernel_size=5)

        # Uniform flat image (e.g. clean paper background)
        flat_img = torch.full((2, 3, 64, 64), 0.85, dtype=torch.float32)
        residual = extractor(flat_img)

        max_residual = torch.max(torch.abs(residual)).item()
        self.assertLess(max_residual, 1e-4, f"Residual of flat uniform image must be ~0, got {max_residual}")

    def test_carrier_preservation_snr(self):
        """Submodule 3.1: Prove that the extractor isolates the carrier lattice with high correlation."""
        extractor = HighFrequencyResidualExtractor(channels=3, kernel_size=5, extract_luminance=True)

        # Generate a protected image
        base_img = np.full((100, 100, 3), 180, dtype=np.uint8)
        key = "ExtractKey2026"
        prot_img, _ = inject_carrier_lattice(base_img, key, period=8, delta=1.5)

        # Convert to PyTorch tensor (1, 3, 100, 100) normalized to [0, 1]
        img_t = torch.from_numpy(prot_img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        residual_t = extractor(img_t) # (1, 1, 100, 100)

        # Theoretical carrier lattice
        carrier_np = generate_carrier_lattice(100, 100, key, period=8, channels=1).squeeze(-1)

        # Measure correlation between extracted residual and carrier
        res_np = residual_t.squeeze().detach().cpu().numpy()
        flat_r = res_np.flatten()
        flat_c = carrier_np.flatten()

        correlation = np.dot(flat_r, flat_c) / (np.linalg.norm(flat_r) * np.linalg.norm(flat_c) + 1e-8)
        print(f"\n[Residual Extractor Benchmark] Carrier Correlation in Extracted Residual: {correlation:.4f}")
        self.assertGreater(correlation, 0.85, "Extracted residual must preserve > 85% correlation with carrier!")

    def test_differentiability(self):
        """Submodule 3.1: Test that gradients flow back smoothly through the residual layer."""
        extractor = HighFrequencyResidualExtractor(channels=3, kernel_size=5, extract_luminance=True)
        x = torch.rand(2, 3, 32, 32, requires_grad=True)

        out = extractor(x)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad, "Gradients must propagate through the residual extractor!")
        self.assertEqual(x.grad.shape, (2, 3, 32, 32))

    def test_phase_coherence_fusion_shape_and_channels(self):
        """Submodule 3.2: Test 8-channel fused feature tensor output shape."""
        from docsafe.models import PhaseCoherenceFusionLayer

        fusion = PhaseCoherenceFusionLayer(pool_sizes=(4, 8, 16))
        img = torch.rand(2, 3, 128, 128, dtype=torch.float32)
        carrier = torch.choice = torch.where(torch.rand(2, 1, 128, 128) > 0.5, 1.0, -1.0)

        fused, res, coh = fusion(img, carrier)

        # Must have exactly 8 channels: 3 (image) + 1 (res) + 1 (coh) + 3 (pools)
        self.assertEqual(fused.shape, (2, 8, 128, 128))
        self.assertEqual(res.shape, (2, 1, 128, 128))
        self.assertEqual(coh.shape, (2, 1, 128, 128))
        self.assertFalse(torch.isnan(fused).any(), "Fused tensor must not contain NaNs")

    def test_coherence_contrast_authentic_vs_tampered(self):
        """Submodule 3.2: Prove that authentic areas produce positive coherence and tampered areas drop to ~0."""
        from docsafe.models import PhaseCoherenceFusionLayer
        from docsafe.core import inject_carrier_lattice, generate_carrier_lattice

        fusion = PhaseCoherenceFusionLayer()

        # 1. Create protected document image
        base_img = np.full((100, 100, 3), 200, dtype=np.uint8)
        key = "FusionProofKey"
        prot_img, _ = inject_carrier_lattice(base_img, key, period=8, delta=1.5)

        # 2. Simulate tampering in a 40x40 patch (erased / reset)
        tampered_img = prot_img.copy()
        tampered_img[30:70, 30:70] = 200 # Inpainted/reset to flat paper

        # Convert to torch tensors
        img_t = torch.from_numpy(tampered_img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        raw_car = generate_carrier_lattice(100, 100, key, period=8, channels=1)
        car_t = torch.from_numpy(raw_car.transpose(2, 0, 1)).float().unsqueeze(0)

        fused, res, coh = fusion(img_t, car_t)

        coh_np = coh.squeeze().detach().cpu().numpy()

        # Untouched area coherence (outside 30:70)
        mask = np.zeros((100, 100), dtype=bool)
        mask[30:70, 30:70] = True

        untouched_coh = float(np.mean(coh_np[~mask]))
        tampered_coh = float(np.mean(coh_np[mask]))

        print(f"\n[Coherence Fusion Proof] Untouched Coherence: {untouched_coh:.4f} vs. Tampered Coherence: {tampered_coh:.4f}")
        self.assertGreater(untouched_coh, 0.003, "Untouched area must have positive coherence!")
        self.assertLess(abs(tampered_coh), 0.0005, "Tampered area coherence must drop near zero!")

    def test_encoder_backbone_hierarchical_stages(self):
        """Submodule 4.1: Test 4-stage multi-scale encoder feature outputs."""
        import time
        from docsafe.models import TamperEncoderBackbone

        backbone = TamperEncoderBackbone(in_channels=8, dims=(32, 64, 128, 256))
        x = torch.rand(2, 8, 256, 256, dtype=torch.float32)

        start = time.perf_counter()
        e1, e2, e3, e4 = backbone(x)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        # Check multi-scale shapes
        self.assertEqual(e1.shape, (2, 32, 128, 128), "Stage 1 must be 1/2 resolution (128x128)")
        self.assertEqual(e2.shape, (2, 64, 64, 64),   "Stage 2 must be 1/4 resolution (64x64)")
        self.assertEqual(e3.shape, (2, 128, 32, 32),  "Stage 3 must be 1/8 resolution (32x32)")
        self.assertEqual(e4.shape, (2, 256, 16, 16),  "Stage 4 must be 1/16 resolution (16x16)")

        # Count parameters
        param_count = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        print(f"\n[Backbone Benchmark] Total Parameters: {param_count:,} | Forward Pass: {elapsed_ms:.2f} ms")
        self.assertLess(param_count, 2_000_000, "Encoder backbone must be compact (< 2M parameters)")

    def test_decoder_fpn_full_resolution_and_flow(self):
        """Submodule 4.2: Test Feature Pyramid Network Decoder full-resolution reconstruction."""
        import time
        from docsafe.models import TamperEncoderBackbone, TamperDecoderFPN

        backbone = TamperEncoderBackbone(in_channels=8, dims=(32, 64, 128, 256))
        decoder = TamperDecoderFPN(encoder_dims=(32, 64, 128, 256), out_dim=32)

        # Input tensor (B=2, C=8, H=128, W=128)
        x = torch.rand(2, 8, 128, 128, dtype=torch.float32, requires_grad=True)

        start = time.perf_counter()
        # 1. Forward through Encoder Backbone
        e1, e2, e3, e4 = backbone(x)
        # 2. Forward through FPN Decoder
        out_features = decoder(e1, e2, e3, e4)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        # Check full resolution shape
        self.assertEqual(out_features.shape, (2, 32, 128, 128), "Decoder must reconstruct exact full resolution (H, W)")

        # Verify gradient flow back to input
        loss = out_features.sum()
        loss.backward()
        self.assertIsNotNone(x.grad, "Gradients must flow back through all skip connections to the input!")

        # Count decoder parameters
        dec_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
        print(f"\n[FPN Decoder Benchmark] Decoder Parameters: {dec_params:,} | Full Loop: {elapsed_ms:.2f} ms")
        self.assertLess(dec_params, 1_500_000, "Decoder must be parameter-efficient (< 1.5M parameters)")

    def test_spatial_attention_and_head(self):
        """Submodule 4.3: Test Spatial Attention Gate and Sigmoid Segmentation Head."""
        from docsafe.models import SpatialAttentionGate, SegmentationHead

        x = torch.rand(2, 32, 64, 64)
        att = SpatialAttentionGate(kernel_size=7)
        gated, att_map = att(x)

        self.assertEqual(gated.shape, (2, 32, 64, 64))
        self.assertEqual(att_map.shape, (2, 1, 64, 64))
        self.assertTrue(0.0 <= att_map.min() and att_map.max() <= 1.0)

        head = SegmentationHead(in_channels=32, mid_channels=16)
        prob = head(gated)

        self.assertEqual(prob.shape, (2, 1, 64, 64))
        self.assertTrue(0.0 <= prob.min() and prob.max() <= 1.0, "Probabilities must be in [0.0, 1.0]")

    def test_docsafe_unet_detector_end_to_end(self):
        """Submodule 4.3: Test the unified DocSafeUNetDetector end-to-end model."""
        import time
        from docsafe.models import DocSafeUNetDetector

        model = DocSafeUNetDetector()
        img = torch.rand(2, 3, 128, 128, dtype=torch.float32)
        carrier = torch.where(torch.rand(2, 1, 128, 128) > 0.5, 1.0, -1.0)

        start = time.perf_counter()
        prob_mask, intermediates = model(img, carrier, return_intermediates=True)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        # Check output mask
        self.assertEqual(prob_mask.shape, (2, 1, 128, 128))
        self.assertTrue(0.0 <= prob_mask.min() and prob_mask.max() <= 1.0)

        # Check intermediate maps
        self.assertIn("residual", intermediates)
        self.assertIn("coherence", intermediates)
        self.assertIn("attention_map", intermediates)

        # Benchmark total parameters
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n[DocSafe UNet Benchmark] Total Model Parameters: {total_params:,} | Full Model Forward: {elapsed_ms:.2f} ms")
        self.assertLess(total_params, 2_500_000, "Unified model must be lightweight (< 2.5M parameters)!")


if __name__ == "__main__":
    unittest.main()
