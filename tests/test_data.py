"""
Unit Test Suite for Module 2.1: Multi-Domain Image Dataset Loader
================================================================
"""

import os
import tempfile
import unittest
import cv2
import numpy as np

from docsafe.data.dataset import ImageLoader


class TestImageLoader(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root = self.tmp_dir.name

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_discover_image_files(self):
        """Test recursive discovery, case insensitivity, and zero-byte filtering."""
        # Create nested folders
        sub1 = os.path.join(self.root, "sub1")
        sub2 = os.path.join(self.root, "sub2")
        os.makedirs(sub1)
        os.makedirs(sub2)

        # 1. Valid image in root
        dummy = np.full((50, 50, 3), 120, dtype=np.uint8)
        cv2.imwrite(os.path.join(self.root, "img1.png"), dummy)

        # 2. Valid image with uppercase extension in sub1
        cv2.imwrite(os.path.join(sub1, "img2.JPG"), dummy)

        # 3. Non-image file in sub2
        with open(os.path.join(sub2, "notes.txt"), "w") as f:
            f.write("text file")

        # 4. Zero-byte image file
        with open(os.path.join(sub2, "corrupt.png"), "w") as f:
            pass

        discovered = ImageLoader.discover_image_files(self.root)
        self.assertEqual(len(discovered), 2, "Only valid, non-zero image files must be discovered")

    def test_image_loading_and_normalization(self):
        """Test RGB conversion, float32 scaling [0.0, 1.0], and target sizing."""
        img_path = os.path.join(self.root, "test_bgr.png")
        # Create image with pure Red in BGR (0, 0, 255)
        pure_red_bgr = np.zeros((100, 150, 3), dtype=np.uint8)
        pure_red_bgr[:, :] = [0, 0, 255] # B=0, G=0, R=255
        cv2.imwrite(img_path, pure_red_bgr)

        loader = ImageLoader(target_size=(128, 128), normalize=True)
        loaded = loader.load_image(img_path)

        # Check shape and dtype
        self.assertEqual(loaded.shape, (128, 128, 3))
        self.assertEqual(loaded.dtype, np.float32)

        # Check RGB ordering (R channel should be 1.0, B channel should be 0.0)
        self.assertAlmostEqual(float(loaded[50, 50, 0]), 1.0, places=3, msg="Channel 0 must be Red")
        self.assertAlmostEqual(float(loaded[50, 50, 2]), 0.0, places=3, msg="Channel 2 must be Blue")

        # Check value bounds
        self.assertTrue(0.0 <= loaded.min() and loaded.max() <= 1.0)

    def test_aspect_ratio_preserving_resize(self):
        """Test letterbox padding mode preserves aspect ratio without squishing."""
        tall_img = np.full((200, 100, 3), 200, dtype=np.uint8)
        loader = ImageLoader(target_size=(100, 100), normalize=False, preserve_aspect_ratio=True)
        resized = loader.resize_image(tall_img, (100, 100), preserve_aspect_ratio=True)

        self.assertEqual(resized.shape, (100, 100, 3))
        # Since height was 200 and width was 100, aspect ratio is 2:1.
        # Inside a 100x100 canvas, new_h=100, new_w=50, with left/right padding of 25px
        # Padding pixels should be 128
        self.assertEqual(int(resized[50, 10, 0]), 128, "Left padding area must have neutral fill")
        self.assertEqual(int(resized[50, 50, 0]), 200, "Center content area must have original pixel value")

    def test_synthetic_sample_generation(self):
        """Test fallback synthetic sample generator for testing."""
        doc = ImageLoader.create_synthetic_sample(200, 200, sample_type="document")
        photo = ImageLoader.create_synthetic_sample(200, 200, sample_type="photo")

        self.assertEqual(doc.shape, (200, 200, 3))
        self.assertEqual(photo.shape, (200, 200, 3))
        self.assertEqual(doc.dtype, np.uint8)
        self.assertEqual(photo.dtype, np.uint8)

    def test_tamper_synthesizer_profiles(self):
        """Submodule 2.2: Test box, brushstroke, splice, and negative attack profiles."""
        from docsafe.data.attacks import TamperSynthesizer
        from docsafe.core import inject_carrier_lattice

        synthesizer = TamperSynthesizer(seed=42)
        base_img = np.full((200, 200, 3), 240, dtype=np.uint8)
        prot_img, _ = inject_carrier_lattice(base_img, "TestSecretKey", period=8)

        # 1. Box attack
        t_box, m_box, meta_box = synthesizer.generate_tampered_pair(prot_img, attack_type="box")
        self.assertEqual(t_box.shape, (200, 200, 3))
        self.assertEqual(m_box.shape, (200, 200))
        self.assertGreater(meta_box["coverage_pct"], 0.0)
        self.assertTrue(set(np.unique(m_box)).issubset({0.0, 1.0}))

        # 2. Brushstroke attack
        t_brush, m_brush, meta_brush = synthesizer.generate_tampered_pair(prot_img, attack_type="brushstroke")
        self.assertGreater(meta_brush["coverage_pct"], 0.0)

        # 3. Splice attack
        t_splice, m_splice, meta_splice = synthesizer.generate_tampered_pair(prot_img, attack_type="splice")
        self.assertGreater(meta_splice["coverage_pct"], 0.0)

        # 4. Negative clean sample
        t_neg, m_neg, meta_neg = synthesizer.generate_tampered_pair(prot_img, attack_type="negative")
        self.assertEqual(meta_neg["coverage_pct"], 0.0)
        self.assertEqual(float(np.sum(m_neg)), 0.0, "Negative sample must have an all-zero mask")

    def test_lattice_destruction_inside_tamper_mask(self):
        """Submodule 2.2: Mathematically prove that tampering destroys carrier lattice power inside the mask."""
        from docsafe.data.attacks import TamperSynthesizer
        from docsafe.core import inject_carrier_lattice, generate_carrier_lattice

        synthesizer = TamperSynthesizer(seed=123)
        base_img = np.full((150, 150, 3), 220, dtype=np.uint8)
        key = "ProofKey99"
        prot_img, _ = inject_carrier_lattice(base_img, key, period=8)

        # Apply brushstroke attack (pure inpainting without glyph)
        tampered_img, mask, _ = synthesizer.generate_tampered_pair(prot_img, attack_type="brushstroke")

        # Untouched area: carrier power should be ~1.0 (since delta=1)
        untouched_power = float(np.mean((prot_img[mask == 0].astype(float) - 220.0) ** 2))
        # Tampered area: carrier has been blurred/reset, power drops towards 0
        tampered_power = float(np.mean((tampered_img[mask == 1].astype(float) - 220.0) ** 2))

        print(f"\n[Lattice Power Proof] Untouched: {untouched_power:.4f} vs. Tampered: {tampered_power:.4f}")
        self.assertGreater(untouched_power, 0.8, "Untouched area must preserve carrier power!")
        self.assertLess(tampered_power, 0.2, "Tampered area must lose carrier lattice power!")

    def test_channel_distortions_jpeg_and_noise(self):
        """Submodule 2.3: Test JPEG compression simulation and sensor noise."""
        from docsafe.data.distortions import ChannelDistortions

        distorter = ChannelDistortions(seed=42)
        sample_img = np.full((120, 120, 3), 0.75, dtype=np.float32)

        # 1. JPEG Compression test
        jpeg_img = distorter.apply_jpeg_compression(sample_img, quality=75)
        self.assertEqual(jpeg_img.shape, (120, 120, 3))
        self.assertEqual(jpeg_img.dtype, np.float32)
        self.assertTrue(0.0 <= jpeg_img.min() and jpeg_img.max() <= 1.0)

        # 2. Sensor noise test
        noisy_img = distorter.apply_sensor_noise(sample_img)
        self.assertEqual(noisy_img.shape, (120, 120, 3))
        self.assertFalse(np.array_equal(sample_img, noisy_img))

        # 3. Pipeline integration test
        pipelined = distorter.apply_channel_pipeline(sample_img)
        self.assertEqual(pipelined.shape, (120, 120, 3))
        self.assertEqual(pipelined.dtype, np.float32)

    def test_carrier_survival_under_whatsapp_jpeg(self):
        """Submodule 2.3: Prove that mid-frequency lattice survives WhatsApp JPEG (Q=75)."""
        from docsafe.core import inject_carrier_lattice, generate_carrier_lattice
        from docsafe.data.distortions import ChannelDistortions

        base_img = np.full((160, 160, 3), 200, dtype=np.uint8)
        key = "WhatsAppTestKey"
        prot_img, _ = inject_carrier_lattice(base_img, key, period=8, delta=1.5)

        # Simulate heavy WhatsApp compression (Q=75)
        distorter = ChannelDistortions()
        compressed_img = distorter.apply_jpeg_compression(prot_img, quality=75)

        # Extract residual R = Compressed - Blur
        gray = cv2.cvtColor(compressed_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        residual = gray - cv2.GaussianBlur(gray, (3, 3), 0.8)
        carrier = generate_carrier_lattice(160, 160, key, period=8, channels=1).squeeze(-1)

        # Measure carrier correlation after JPEG compression
        flat_r = residual.flatten()
        flat_c = carrier.flatten()
        correlation = np.dot(flat_r, flat_c) / (np.linalg.norm(flat_r) * np.linalg.norm(flat_c) + 1e-8)

        print(f"\n[WhatsApp Robustness Proof] Lattice Correlation after Q=75 JPEG: {correlation:.4f}")
        self.assertGreater(correlation, 0.40, "Carrier must retain positive correlation even after Q=75 JPEG compression!")

    def test_pytorch_dataset_and_dataloader(self):
        """Submodule 2.4: Test PyTorch TamperDetectionDataset and DataLoader batch generation."""
        import torch
        from docsafe.data import TamperDetectionDataset, create_dataloader

        # Initialize dataset with synthetic samples
        ds = TamperDetectionDataset(
            secret_key="KeyLoaderTest",
            target_size=(128, 128),
            period=8,
            enable_distortions=True,
            virtual_length=8,
        )

        self.assertEqual(len(ds), 8)

        # 1. Test single sample output
        sample = ds[0]
        self.assertIn("image", sample)
        self.assertIn("carrier", sample)
        self.assertIn("mask", sample)

        img_t = sample["image"]
        car_t = sample["carrier"]
        msk_t = sample["mask"]

        self.assertIsInstance(img_t, torch.Tensor)
        self.assertEqual(img_t.shape, (3, 128, 128))
        self.assertEqual(car_t.shape, (1, 128, 128))
        self.assertEqual(msk_t.shape, (1, 128, 128))

        # Check bounds
        self.assertTrue(0.0 <= img_t.min() and img_t.max() <= 1.0)
        self.assertTrue(set(car_t.unique().tolist()).issubset({-1.0, 1.0}))
        self.assertTrue(set(msk_t.unique().tolist()).issubset({0.0, 1.0}))

        # 2. Test DataLoader batch compilation
        loader = create_dataloader(ds, batch_size=4, shuffle=False)
        batch = next(iter(loader))

        self.assertEqual(batch["image"].shape, (4, 3, 128, 128))
        self.assertEqual(batch["carrier"].shape, (4, 1, 128, 128))
        self.assertEqual(batch["mask"].shape, (4, 1, 128, 128))
        print("\n[DataLoader Batch Test] Batch shapes verified: Image (4, 3, 128, 128), Mask (4, 1, 128, 128)")


if __name__ == "__main__":
    unittest.main()

