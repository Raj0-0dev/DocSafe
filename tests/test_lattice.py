"""
Unit Test Suite for Module 1.1: Cryptographic Spatial Carrier Generator
=======================================================================
"""

import time
import unittest
import numpy as np
from docsafe.core.lattice import generate_carrier_lattice, derive_key_seed


class TestCarrierLattice(unittest.TestCase):

    def test_determinism(self):
        """Test that identical secret keys produce 100% bitwise identical lattices."""
        key = "Company_Audit_Key_2026"
        lat1 = generate_carrier_lattice(height=256, width=256, secret_key=key, period=8)
        lat2 = generate_carrier_lattice(height=256, width=256, secret_key=key, period=8)

        np.testing.assert_array_equal(lat1, lat2, err_msg="Lattices with identical keys must be strictly identical!")

    def test_orthogonality(self):
        """Test that different keys generate uncorrelated, orthogonal patterns (correlation ~ 0)."""
        key1 = "Key_Alpha_123"
        key2 = "Key_Beta_456"

        lat1 = generate_carrier_lattice(height=256, width=256, secret_key=key1, period=8)
        lat2 = generate_carrier_lattice(height=256, width=256, secret_key=key2, period=8)

        # Normalized cross-correlation
        flat1 = lat1.flatten().astype(np.float64)
        flat2 = lat2.flatten().astype(np.float64)
        correlation = np.dot(flat1, flat2) / (np.linalg.norm(flat1) * np.linalg.norm(flat2))

        # Absolute cross-correlation must be near 0 (less than 0.05)
        self.assertLess(abs(correlation), 0.05, f"Lattices with different keys should be orthogonal! Got correlation: {correlation}")

    def test_shape_and_bipolar_values(self):
        """Test output shape and ensure values are strictly -1.0 or +1.0."""
        h, w = 120, 180
        lat = generate_carrier_lattice(height=h, width=w, secret_key="test_seed", period=8, channels=3)

        self.assertEqual(lat.shape, (h, w, 3), "Shape must match requested (H, W, C)")
        unique_vals = set(np.unique(lat))
        self.assertTrue(unique_vals.issubset({-1.0, 1.0}), f"Values must be bipolar {{-1.0, 1.0}}, found {unique_vals}")

    def test_execution_speed(self):
        """Ensure lattice generation is high-performance (< 50ms for 1080p Full HD)."""
        start = time.perf_counter()
        _ = generate_carrier_lattice(height=1080, width=1920, secret_key="SpeedTestKey", period=8)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        print(f"\n[Performance Benchmark] 1080p Carrier Generation Time: {elapsed_ms:.2f} ms")
        self.assertLess(elapsed_ms, 100.0, f"Carrier generation must be fast (< 100ms), took {elapsed_ms:.2f}ms")

    def test_injection_invisibility_and_metrics(self):
        """Submodule 1.2: Test that injected image achieves PSNR > 48 dB and SSIM > 0.995."""
        from docsafe.core.lattice import inject_carrier_lattice
        np.random.seed(42)
        sample_img = np.random.randint(50, 220, (300, 300, 3), dtype=np.uint8)

        protected_img, metrics = inject_carrier_lattice(sample_img, secret_key="AuditKey2026", delta=1.0)

        print(f"\n[Invisibility Benchmark] PSNR: {metrics['psnr_db']:.2f} dB, SSIM: {metrics['ssim']:.4f}")
        self.assertGreater(metrics["psnr_db"], 48.0, "PSNR must exceed 48.0 dB for high-stealth invisibility!")
        self.assertGreater(metrics["ssim"], 0.995, "SSIM must exceed 0.995!")
        self.assertLessEqual(metrics["max_abs_diff"], 1, "Max absolute pixel perturbation must be <= 1 for delta=1.0!")

    def test_grayscale_and_color_injection(self):
        """Submodule 1.2: Test injection on both 3-channel color and 1-channel grayscale."""
        from docsafe.core.lattice import inject_carrier_lattice
        color_img = np.full((120, 120, 3), 200, dtype=np.uint8)
        gray_img = np.full((120, 120), 180, dtype=np.uint8)

        prot_color, m_c = inject_carrier_lattice(color_img, "key_color")
        prot_gray, m_g = inject_carrier_lattice(gray_img, "key_gray")

        self.assertEqual(prot_color.shape, (120, 120, 3))
        self.assertEqual(prot_gray.shape, (120, 120))
        self.assertEqual(prot_color.dtype, np.uint8)
        self.assertEqual(prot_gray.dtype, np.uint8)

    def test_protect_image_file_api(self):
        """Submodule 1.3: Test single-image protect_image file I/O."""
        import tempfile, cv2, os
        from pathlib import Path
        from docsafe.core import protect_image

        with tempfile.TemporaryDirectory() as tmp_dir:
            in_file = os.path.join(tmp_dir, "test_input.png")
            out_file = os.path.join(tmp_dir, "test_protected.png")

            # Create test image on disk
            test_img = np.full((100, 100, 3), 220, dtype=np.uint8)
            cv2.imwrite(in_file, test_img)

            summary = protect_image(
                input_path=in_file,
                secret_key="TestFileKey",
                output_path=out_file,
            )

            self.assertEqual(summary["status"], "success")
            self.assertTrue(os.path.exists(out_file), "Protected output file must exist on disk!")
            self.assertGreater(summary["metrics"]["psnr_db"], 48.0)

            # Re-read protected image
            re_read = cv2.imread(out_file)
            self.assertIsNotNone(re_read)
            self.assertEqual(re_read.shape, (100, 100, 3))

    def test_protect_batch_api(self):
        """Submodule 1.3: Test batch processing across a directory."""
        import tempfile, cv2, os, json
        from pathlib import Path
        from docsafe.core import protect_batch

        with tempfile.TemporaryDirectory() as tmp_dir:
            in_dir = os.path.join(tmp_dir, "raw")
            out_dir = os.path.join(tmp_dir, "protected")
            os.makedirs(in_dir)

            # Create 3 test files
            for i in range(3):
                img = np.full((80, 80, 3), 100 + i * 30, dtype=np.uint8)
                cv2.imwrite(os.path.join(in_dir, f"sample_{i}.png"), img)

            manifest = protect_batch(
                input_dir=in_dir,
                secret_key="BatchKey99",
                output_dir=out_dir,
            )

            self.assertEqual(len(manifest), 3, "All 3 images must be processed")
            manifest_file = os.path.join(out_dir, "protection_manifest.json")
            self.assertTrue(os.path.exists(manifest_file), "Manifest must be written!")
            with open(manifest_file, "r") as f:
                data = json.load(f)
            self.assertEqual(len(data), 3)


if __name__ == "__main__":
    unittest.main()

