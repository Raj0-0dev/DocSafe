"""
Unit Tests for DocSafe Inference Engine (Module 6: Deployment & Verification)
================================================================================
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from docsafe.inference.protect import ProtectionEngine, run_cli as run_protect_cli
from docsafe.inference.verify import VerificationEngine, run_cli as run_verify_cli


class TestProtectionEngine(unittest.TestCase):

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.sample_img_path = self.temp_dir / "sample_doc.png"

        # Generate a test receipt/document image (200x200 uint8)
        img = np.full((200, 200, 3), 245, dtype=np.uint8)
        cv2.putText(img, "INVOICE #1024", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
        cv2.putText(img, "TOTAL: $500.00", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
        cv2.imwrite(str(self.sample_img_path), img)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_protect_file(self):
        """Submodule 6.1: Test single file protection through ProtectionEngine."""
        engine = ProtectionEngine(secret_key="TestSecret2026", delta=1.0)
        out_path = self.temp_dir / "sample_protected.png"

        res = engine.protect_file(self.sample_img_path, out_path)

        self.assertTrue(out_path.exists())
        self.assertGreater(res["psnr_db"], 45.0)
        self.assertGreater(res["ssim"], 0.99)
        self.assertEqual(res["resolution"], "200x200")

    def test_protect_directory(self):
        """Submodule 6.1: Test batch directory protection and manifest generation."""
        batch_in = self.temp_dir / "batch_in"
        batch_out = self.temp_dir / "batch_out"
        batch_in.mkdir(parents=True, exist_ok=True)

        for i in range(3):
            sub_p = batch_in / f"doc_{i}.png"
            shutil.copy(self.sample_img_path, sub_p)

        engine = ProtectionEngine(secret_key="BatchSecret2026")
        results = engine.protect_directory(batch_in, batch_out)

        self.assertEqual(len(results), 3)
        manifest_p = batch_out / "protection_manifest.json"
        self.assertTrue(manifest_p.exists())

        with open(manifest_p, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        self.assertEqual(len(manifest_data), 3)
        self.assertIn("metrics", manifest_data[0])

    def test_protect_cli_execution(self):
        """Submodule 6.1: Test CLI entry point parsing and exit codes."""
        out_file = self.temp_dir / "cli_protected.png"
        exit_code = run_protect_cli([
            "-i", str(self.sample_img_path),
            "-o", str(out_file),
            "-k", "CliKey2026",
            "-d", "1.0",
        ])
        self.assertEqual(exit_code, 0)
        self.assertTrue(out_file.exists())

        # Test non-existent file handling
        bad_code = run_protect_cli(["-i", "non_existent_file.png", "-o", "out.png"])
        self.assertEqual(bad_code, 1)


class TestVerificationEngine(unittest.TestCase):

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.protected_img_path = self.temp_dir / "protected_doc.png"

        # Generate a protected document
        img = np.full((128, 128, 3), 240, dtype=np.uint8)
        cv2.putText(img, "BAL: $100", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        prot_engine = ProtectionEngine(secret_key="VerifyKey2026", delta=1.0)
        prot_engine.protect_file(img, self.protected_img_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_verify_image_output_structure(self):
        """Submodule 6.2: Test zero-reference verification output contract."""
        verifier = VerificationEngine(
            secret_key="VerifyKey2026",
            device="cpu",
            threshold=0.5,
        )

        result = verifier.verify_image(self.protected_img_path)

        self.assertIn("status", result)
        self.assertIn("is_tampered", result)
        self.assertIn("bboxes", result)
        self.assertIn("prob_heatmap", result)
        self.assertIn("annotated_bgr", result)
        self.assertIn("execution_time_ms", result)

        self.assertEqual(result["prob_heatmap"].shape, (128, 128))
        self.assertTrue(np.all(result["prob_heatmap"] >= 0.0))
        self.assertTrue(np.all(result["prob_heatmap"] <= 1.0))
        self.assertGreater(result["execution_time_ms"], 0.0)

    def test_generate_diagnostic_board(self):
        """Submodule 6.2: Test 4-panel visual verification diagnostic board generation."""
        verifier = VerificationEngine(
            secret_key="VerifyKey2026",
            device="cpu",
        )
        result = verifier.verify_image(self.protected_img_path)
        board = verifier.generate_diagnostic_board(result)

        # Original is 128x128, board is 2x2 grid => 256x256
        self.assertEqual(board.shape, (256, 256, 3))
        self.assertEqual(board.dtype, np.uint8)

    def test_verify_cli_execution(self):
        """Submodule 6.2: Test verify CLI execution, visual board and JSON report output."""
        board_file = self.temp_dir / "cli_board.png"
        report_file = self.temp_dir / "cli_report.json"

        exit_code = run_verify_cli([
            "-i", str(self.protected_img_path),
            "-o", str(board_file),
            "-k", "VerifyKey2026",
            "--report", str(report_file),
            "--device", "cpu",
        ])

        self.assertEqual(exit_code, 0)
        self.assertTrue(board_file.exists())
        self.assertTrue(report_file.exists())

        with open(report_file, "r", encoding="utf-8") as f:
            report_data = json.load(f)

        self.assertIn("status", report_data)
        self.assertIn("overall_confidence", report_data)
        self.assertIn("bboxes", report_data)


if __name__ == "__main__":
    unittest.main()
