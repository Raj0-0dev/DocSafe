"""
Unit Tests for DocSafe PDF Engine (Dual-Layer Searchable PDF Module)
=======================================================================
"""

import tempfile
import unittest
from pathlib import Path
import fitz
import numpy as np

from docsafe.core.lattice import inject_carrier_lattice
from docsafe.core.pdf_engine import PDFEngine, DEFAULT_PDF_DPI
from docsafe.inference.protect import ProtectionEngine
from docsafe.inference.verify import VerificationEngine


class TestPDFEngine(unittest.TestCase):

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.sample_pdf_path = self.temp_dir / "sample_invoice.pdf"

        # Create a sample multi-page digital invoice PDF using PyMuPDF
        doc = fitz.open()

        # Page 1: Header and Invoice Items
        p1 = doc.new_page(width=595, height=842)  # A4
        p1.insert_text((50, 60), "TAX INVOICE #1024", fontsize=18, color=(0.1, 0.1, 0.1))
        p1.insert_text((50, 100), "Date: 2026-11-01", fontsize=11, color=(0.3, 0.3, 0.3))
        p1.insert_text((50, 130), "Item: Enterprise Cloud License", fontsize=11, color=(0.2, 0.2, 0.2))
        p1.insert_text((50, 160), "Total Amount: $1,815.00", fontsize=13, color=(0.1, 0.1, 0.1))

        # Page 2: Terms and Conditions
        p2 = doc.new_page(width=595, height=842)
        p2.insert_text((50, 60), "TERMS & CONDITIONS", fontsize=16, color=(0.1, 0.1, 0.1))
        p2.insert_text((50, 100), "All accounts payable within 30 days.", fontsize=11, color=(0.3, 0.3, 0.3))
        p2.insert_text((50, 130), "Authorized Signature: Verified", fontsize=12, color=(0.2, 0.2, 0.2))

        doc.save(str(self.sample_pdf_path))
        doc.close()

    def test_render_pdf_to_pages(self):
        """Test multi-page rendering to OpenCV BGR arrays at 200 DPI."""
        pages = PDFEngine.render_pdf_to_pages(self.sample_pdf_path, dpi=DEFAULT_PDF_DPI)

        self.assertEqual(len(pages), 2)
        for i, page_info in enumerate(pages):
            self.assertEqual(page_info["page_index"], i)
            self.assertIn("image_bgr", page_info)
            self.assertEqual(page_info["dpi"], DEFAULT_PDF_DPI)
            img = page_info["image_bgr"]
            self.assertIsInstance(img, np.ndarray)
            self.assertEqual(img.dtype, np.uint8)
            self.assertGreater(img.shape[0], 1000)
            self.assertGreater(img.shape[1], 1000)

    def test_extract_text_layer(self):
        """Test extraction of searchable word bounding boxes and tokens."""
        pages_words = PDFEngine.extract_text_layer(self.sample_pdf_path)

        self.assertEqual(len(pages_words), 2)
        p1_words = [w["text"] for w in pages_words[0]]
        self.assertIn("INVOICE", p1_words)
        self.assertIn("$1,815.00", p1_words)

        p2_words = [w["text"] for w in pages_words[1]]
        self.assertIn("TERMS", p2_words)

    def test_compile_dual_layer_pdf(self):
        """Test compilation of searchable dual-layer PDF preserving searchability and carrier."""
        # 1. Render pages
        pages = PDFEngine.render_pdf_to_pages(self.sample_pdf_path, dpi=DEFAULT_PDF_DPI)
        text_metadata = PDFEngine.extract_text_layer(self.sample_pdf_path)

        # 2. Inject carrier lattice into each page image
        protected_bgr_list = []
        dimensions_pt = []

        for p in pages:
            prot_img, metrics = inject_carrier_lattice(
                p["image_bgr"],
                secret_key="PDFTestSecret2026",
                period=8,
                delta=1.0,
            )
            self.assertGreater(metrics["psnr_db"], 48.0)
            self.assertGreater(metrics["ssim"], 0.99)
            protected_bgr_list.append(prot_img)
            dimensions_pt.append((p["width_pt"], p["height_pt"]))

        # 3. Compile Dual-Layer Searchable PDF
        out_pdf_path = self.temp_dir / "protected_searchable.pdf"
        pdf_bytes = PDFEngine.compile_dual_layer_pdf(
            protected_pages_bgr=protected_bgr_list,
            text_metadata_per_page=text_metadata,
            page_dimensions_pt=dimensions_pt,
            output_path=out_pdf_path,
            dpi=DEFAULT_PDF_DPI,
        )

        self.assertTrue(out_pdf_path.exists())
        self.assertGreater(len(pdf_bytes), 1000)

        # 4. Verify Searchability (Ctrl+F emulation)
        check_doc = fitz.open(str(out_pdf_path))
        self.assertEqual(len(check_doc), 2)

        # Search for exact invoice string on page 0
        matches = check_doc[0].search_for("$1,815.00")
        self.assertGreater(len(matches), 0, "Text '$1,815.00' must be searchable on Page 1")

        # Search for terms on page 1
        matches_p2 = check_doc[1].search_for("TERMS")
        self.assertGreater(len(matches_p2), 0, "Text 'TERMS' must be searchable on Page 2")

        check_doc.close()

    def test_engine_protect_and_verify_pdf(self):
        """Test full end-to-end PDF protection and zero-reference verification."""
        prot_engine = ProtectionEngine(secret_key="PDFTestSecret2026", delta=1.0)
        protected_pdf_path = self.temp_dir / "engine_protected.pdf"

        # 1. Protect PDF
        res_prot = prot_engine.protect_pdf(
            input_pdf=self.sample_pdf_path,
            output_pdf=protected_pdf_path,
            dpi=DEFAULT_PDF_DPI,
        )

        self.assertEqual(res_prot["status"], "success")
        self.assertEqual(res_prot["total_pages"], 2)
        self.assertTrue(protected_pdf_path.exists())
        self.assertGreater(res_prot["avg_psnr_db"], 48.0)

        # 2. Verify protected PDF (Untouched) -> Must be AUTHENTIC
        verify_engine = VerificationEngine(
            secret_key="PDFTestSecret2026",
            device="cpu",
        )
        res_verify = verify_engine.verify_pdf(protected_pdf_path, dpi=DEFAULT_PDF_DPI)

        self.assertEqual(res_verify["status"], "AUTHENTIC")
        self.assertFalse(res_verify["is_tampered"])
        self.assertEqual(res_verify["total_pages"], 2)
        self.assertEqual(res_verify["tampered_pages_count"], 0)


if __name__ == "__main__":
    unittest.main()
