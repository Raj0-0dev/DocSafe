"""
DocSafe: Searchable Dual-Layer (Sandwich) PDF Engine
======================================================
Submodule 1.4: High-Performance Multi-Page PDF Rasterization,
Invisible Text Layer Extraction, and Searchable Dual-Layer Assembly.

Architecture:
-------------
- Layer 1 (Visible / Top): High-resolution rendered page (200 DPI) containing
  the invisible cryptographic carrier lattice (+/-1 pixel delta).
- Layer 2 (Invisible / Underneath): Searchable vector text stream positioned
  at matching coordinate bounds, preserving text selection, search (Ctrl+F),
  and copy-paste across standard PDF viewers.
"""

import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import fitz  # PyMuPDF
import numpy as np


# Standard forensic rendering DPI for coordinate and lattice stability
DEFAULT_PDF_DPI = 200
POINTS_PER_INCH = 72.0


class PDFEngine:
    """
    Core engine for bidirectional conversion between PDF vector streams
    and forensic pixel arrays with dual-layer text searchability.
    """

    @staticmethod
    def render_pdf_to_pages(
        pdf_source: Union[str, Path, bytes, io.BytesIO],
        dpi: int = DEFAULT_PDF_DPI,
    ) -> List[Dict[str, Any]]:
        """
        Renders each page of a PDF document into an OpenCV BGR image array
        along with original point dimensions and page metadata.

        Returns:
        --------
        pages : list of dict
            [
                {
                    "page_index": int (0-based),
                    "image_bgr": np.ndarray (H, W, 3) uint8,
                    "width_pt": float,
                    "height_pt": float,
                    "width_px": int,
                    "height_px": int,
                    "dpi": int,
                }, ...
            ]
        """
        if isinstance(pdf_source, (bytes, io.BytesIO)):
            raw_bytes = pdf_source.getvalue() if isinstance(pdf_source, io.BytesIO) else pdf_source
            doc = fitz.open(stream=raw_bytes, filetype="pdf")
        else:
            p = Path(pdf_source)
            if not p.exists():
                raise FileNotFoundError(f"PDF source not found: {pdf_source}")
            doc = fitz.open(str(p))

        rendered_pages = []

        try:
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                rect = page.rect
                width_pt, height_pt = rect.width, rect.height

                # Render page to high-res pixmap at requested DPI
                pix = page.get_pixmap(dpi=dpi, alpha=False)

                # Convert raw pixmap samples to OpenCV BGR array
                img_rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

                rendered_pages.append({
                    "page_index": page_idx,
                    "image_bgr": img_bgr,
                    "width_pt": float(width_pt),
                    "height_pt": float(height_pt),
                    "width_px": int(pix.width),
                    "height_px": int(pix.height),
                    "dpi": int(dpi),
                })
        finally:
            doc.close()

        return rendered_pages

    @staticmethod
    def extract_text_layer(
        pdf_source: Union[str, Path, bytes, io.BytesIO],
    ) -> List[List[Dict[str, Any]]]:
        """
        Extracts individual word tokens, line structures, and point bounding boxes
        from each page of the PDF to build the underlying searchable text layer.

        Returns:
        --------
        pages_words : list of lists
            Per-page word dictionaries containing (x0, y0, x1, y1, text, block_no, line_no, word_no)
        """
        if isinstance(pdf_source, (bytes, io.BytesIO)):
            raw_bytes = pdf_source.getvalue() if isinstance(pdf_source, io.BytesIO) else pdf_source
            doc = fitz.open(stream=raw_bytes, filetype="pdf")
        else:
            p = Path(pdf_source)
            if not p.exists():
                raise FileNotFoundError(f"PDF source not found: {pdf_source}")
            doc = fitz.open(str(p))

        pages_data = []

        try:
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                raw_words = page.get_text("words")  # (x0, y0, x1, y1, word, block_no, line_no, word_no)

                word_records = []
                for w in raw_words:
                    word_records.append({
                        "x0": float(w[0]),
                        "y0": float(w[1]),
                        "x1": float(w[2]),
                        "y1": float(w[3]),
                        "text": str(w[4]),
                        "block_no": int(w[5]),
                        "line_no": int(w[6]),
                        "word_no": int(w[7]),
                    })
                pages_data.append(word_records)
        finally:
            doc.close()

        return pages_data

    @staticmethod
    def compile_dual_layer_pdf(
        protected_pages_bgr: List[np.ndarray],
        text_metadata_per_page: Optional[List[List[Dict[str, Any]]]] = None,
        page_dimensions_pt: Optional[List[Tuple[float, float]]] = None,
        output_path: Optional[Union[str, Path]] = None,
        dpi: int = DEFAULT_PDF_DPI,
    ) -> bytes:
        """
        Compiles protected page image arrays and original vector text coordinates
        into a Searchable Dual-Layer (Sandwich) PDF.

        Parameters:
        -----------
        protected_pages_bgr : list of np.ndarray
            Protected page images (H, W, 3) uint8 containing the carrier lattice.
        text_metadata_per_page : list of list of dict, optional
            Extracted word coordinates per page. If None or empty, generates image-backed PDF.
        page_dimensions_pt : list of (width_pt, height_pt), optional
            Original page dimensions in points (1/72 in). If None, calculated from image size / DPI.
        output_path : str or Path, optional
            File destination. If None, returns PDF bytes in memory.
        dpi : int
            DPI used during rasterization (default 200).

        Returns:
        --------
        pdf_bytes : bytes
            Binary stream of the compiled searchable PDF.
        """
        out_doc = fitz.open()

        for page_idx, page_bgr in enumerate(protected_pages_bgr):
            img_h, img_w = page_bgr.shape[:2]

            # Calculate page dimensions in PDF points (72 pt = 1 inch)
            if page_dimensions_pt and page_idx < len(page_dimensions_pt):
                pt_w, pt_h = page_dimensions_pt[page_idx]
            else:
                pt_w = (img_w / float(dpi)) * POINTS_PER_INCH
                pt_h = (img_h / float(dpi)) * POINTS_PER_INCH

            new_page = out_doc.new_page(width=pt_w, height=pt_h)
            page_rect = fitz.Rect(0, 0, pt_w, pt_h)

            # Encode protected BGR image to lossless PNG byte stream
            success, img_encoded = cv2.imencode(".png", page_bgr)
            if not success:
                raise ValueError(f"Failed to encode protected page {page_idx} to PNG")
            img_bytes = img_encoded.tobytes()

            # Layer 1 (Visible): Insert high-resolution protected image spanning entire page
            new_page.insert_image(page_rect, stream=img_bytes)

            # Layer 2 (Searchable / Underneath): Insert invisible text layer at exact coordinates
            if text_metadata_per_page and page_idx < len(text_metadata_per_page):
                words = text_metadata_per_page[page_idx]
                for w in words:
                    wx0, wy0, wx1, wy1 = w["x0"], w["y0"], w["x1"], w["y1"]
                    text_str = w["text"]

                    # Calculate approximate font size matching word bounding box height
                    box_h = wy1 - wy0
                    fontsize = max(4.0, box_h * 0.85)

                    # Text insertion point (baseline position)
                    insert_pt = fitz.Point(wx0, wy1 - (box_h * 0.15))

                    # render_mode=3 indicates invisible text (standard searchable PDF trick)
                    try:
                        new_page.insert_text(
                            insert_pt,
                            text_str + " ",
                            fontsize=fontsize,
                            render_mode=3,
                        )
                    except Exception:
                        pass

        pdf_bytes = out_doc.tobytes(deflate=True)
        out_doc.close()

        if output_path is not None:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "wb") as f:
                f.write(pdf_bytes)

        return pdf_bytes
