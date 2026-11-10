"""
DocSafe: Protection Engine & CLI
===================================
Submodule 6.1: Standalone CLI and API for invisibly protecting digital documents
and images by injecting the deterministic cryptographic carrier lattice.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import io
import cv2
import numpy as np

from docsafe.core.lattice import (
    derive_key_seed,
    protect_image,
    protect_batch,
    inject_carrier_lattice,
)
from docsafe.core.pdf_engine import (
    PDFEngine,
    DEFAULT_PDF_DPI,
)


class ProtectionEngine:
    """
    Submodule 6.1: Protection service for single images and batch directories.
    Injects invisible, high-frequency carrier lattices with zero perceptual distortion.
    """

    def __init__(
        self,
        secret_key: str = "DocSafeKey2026",
        period: int = 8,
        delta: float = 1.0,
    ):
        self.secret_key = secret_key
        self.period = int(period)
        self.delta = float(delta)
        self.key_seed = derive_key_seed(secret_key)

    def protect_file(
        self,
        input_path: Union[str, Path, "np.ndarray"],
        output_path: Union[str, Path],
    ) -> dict:
        """
        Protect a single document or photo from a file path or in-memory array.
        """
        import cv2
        import numpy as np
        from docsafe.core.lattice import inject_carrier_lattice

        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(input_path, np.ndarray):
            prot_img, metrics = inject_carrier_lattice(
                image=input_path,
                secret_key=self.secret_key,
                period=self.period,
                delta=self.delta,
            )
            cv2.imwrite(str(out_p), prot_img)
            summary = {
                "status": "success",
                "input_file": "<ndarray>",
                "output_file": out_p.name,
                "resolution": f"{input_path.shape[1]}x{input_path.shape[0]}",
                "period": self.period,
                "delta": self.delta,
                "metrics": metrics,
            }
        else:
            summary = protect_image(
                input_path=str(input_path),
                secret_key=self.secret_key,
                output_path=str(output_path),
                period=self.period,
                delta=self.delta,
            )
        # Expose top-level convenience metrics
        metrics = summary.get("metrics", {})
        summary["psnr_db"] = metrics.get("psnr_db", 0.0)
        summary["ssim"] = metrics.get("ssim", 0.0)
        summary["max_delta"] = metrics.get("max_pixel_delta", 0.0)
        return summary

    def protect_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        max_images: Optional[int] = None,
    ) -> List[dict]:
        """
        Recursively discover and protect all images in a folder.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return protect_batch(
            input_dir=str(input_dir),
            secret_key=self.secret_key,
            output_dir=str(output_dir),
            period=self.period,
            delta=self.delta,
            max_images=max_images,
        )

    def protect_pdf(
        self,
        input_pdf: Union[str, Path, bytes, io.BytesIO],
        output_pdf: Optional[Union[str, Path]] = None,
        dpi: int = DEFAULT_PDF_DPI,
    ) -> Dict[str, Any]:
        """
        Invisibly protects a multi-page PDF document by injecting the cryptographic
        carrier lattice into each page while compiling a Searchable Dual-Layer PDF.
        Preserves 100% text selection (Ctrl+A), searchability (Ctrl+F), and copying.

        Returns:
        --------
        summary : dict
            Contains status, page count, per-page metrics, and compiled PDF bytes.
        """
        # 1. Render all pages to high-res BGR arrays (sequential memory-safe)
        pages = PDFEngine.render_pdf_to_pages(input_pdf, dpi=dpi)
        text_metadata = PDFEngine.extract_text_layer(input_pdf)

        protected_images = []
        dimensions_pt = []
        page_summaries = []

        for p in pages:
            prot_img, metrics = inject_carrier_lattice(
                image=p["image_bgr"],
                secret_key=self.secret_key,
                period=self.period,
                delta=self.delta,
                chroma_mode=True,
            )
            protected_images.append(prot_img)
            dimensions_pt.append((p["width_pt"], p["height_pt"]))
            page_summaries.append({
                "page_index": p["page_index"],
                "resolution": f"{p['width_px']}x{p['height_px']}",
                "psnr_db": float(metrics.get("psnr_db", 0.0)),
                "ssim": float(metrics.get("ssim", 0.0)),
                "max_delta": float(metrics.get("max_abs_diff", metrics.get("max_pixel_delta", 1.0))),
            })

        # 2. Compile Dual-Layer Searchable PDF
        pdf_bytes = PDFEngine.compile_dual_layer_pdf(
            protected_pages_bgr=protected_images,
            text_metadata_per_page=text_metadata,
            page_dimensions_pt=dimensions_pt,
            output_path=output_pdf,
            dpi=dpi,
        )

        avg_psnr = float(np.mean([m["psnr_db"] for m in page_summaries])) if page_summaries else 0.0
        avg_ssim = float(np.mean([m["ssim"] for m in page_summaries])) if page_summaries else 0.0

        return {
            "status": "success",
            "file_type": "pdf",
            "total_pages": len(pages),
            "dpi": dpi,
            "avg_psnr_db": avg_psnr,
            "avg_ssim": avg_ssim,
            "pages": page_summaries,
            "pdf_bytes": pdf_bytes,
            "output_pdf_path": str(output_pdf) if output_pdf else None,
        }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="protect",
        description="DocSafe Protection CLI: Invisible Carrier Lattice Injection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Path to input image file or directory of images to protect",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Path to output file (if input is a file) or output directory (if input is a directory)",
    )
    parser.add_argument(
        "-k", "--secret-key",
        type=str,
        default="DocSafeKey2026",
        help="Secret cryptographic key to seed the carrier lattice",
    )
    parser.add_argument(
        "-d", "--delta",
        type=float,
        default=1.0,
        help="Modulation amplitude (delta +/- 1.0 achieves PSNR > 48 dB)",
    )
    parser.add_argument(
        "-p", "--period",
        type=int,
        default=8,
        help="Spatial harmonic period in pixels",
    )
    return parser.parse_args(argv)


def run_cli(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    in_path = Path(args.input)

    if not in_path.exists():
        print(f"Error: Input path does not exist: {args.input}", file=sys.stderr)
        return 1

    engine = ProtectionEngine(
        secret_key=args.secret_key,
        period=args.period,
        delta=args.delta,
    )

    print("=" * 70)
    print("  DEEP-GUARD: INVISIBLE CARRIER LATTICE PROTECTION ENGINE")
    print("=" * 70)
    print(f"  Secret Key Hash   : {engine.key_seed:#010x}")
    print(f"  Modulation Delta  : +/-{args.delta} pixel intensity")
    print(f"  Lattice Period    : {args.period} px")
    print("-" * 70)

    if in_path.is_file():
        # Single image protection
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  Protecting single file: {in_path.name}")
        t0 = time.time()
        result = engine.protect_file(in_path, out_path)
        dur = (time.time() - t0) * 1000.0

        print(f"  Saved Protected   : {out_path}")
        print(f"  Resolution        : {result.get('resolution')}")
        print(f"  PSNR Invisibility : {result.get('psnr_db', 0.0):.2f} dB (Standard: > 45 dB)")
        print(f"  SSIM Similarity   : {result.get('ssim', 0.0):.4f} (Standard: > 0.99)")
        print(f"  Processing Time   : {dur:.2f} ms")
        print("=" * 70)
        return 0

    elif in_path.is_dir():
        # Batch directory protection
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Protecting batch directory: {in_path}")
        t0 = time.time()
        results = engine.protect_directory(
            input_dir=in_path,
            output_dir=out_dir,
        )
        total_time = time.time() - t0

        if not results:
            print("  Warning: No valid images found in input directory.")
            return 0

        avg_psnr = sum(r.get("metrics", {}).get("psnr_db", 0.0) for r in results) / len(results)
        avg_ssim = sum(r.get("metrics", {}).get("ssim", 0.0) for r in results) / len(results)

        print("-" * 70)
        print(f"  Successfully protected {len(results)} images in {total_time:.2f} s ({len(results)/max(1e-3, total_time):.1f} img/s).")
        print(f"  Mean Batch PSNR   : {avg_psnr:.2f} dB")
        print(f"  Mean Batch SSIM   : {avg_ssim:.4f}")
        print(f"  Audit Manifest    : {out_dir / 'protection_manifest.json'}")
        print("=" * 70)
        return 0

    else:
        print(f"Error: Unknown path type: {args.input}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(run_cli())
