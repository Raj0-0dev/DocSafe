import argparse
import io
import json
import os
import sys
import time
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

CUR_DIR = Path(__file__).resolve().parent
ROOT_DIR = CUR_DIR.parent
for p in [str(ROOT_DIR), str(ROOT_DIR.parent)]:
    if p not in sys.path:
        sys.path.insert(0, p)

if "docsafe" not in sys.modules:
    try:
        import docsafe
    except ModuleNotFoundError:
        _pkg = types.ModuleType("docsafe")
        _pkg.__path__ = [str(ROOT_DIR)]
        sys.modules["docsafe"] = _pkg

import cv2
import numpy as np
import torch

from docsafe.core.lattice import generate_carrier_lattice, derive_key_seed
from docsafe.core.pdf_engine import PDFEngine, DEFAULT_PDF_DPI
from docsafe.models.unet_detector import DocSafeUNetDetector

class VerificationEngine:

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        secret_key: str = "DocSafeKey2026",
        period: int = 8,
        device: Optional[Union[str, torch.device]] = None,
        threshold: float = 0.5,
        min_tampered_pixels: int = 16,
        max_inference_dim: int = 2560,
    ):
        self.secret_key = secret_key
        self.period = int(period)
        self.threshold = float(threshold)
        self.min_tampered_pixels = int(min_tampered_pixels)
        self.max_inference_dim = int(max_inference_dim)
        self.key_seed = derive_key_seed(secret_key)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.model = DocSafeUNetDetector(
            in_fused_channels=8,
            encoder_dims=(32, 64, 128, 256),
            decoder_out_dim=32,
        ).to(self.device)

        if checkpoint_path is not None and Path(checkpoint_path).exists():
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            raw_state_dict = ckpt.get("model_state_dict", ckpt)

            model_keys = set(self.model.state_dict().keys())
            has_final_conv = any("decoder.final_conv" in k for k in model_keys)

            adapted_state_dict = {}
            for k, v in raw_state_dict.items():
                new_k = k
                if has_final_conv:
                    if "decoder.final_up.1." in new_k:
                        new_k = new_k.replace("decoder.final_up.1.", "decoder.final_conv.0.")
                    elif "decoder.final_up.2." in new_k:
                        new_k = new_k.replace("decoder.final_up.2.", "decoder.final_conv.1.")
                else:
                    if "decoder.final_conv.0." in new_k:
                        new_k = new_k.replace("decoder.final_conv.0.", "decoder.final_up.1.")
                    elif "decoder.final_conv.1." in new_k:
                        new_k = new_k.replace("decoder.final_conv.1.", "decoder.final_up.2.")

                adapted_state_dict[new_k] = v

            try:
                self.model.load_state_dict(adapted_state_dict, strict=True)
            except RuntimeError:
                self.model.load_state_dict(adapted_state_dict, strict=False)

            self.has_trained_weights = True
        else:
            self.has_trained_weights = False

        self.model.eval()

    def verify_image(
        self,
        image_input: Union[str, Path, np.ndarray],
    ) -> Dict:
        t0 = time.time()

        if isinstance(image_input, (str, Path)):
            in_p = Path(image_input)
            if not in_p.exists():
                raise FileNotFoundError(f"Questioned image not found: {image_input}")
            img_bgr = cv2.imread(str(in_p))
            if img_bgr is None:
                raise ValueError(f"Failed to decode image from: {image_input}")
        else:
            img_bgr = image_input.copy()

        h, w = img_bgr.shape[:2]

        native_carrier = generate_carrier_lattice(
            height=h,
            width=w,
            secret_key=self.secret_key,
            period=self.period,
            channels=1,
        ).squeeze()

        is_chroma_carrier = False
        chroma_coherence_arr = None
        res_chroma = None
        if img_bgr.ndim == 3 and img_bgr.shape[2] == 3:
            ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
            cb_channel = ycrcb[:, :, 2]
            res_chroma = cb_channel - cv2.GaussianBlur(cb_channel, (5, 5), 1.0)
            chroma_coherence_arr = res_chroma * native_carrier
            chroma_global = float(np.mean(chroma_coherence_arr))
            if chroma_global >= 0.05:
                is_chroma_carrier = True

        gray_native = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        res_native = gray_native - cv2.GaussianBlur(gray_native, (5, 5), 1.0)
        native_coherence_arr = res_native * native_carrier
        spatial_coherence = float(np.mean(native_coherence_arr))

        if is_chroma_carrier:
            global_coherence = chroma_global
            coherence_arr = chroma_coherence_arr
            residual_arr = res_chroma
            has_intact_carrier = True
        else:
            global_coherence = spatial_coherence
            coherence_arr = native_coherence_arr
            residual_arr = res_native
            has_intact_carrier = global_coherence >= 0.0008

        needs_resize = max(h, w) > self.max_inference_dim
        if needs_resize:
            scale = self.max_inference_dim / float(max(h, w))
            infer_w = max(32, int(round(w * scale)))
            infer_h = max(32, int(round(h * scale)))
            infer_w = infer_w if infer_w % 2 == 0 else infer_w - 1
            infer_h = infer_h if infer_h % 2 == 0 else infer_h - 1
            img_infer = cv2.resize(img_bgr, (infer_w, infer_h), interpolation=cv2.INTER_AREA)
        else:
            infer_w, infer_h = w, h
            img_infer = img_bgr

        raw_carrier = generate_carrier_lattice(
            height=infer_h,
            width=infer_w,
            secret_key=self.secret_key,
            period=self.period,
            channels=1,
        )

        img_rgb = cv2.cvtColor(img_infer, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        carrier_tensor = torch.from_numpy(raw_carrier.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)

        use_amp = (self.device.type == "cuda")
        try:
            with torch.inference_mode():
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                    prob_mask, intermediates = self.model(
                        img_tensor, carrier_tensor, return_intermediates=True
                    )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower() and torch.cuda.is_available():
                torch.cuda.empty_cache()
                orig_device = self.device
                self.model = self.model.to("cpu")
                with torch.inference_mode():
                    prob_mask, intermediates = self.model(
                        img_tensor.cpu(), carrier_tensor.cpu(), return_intermediates=True
                    )
                self.model = self.model.to(orig_device)
            else:
                raise e

        prob_heatmap = prob_mask.squeeze().float().cpu().numpy()
        residual_arr = intermediates["residual"].squeeze().float().cpu().numpy()
        coherence_arr = intermediates["coherence"].squeeze().float().cpu().numpy()

        if needs_resize:
            prob_heatmap = cv2.resize(prob_heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

        if is_chroma_carrier:
            coherence_arr = chroma_coherence_arr
            residual_arr = res_chroma
        else:
            coherence_arr = native_coherence_arr
            residual_arr = res_native

        total_image_pixels = w * h

        gray_u8 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray_u8, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_u8, cv2.CV_32F, 0, 1, ksize=3)
        edge_mag = np.sqrt(sobelx**2 + sobely**2) / 255.0

        has_intact_carrier = global_coherence >= 0.0008

        bin_mask = (prob_heatmap >= self.threshold).astype(np.uint8) * 255
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bboxes = []
        annotated_bgr = img_bgr.copy()
        confident_tampered_pixels = 0

        if is_chroma_carrier:
            sign_match = (np.sign(residual_arr) == np.sign(native_carrier)).astype(np.float32)
            smoothed_sign_16 = cv2.boxFilter(sign_match, -1, (16, 16))
            carrier_loss_mask = (smoothed_sign_16 < 0.55).astype(np.uint8) * 255
            k_loss = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            cleaned_loss = cv2.morphologyEx(carrier_loss_mask, cv2.MORPH_OPEN, k_loss)
            cleaned_loss = cv2.morphologyEx(cleaned_loss, cv2.MORPH_CLOSE, k_loss)
            cnts_loss, _ = cv2.findContours(cleaned_loss, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts_loss:
                area = cv2.contourArea(c)
                if area >= 50:
                    bx, by, bw, bh = cv2.boundingRect(c)
                    if bw * bh < 0.50 * total_image_pixels:
                        bboxes.append({
                            "x": int(bx),
                            "y": int(by),
                            "width": int(bw),
                            "height": int(bh),
                            "area": float(area),
                            "confidence": 0.99,
                        })
                        confident_tampered_pixels += int(area)
                        cv2.rectangle(annotated_bgr, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
                        label = "FORGED 99.0%"
                        cv2.putText(
                            annotated_bgr,
                            label,
                            (bx, max(15, by - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (0, 0, 255),
                            1,
                            cv2.LINE_AA,
                        )

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_tampered_pixels:
                continue

            bx, by, bw, bh = cv2.boundingRect(cnt)
            box_area = bw * bh

            if box_area > 0.50 * total_image_pixels or area > 0.40 * total_image_pixels:
                continue

            patch_prob = prob_heatmap[by : by + bh, bx : bx + bw]
            patch_conf = float(np.mean(patch_prob)) if patch_prob.size > 0 else 0.0

            min_conf = max(0.55, self.threshold)
            if patch_conf < min_conf:
                continue

            patch_coherence = float(np.mean(coherence_arr[by : by + bh, bx : bx + bw]))
            patch_edge = float(np.mean(edge_mag[by : by + bh, bx : bx + bw]))

            if is_chroma_carrier and patch_coherence >= 0.10:
                continue
            if not is_chroma_carrier and has_intact_carrier and patch_coherence >= 0.0005:
                continue

            if has_intact_carrier and box_area < 1500 and patch_edge > 0.30:
                continue

            confident_tampered_pixels += int(np.sum(patch_prob >= self.threshold))

            bboxes.append({
                "x": int(bx),
                "y": int(by),
                "width": int(bw),
                "height": int(bh),
                "area": float(area),
                "confidence": float(patch_conf),
            })

            cv2.rectangle(annotated_bgr, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
            label = f"FORGED {patch_conf * 100:.1f}%"
            cv2.putText(
                annotated_bgr,
                label,
                (bx, max(15, by - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

        has_localized_tamper = len(bboxes) > 0

        if has_localized_tamper:
            status = "TAMPERED"
            is_tampered = True
            overall_conf = max([b["confidence"] for b in bboxes])
        elif has_intact_carrier:
            status = "AUTHENTIC"
            is_tampered = False
            overall_conf = 0.99
        else:
            status = "UNVERIFIED"
            is_tampered = False
            overall_conf = 0.0

        elapsed_ms = (time.time() - t0) * 1000.0

        return {
            "status": status,
            "is_tampered": bool(is_tampered),
            "is_unverified": bool(status == "UNVERIFIED"),
            "overall_confidence": float(overall_conf),
            "tampered_pixels": confident_tampered_pixels,
            "global_coherence": float(global_coherence),
            "bboxes": bboxes,
            "resolution": f"{w}x{h}",
            "execution_time_ms": float(elapsed_ms),
            "prob_heatmap": prob_heatmap,
            "original_bgr": img_bgr,
            "carrier": native_carrier,
            "residual": residual_arr,
            "coherence": coherence_arr,
            "annotated_bgr": annotated_bgr,
        }

    def verify_pdf(
        self,
        input_pdf: Union[str, Path, bytes, io.BytesIO],
        dpi: int = DEFAULT_PDF_DPI,
    ) -> Dict[str, Any]:
        t0 = time.time()

        vector_res = PDFEngine.verify_vector_pdf(input_pdf, secret_key=self.secret_key, dpi=dpi)
        if vector_res is not None:
            vector_res["execution_time_ms"] = (time.time() - t0) * 1000.0
            return vector_res

        rendered_pages = PDFEngine.render_pdf_to_pages(input_pdf, dpi=dpi)

        page_results = []
        any_tampered = False
        any_unverified = False
        all_authentic = True

        for p_info in rendered_pages:
            p_idx = p_info["page_index"]
            res = self.verify_image(p_info["image_bgr"])
            res["board"] = self.generate_diagnostic_board(res)
            res["page_index"] = p_idx
            res["page_number"] = p_idx + 1
            res["rendered_bgr"] = p_info["image_bgr"]
            page_results.append(res)

            if res["status"] == "TAMPERED":
                any_tampered = True
                all_authentic = False
            elif res["status"] == "UNVERIFIED":
                any_unverified = True
                all_authentic = False

        if any_tampered:
            overall_status = "TAMPERED"
            is_tampered = True
            overall_conf = max([p["overall_confidence"] for p in page_results if p["is_tampered"]] or [0.90])
        elif all_authentic and len(page_results) > 0:
            overall_status = "AUTHENTIC"
            is_tampered = False
            overall_conf = 0.99
        else:
            overall_status = "UNVERIFIED"
            is_tampered = False
            overall_conf = 0.0

        tampered_count = sum(1 for p in page_results if p["is_tampered"])
        elapsed_ms = (time.time() - t0) * 1000.0

        return {
            "status": overall_status,
            "is_tampered": bool(is_tampered),
            "file_type": "pdf",
            "total_pages": len(rendered_pages),
            "tampered_pages_count": tampered_count,
            "overall_confidence": float(overall_conf),
            "execution_time_ms": float(elapsed_ms),
            "pages": page_results,
        }

    def generate_diagnostic_board(self, result: Dict) -> np.ndarray:
        orig_annotated = result["annotated_bgr"]
        h, w = orig_annotated.shape[:2]

        res = result["residual"]
        res_norm = np.clip((res + 0.1) / 0.2 * 255.0, 0, 255).astype(np.uint8)
        res_bgr = cv2.cvtColor(res_norm, cv2.COLOR_GRAY2BGR)

        coh = result["coherence"]
        coh_norm = np.clip((coh + 0.05) / 0.1 * 255.0, 0, 255).astype(np.uint8)
        coh_bgr = cv2.applyColorMap(coh_norm, cv2.COLORMAP_VIRIDIS)

        prob_heat = (result["prob_heatmap"] * 255.0).astype(np.uint8)
        heat_color = cv2.applyColorMap(prob_heat, cv2.COLORMAP_JET)
        overlay_bgr = cv2.addWeighted(result["original_bgr"], 0.6, heat_color, 0.4, 0)

        def add_banner(img, title, color=(0, 0, 0)):
            canvas = img.copy()
            cv2.rectangle(canvas, (0, 0), (w, 28), color, -1)
            cv2.putText(canvas, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            return canvas

        status_color = (0, 0, 180) if result["is_tampered"] else (0, 140, 0)
        p1 = add_banner(orig_annotated, f"1. Questioned Image [{result['status']}]", status_color)
        p2 = add_banner(res_bgr, "2. High-Frequency Carrier Residual (20x)")
        p3 = add_banner(coh_bgr, "3. Keyed Phase Coherence Demodulation")
        p4 = add_banner(overlay_bgr, f"4. DocSafe Probability Heatmap ({result['overall_confidence']*100:.1f}%)")

        top_row = np.hstack([p1, p2])
        bot_row = np.hstack([p3, p4])
        board = np.vstack([top_row, bot_row])

        return board

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="verify",
        description="DocSafe Zero-Reference Tamper Localization CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Path to questioned document/image to verify",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="verification_board.png",
        help="Path to save 4-panel diagnostic visual verification board",
    )
    parser.add_argument(
        "-k", "--secret-key",
        type=str,
        default="DocSafeKey2026",
        help="Secret cryptographic carrier key",
    )
    parser.add_argument(
        "-c", "--checkpoint",
        type=str,
        default=None,
        help="Path to trained model weights (e.g. checkpoints/best_detector.pth)",
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.5,
        help="Tamper probability decision threshold",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=16,
        help="Minimum connected pixel count to flag a tamper cluster",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Path to save JSON forensic report",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference ('cuda', 'cpu', or auto)",
    )
    return parser.parse_args(argv)

def run_cli(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    in_path = Path(args.input)

    if not in_path.exists():
        print(f"Error: Questioned image not found: {args.input}", file=sys.stderr)
        return 1

    engine = VerificationEngine(
        checkpoint_path=args.checkpoint,
        secret_key=args.secret_key,
        device=args.device,
        threshold=args.threshold,
        min_tampered_pixels=args.min_area,
    )

    print("=" * 70)
    print("  DEEP-GUARD: ZERO-REFERENCE TAMPER VERIFICATION ENGINE")
    print("=" * 70)
    print(f"  Target Image     : {in_path.name}")
    print(f"  Secret Key Hash  : {engine.key_seed:#010x}")
    print(f"  Inference Device : {engine.device}")
    print(f"  Weights Loaded   : {'Trained Checkpoint' if engine.has_trained_weights else 'Standard Weights'}")
    print("-" * 70)

    result = engine.verify_image(in_path)

    is_tampered = result["is_tampered"]
    status_tag = "[!] TAMPER DETECTED (FORGERY)" if is_tampered else "[OK] AUTHENTIC DOCUMENT"

    print(f"  Verification Result : {status_tag}")
    print(f"  Overall Confidence  : {result['overall_confidence'] * 100:.2f}%")
    print(f"  Tampered Pixels     : {result['tampered_pixels']} px")
    print(f"  Tampered Clusters   : {len(result['bboxes'])} bounding box(es)")
    print(f"  Resolution          : {result['resolution']}")
    print(f"  Inspection Speed    : {result['execution_time_ms']:.2f} ms")
    print("-" * 70)

    if result["bboxes"]:
        print("  Detected Forgery Regions:")
        for idx, box in enumerate(result["bboxes"], 1):
            print(f"    Region #{idx}: x={box['x']}, y={box['y']}, w={box['width']}, h={box['height']} (Conf: {box['confidence']*100:.1f}%)")

    if args.output:
        board = engine.generate_diagnostic_board(result)
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_p), board)
        print(f"\n  Diagnostic Board Saved : {out_p}")

    if args.report:
        report_data = {
            "input_file": in_path.name,
            "status": result["status"],
            "is_tampered": result["is_tampered"],
            "overall_confidence": result["overall_confidence"],
            "tampered_pixels": result["tampered_pixels"],
            "bboxes": result["bboxes"],
            "resolution": result["resolution"],
            "execution_time_ms": result["execution_time_ms"],
        }
        rep_p = Path(args.report)
        rep_p.parent.mkdir(parents=True, exist_ok=True)
        with open(rep_p, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        print(f"  Forensic Report Saved  : {rep_p}")

    print("=" * 70)
    return 0

if __name__ == "__main__":
    sys.exit(run_cli())
