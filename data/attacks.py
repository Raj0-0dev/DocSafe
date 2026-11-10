import random
from typing import Tuple, Dict, Any, Optional

import cv2
import numpy as np

class TamperSynthesizer:

    def __init__(
        self,
        min_mask_ratio: float = 0.005,
        max_mask_ratio: float = 0.15,
        negative_sample_prob: float = 0.15,
        seed: Optional[int] = None,
    ):
        self.min_mask_ratio = min_mask_ratio
        self.max_mask_ratio = max_mask_ratio
        self.negative_sample_prob = negative_sample_prob
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def generate_tampered_pair(
        self,
        protected_img: np.ndarray,
        attack_type: str = "random",
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        h, w = protected_img.shape[:2]
        is_float = np.issubdtype(protected_img.dtype, np.floating)

        if is_float:
            img_uint8 = np.clip(protected_img * 255.0, 0, 255).astype(np.uint8)
        else:
            img_uint8 = protected_img.copy()

        if attack_type == "random":
            if random.random() < self.negative_sample_prob:
                attack_type = "negative"
            else:
                attack_type = random.choice(["box", "brushstroke", "splice"])

        if attack_type == "negative":
            mask = np.zeros((h, w), dtype=np.float32)
            meta = {"attack_type": "negative", "bboxes": [], "coverage_pct": 0.0}
            return protected_img.copy(), mask, meta

        mask_uint8 = np.zeros((h, w), dtype=np.uint8)
        bboxes = []

        if attack_type == "box":
            num_boxes = random.randint(1, 3)
            for _ in range(num_boxes):
                max_bw = max(8, min(80, int(w * 0.35)))
                min_bw = min(8, max_bw)
                bw = random.randint(min_bw, max_bw)

                max_bh = max(6, min(50, int(h * 0.25)))
                min_bh = min(6, max_bh)
                bh = random.randint(min_bh, max_bh)

                max_x = max(2, w - bw - 2)
                max_y = max(2, h - bh - 2)
                bx = random.randint(1, max_x)
                by = random.randint(1, max_y)
                cv2.rectangle(mask_uint8, (bx, by), (bx + bw, by + bh), 255, -1)
                bboxes.append([bx, by, bw, bh])

        elif attack_type == "brushstroke":
            num_strokes = random.randint(1, 2)
            for _ in range(num_strokes):
                num_points = random.randint(3, 6)
                points = []
                center_x = random.randint(max(2, int(w * 0.1)), max(3, int(w * 0.9)))
                center_y = random.randint(max(2, int(h * 0.1)), max(3, int(h * 0.9)))
                span = max(4, min(w, h) // 4)

                for _ in range(num_points):
                    px = int(np.clip(center_x + random.randint(-span, span), 2, w - 3))
                    py = int(np.clip(center_y + random.randint(-span, span), 2, h - 3))
                    points.append([px, py])

                max_thick = max(4, min(24, min(w, h) // 8))
                min_thick = max(2, min(8, max_thick // 2))
                thickness = random.randint(min_thick, max_thick)
                for i in range(len(points) - 1):
                    cv2.line(mask_uint8, tuple(points[i]), tuple(points[i + 1]), 255, thickness)
                cv2.circle(mask_uint8, tuple(points[-1]), thickness // 2, 255, -1)

        elif attack_type == "splice":
            max_bw = max(10, min(90, int(w * 0.35)))
            min_bw = min(10, max_bw)
            bw = random.randint(min_bw, max_bw)

            max_bh = max(8, min(70, int(h * 0.25)))
            min_bh = min(8, max_bh)
            bh = random.randint(min_bh, max_bh)

            max_x = max(2, w - bw - 2)
            max_y = max(2, h - bh - 2)
            dst_x = random.randint(1, max_x)
            dst_y = random.randint(1, max_y)
            src_x = random.randint(1, max_x)
            src_y = random.randint(1, max_y)

            src_patch = img_uint8[src_y : src_y + bh, src_x : src_x + bw].copy()
            img_uint8[dst_y : dst_y + bh, dst_x : dst_x + bw] = src_patch
            cv2.rectangle(mask_uint8, (dst_x, dst_y), (dst_x + bw, dst_y + bh), 255, -1)
            bboxes.append([dst_x, dst_y, bw, bh])

        if attack_type in ["box", "brushstroke"]:
            blurred_bg = cv2.GaussianBlur(img_uint8, (15, 15), 0)
            img_uint8[mask_uint8 == 255] = blurred_bg[mask_uint8 == 255]

            if attack_type == "box":
                for bx, by, bw, bh in bboxes:
                    glyph = random.choice(["8", "9", "5", "0"])
                    cx = bx + max(2, bw // 4)
                    cy = by + int(bh * 0.8)
                    cv2.putText(
                        img_uint8,
                        glyph,
                        (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        min(bw, bh) / 28.0,
                        (40, 40, 40),
                        max(1, int(min(bw, bh) / 16)),
                        cv2.LINE_AA,
                    )

        mask_f32 = (mask_uint8 > 127).astype(np.float32)

        if is_float:
            tampered_out = img_uint8.astype(np.float32) / 255.0
        else:
            tampered_out = img_uint8

        coverage = float(np.sum(mask_f32) / (h * w) * 100.0)
        meta = {
            "attack_type": attack_type,
            "bboxes": bboxes,
            "coverage_pct": round(coverage, 3),
        }

        return tampered_out, mask_f32, meta
