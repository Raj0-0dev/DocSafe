from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch

def compute_pixel_metrics(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> Dict[str, float]:
    if isinstance(pred, torch.Tensor):
        pred_arr = pred.detach().cpu().numpy()
    else:
        pred_arr = np.asarray(pred)

    if isinstance(target, torch.Tensor):
        target_arr = target.detach().cpu().numpy()
    else:
        target_arr = np.asarray(target)

    p_bin = (pred_arr >= threshold).astype(np.uint8).ravel()
    t_bin = (target_arr >= 0.5).astype(np.uint8).ravel()

    tp = float(np.sum((p_bin == 1) & (t_bin == 1)))
    fp = float(np.sum((p_bin == 1) & (t_bin == 0)))
    fn = float(np.sum((p_bin == 0) & (t_bin == 1)))
    tn = float(np.sum((p_bin == 0) & (t_bin == 0)))

    total = tp + fp + fn + tn
    total = max(1.0, total)

    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)
    accuracy = (tp + tn) / total

    if (tp + fn) == 0.0:
        if fp == 0.0:
            precision = 1.0
            recall = 1.0
            f1 = 1.0
            iou = 1.0
        else:
            precision = 0.0
            recall = 1.0
            f1 = 0.0
            iou = 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "specificity": float(specificity),
        "accuracy": float(accuracy),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }

def compute_image_level_decision(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    pixel_threshold: float = 0.5,
    min_tampered_pixels: int = 16,
) -> Dict[str, Union[bool, float]]:
    if isinstance(pred, torch.Tensor):
        p_arr = pred.detach().cpu().numpy()
    else:
        p_arr = np.asarray(pred)

    if isinstance(target, torch.Tensor):
        t_arr = target.detach().cpu().numpy()
    else:
        t_arr = np.asarray(target)

    pred_tampered_pixels = int(np.sum(p_arr >= pixel_threshold))
    true_tampered_pixels = int(np.sum(t_arr >= 0.5))

    is_tampered_pred = pred_tampered_pixels >= min_tampered_pixels
    is_tampered_true = true_tampered_pixels >= min_tampered_pixels

    return {
        "is_tampered_pred": bool(is_tampered_pred),
        "is_tampered_true": bool(is_tampered_true),
        "is_correct": bool(is_tampered_pred == is_tampered_true),
        "pred_tampered_pixels": pred_tampered_pixels,
        "true_tampered_pixels": true_tampered_pixels,
    }

class TamperMetricsEvaluator:

    def __init__(self, threshold: float = 0.5, min_tampered_pixels: int = 16):
        self.threshold = threshold
        self.min_tampered_pixels = min_tampered_pixels
        self.reset()

    def reset(self) -> None:
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        self.total_tn = 0

        self.clean_images_total = 0
        self.clean_images_flagged = 0

        self.tampered_images_total = 0
        self.tampered_images_detected = 0

        self.sample_ious: list = []

    def update(
        self,
        preds: Union[torch.Tensor, np.ndarray],
        targets: Union[torch.Tensor, np.ndarray],
    ) -> None:
        if isinstance(preds, torch.Tensor):
            preds_arr = preds.detach().cpu().numpy()
        else:
            preds_arr = np.asarray(preds)

        if isinstance(targets, torch.Tensor):
            targets_arr = targets.detach().cpu().numpy()
        else:
            targets_arr = np.asarray(targets)

        batch_size = preds_arr.shape[0]

        for i in range(batch_size):
            p_i = preds_arr[i]
            t_i = targets_arr[i]

            m = compute_pixel_metrics(p_i, t_i, threshold=self.threshold)
            self.total_tp += m["tp"]
            self.total_fp += m["fp"]
            self.total_fn += m["fn"]
            self.total_tn += m["tn"]
            self.sample_ious.append(m["iou"])

            img_dec = compute_image_level_decision(
                p_i, t_i,
                pixel_threshold=self.threshold,
                min_tampered_pixels=self.min_tampered_pixels,
            )

            if not img_dec["is_tampered_true"]:
                self.clean_images_total += 1
                if img_dec["is_tampered_pred"]:
                    self.clean_images_flagged += 1
            else:
                self.tampered_images_total += 1
                if img_dec["is_tampered_pred"]:
                    self.tampered_images_detected += 1

    def compute(self, eps: float = 1e-7) -> Dict[str, float]:
        tp = float(self.total_tp)
        fp = float(self.total_fp)
        fn = float(self.total_fn)
        tn = float(self.total_tn)

        precision = (tp + eps) / (tp + fp + eps)
        recall = (tp + eps) / (tp + fn + eps)
        f1 = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
        global_iou = (tp + eps) / (tp + fp + fn + eps)
        mean_sample_iou = float(np.mean(self.sample_ious)) if self.sample_ious else 0.0

        clean_fpr = (
            float(self.clean_images_flagged) / float(self.clean_images_total)
            if self.clean_images_total > 0
            else 0.0
        )
        tamper_tpr = (
            float(self.tampered_images_detected) / float(self.tampered_images_total)
            if self.tampered_images_total > 0
            else 0.0
        )

        return {
            "pixel_precision": float(precision),
            "pixel_recall": float(recall),
            "pixel_f1": float(f1),
            "pixel_iou": float(global_iou),
            "mean_sample_iou": mean_sample_iou,
            "clean_fpr": clean_fpr,
            "tamper_tpr": tamper_tpr,
            "total_pixels_evaluated": int(tp + fp + fn + tn),
            "total_images_evaluated": self.clean_images_total + self.tampered_images_total,
        }

