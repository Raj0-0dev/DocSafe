from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        from_logits: bool = True,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.from_logits = from_logits
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            if pred.ndim == 4 and target.ndim == 3:
                target = target.unsqueeze(1)
            elif pred.ndim == 3 and target.ndim == 4:
                pred = pred.unsqueeze(1)

        target = target.float()

        if self.from_logits:
            bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
            p = torch.sigmoid(pred)
        else:
            pred_clamped = torch.clamp(pred, min=1e-7, max=1.0 - 1e-7)
            bce_loss = F.binary_cross_entropy(pred_clamped, target, reduction="none")
            p = pred_clamped

        p_t = p * target + (1.0 - p) * (1.0 - target)
        alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        focal_weight = alpha_t * torch.pow((1.0 - p_t), self.gamma)

        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

class DiceLoss(nn.Module):
    def __init__(
        self,
        smooth: float = 1e-6,
        from_logits: bool = True,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.smooth = float(smooth)
        self.from_logits = from_logits
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            if pred.ndim == 4 and target.ndim == 3:
                target = target.unsqueeze(1)
            elif pred.ndim == 3 and target.ndim == 4:
                pred = pred.unsqueeze(1)

        target = target.float()

        if self.from_logits:
            probs = torch.sigmoid(pred)
        else:
            probs = torch.clamp(pred, min=0.0, max=1.0)

        batch_size = probs.shape[0]
        p_flat = probs.view(batch_size, -1)
        t_flat = target.view(batch_size, -1)

        intersection = torch.sum(p_flat * t_flat, dim=1)
        cardinality = torch.sum(p_flat, dim=1) + torch.sum(t_flat, dim=1)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        loss = 1.0 - dice

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

class IoULoss(nn.Module):
    def __init__(
        self,
        smooth: float = 1e-6,
        from_logits: bool = True,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.smooth = float(smooth)
        self.from_logits = from_logits
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            if pred.ndim == 4 and target.ndim == 3:
                target = target.unsqueeze(1)
            elif pred.ndim == 3 and target.ndim == 4:
                pred = pred.unsqueeze(1)

        target = target.float()

        if self.from_logits:
            probs = torch.sigmoid(pred)
        else:
            probs = torch.clamp(pred, min=0.0, max=1.0)

        batch_size = probs.shape[0]
        p_flat = probs.view(batch_size, -1)
        t_flat = target.view(batch_size, -1)

        intersection = torch.sum(p_flat * t_flat, dim=1)
        union = torch.sum(p_flat, dim=1) + torch.sum(t_flat, dim=1) - intersection

        iou = (intersection + self.smooth) / (union + self.smooth)
        loss = 1.0 - iou

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

class CompoundTamperLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        lambda_focal: float = 1.0,
        lambda_dice: float = 1.0,
        from_logits: bool = True,
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()
        self.lambda_focal = float(lambda_focal)
        self.lambda_dice = float(lambda_dice)
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, from_logits=from_logits, reduction="mean")
        self.dice_loss = DiceLoss(smooth=smooth, from_logits=from_logits, reduction="mean")
        self.last_metrics: Dict[str, float] = {}

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        l_focal = self.focal_loss(pred, target)
        l_dice = self.dice_loss(pred, target)

        total_loss = self.lambda_focal * l_focal + self.lambda_dice * l_dice

        self.last_metrics = {
            "loss_total": float(total_loss.detach().item()),
            "loss_focal": float(l_focal.detach().item()),
            "loss_dice": float(l_dice.detach().item()),
        }

        if return_dict:
            return total_loss, self.last_metrics
        return total_loss

