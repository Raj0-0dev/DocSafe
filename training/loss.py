"""
DocSafe: Compound Loss Engine for Tamper Localization
=========================================================
Implements specialized class-imbalanced segmentation losses:
1. Focal Loss: Addresses extreme spatial imbalance (tampered pixels << clean background).
2. Dice Loss: Enforces boundary sharpness and connected-component spatial coherence.
3. IoU (Jaccard) Loss: Directly optimizes bounding region intersection-over-union.
4. CompoundTamperLoss: Weighted hybrid loss with component telemetry.
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    r"""
    Focal Loss for Dense Binary Imbalanced Segmentation.
    
    .. math::
        \text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)
        
    Down-weights easy well-classified background pixels, focusing gradients
    on hard boundary edges and subtle altered digits.
    
    Args:
        alpha: Weighting factor for positive (tampered) class (default: 0.75).
        gamma: Focusing parameter for modulating hard examples (default: 2.0).
        from_logits: Whether input tensor contains raw logits (True) or probabilities (False).
        reduction: 'mean', 'sum', or 'none'.
    """
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
        """
        Compute focal loss.
        
        Args:
            pred: (B, 1, H, W) or (B, H, W) logits or probabilities.
            target: (B, 1, H, W) or (B, H, W) binary ground truth labels in [0, 1].
            
        Returns:
            Scalar tensor if reduction='mean' or 'sum', else tensor of loss values.
        """
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
    r"""
    Soft Sørensen–Dice Loss for Image Segmentation.
    
    .. math::
        \text{Dice} = \frac{2 \sum (p \cdot y) + \epsilon}{\sum p + \sum y + \epsilon}
        \mathcal{L}_{\text{dice}} = 1 - \text{Dice}
        
    Directly measures spatial overlap, invariant to background-to-foreground ratio.
    Ensures connected components and prevents false positive speckles.
    
    Args:
        smooth: Laplace smoothing constant to prevent zero-division (default: 1e-6).
        from_logits: Whether input tensor contains raw logits (default: True).
        reduction: 'mean' across batch or 'none'.
    """
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
        """
        Compute soft dice loss.
        
        Args:
            pred: (B, 1, H, W) or (B, H, W) predictions.
            target: (B, 1, H, W) or (B, H, W) ground truth in [0, 1].
            
        Returns:
            Scalar tensor representing 1 - Dice.
        """
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

        # Flatten spatial dimensions per batch sample: (B, N)
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
    r"""
    Soft Jaccard / Intersection-over-Union Loss.
    
    .. math::
        \text{IoU} = \frac{\sum (p \cdot y) + \epsilon}{\sum p + \sum y - \sum (p \cdot y) + \epsilon}
        \mathcal{L}_{\text{IoU}} = 1 - \text{IoU}
    """
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
    r"""
    Unified Compound Objective for DocSafe Tamper Localization.
    
    Combines Focal Loss and Dice Loss to simultaneously handle extreme class
    imbalance (0.1% - 5% tampered pixels) and maintain precise character boundary delineations:
    
    .. math::
        \mathcal{L}_{\text{total}} = \lambda_{\text{focal}} \mathcal{L}_{\text{focal}} + \lambda_{\text{dice}} \mathcal{L}_{\text{dice}}
        
    Args:
        alpha: Positive class weight for Focal Loss (default: 0.75).
        gamma: Hard-example focusing factor for Focal Loss (default: 2.0).
        lambda_focal: Weighting multiplier for Focal Loss (default: 1.0).
        lambda_dice: Weighting multiplier for Dice Loss (default: 1.0).
        from_logits: True if input is raw unnormalized logits (default: True).
        smooth: Smoothing constant for Dice denominator (default: 1e-6).
    """
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
        """
        Compute compound loss.
        
        Args:
            pred: Predicted logits or probability map of shape (B, 1, H, W).
            target: Ground truth binary mask of shape (B, 1, H, W) with values {0, 1}.
            return_dict: If True, returns (total_loss, metrics_dict). If False, returns total_loss.
            
        Returns:
            total_loss or tuple (total_loss, metrics_dict).
        """
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

