"""
DocSafe: Deep Learning Training & Optimization Engine
=========================================================
Submodule 5.2: TamperDetectorTrainer
Orchestrates model optimization with:
- AdamW optimizer & Cosine Annealing learning rate schedule
- Gradient clipping (L2 norm) for numerical stability
- Compound loss tracking (Focal + Dice)
- Model checkpointing & best-weight preservation
- Device-agnostic execution (NVIDIA CUDA / Apple MPS / Multi-threaded CPU)
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from docsafe.training.loss import CompoundTamperLoss


class TamperDetectorTrainer:
    """
    Submodule 5.2: Production-grade trainer for DocSafe Tamper Localization.
    
    Coordinates the training loop, gradient backpropagation, learning rate
    scheduling, validation telemetry, and persistent weight checkpointing.
    
    Args:
        model: nn.Module instance (e.g. DocSafeUNetDetector).
        criterion: Loss function module (defaults to CompoundTamperLoss).
        optimizer: PyTorch optimizer (defaults to AdamW with lr=1e-3, wd=1e-4).
        scheduler: Optional learning rate scheduler.
        device: 'cuda', 'cpu', or torch.device.
        gradient_clip: Maximum L2 gradient norm for clipping (default: 1.0).
        checkpoint_dir: Directory path where weights are saved.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[Union[str, torch.device]] = None,
        gradient_clip: float = 1.0,
        checkpoint_dir: Union[str, Path] = "checkpoints",
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.model = model.to(self.device)

        # Detector output passes through Sigmoid in SegmentationHead, so from_logits=False
        if criterion is None:
            self.criterion = CompoundTamperLoss(
                alpha=0.75,
                gamma=2.0,
                lambda_focal=1.0,
                lambda_dice=1.0,
                from_logits=False,
            )
        else:
            self.criterion = criterion.to(self.device)

        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=1e-3,
                weight_decay=1e-4,
                betas=(0.9, 0.999),
            )
        else:
            self.optimizer = optimizer

        self.scheduler = scheduler
        self.gradient_clip = float(gradient_clip)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_focal": [],
            "train_dice": [],
            "val_loss": [],
            "lr": [],
        }
        self.best_val_loss = float("inf")

    def train_step(self, batch: dict) -> Tuple[float, Dict[str, float]]:
        """
        Execute a single forward-backward optimization step on a mini-batch.
        
        Args:
            batch: Dictionary with 'image', 'carrier', 'mask'.
            
        Returns:
            Tuple of (scalar_loss_float, metrics_dict).
        """
        self.model.train()
        self.optimizer.zero_grad()

        images = batch["image"].to(self.device)
        carriers = batch["carrier"].to(self.device)
        masks = batch["mask"].to(self.device)

        # Forward pass
        preds = self.model(images, carriers)

        # Loss calculation
        loss, loss_dict = self.criterion(preds, masks, return_dict=True)

        # Backward propagation
        loss.backward()

        # Gradient clipping
        if self.gradient_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip)

        # Optimizer update
        self.optimizer.step()

        return loss.item(), loss_dict

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Execute training across one complete epoch of data.
        
        Args:
            dataloader: DataLoader streaming training batches.
            
        Returns:
            Dictionary of averaged epoch training metrics.
        """
        self.model.train()
        running_total = 0.0
        running_focal = 0.0
        running_dice = 0.0
        total_batches = 0

        for batch in dataloader:
            loss_val, metrics = self.train_step(batch)
            running_total += loss_val
            running_focal += metrics.get("loss_focal", 0.0)
            running_dice += metrics.get("loss_dice", 0.0)
            total_batches += 1

        n = max(1, total_batches)
        current_lr = self.optimizer.param_groups[0]["lr"]

        epoch_metrics = {
            "loss": running_total / n,
            "focal": running_focal / n,
            "dice": running_dice / n,
            "lr": current_lr,
        }

        self.history["train_loss"].append(epoch_metrics["loss"])
        self.history["train_focal"].append(epoch_metrics["focal"])
        self.history["train_dice"].append(epoch_metrics["dice"])
        self.history["lr"].append(current_lr)

        return epoch_metrics

    @torch.no_grad()
    def validate_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model performance on an unseen validation dataset without gradient computation.
        
        Args:
            dataloader: DataLoader streaming validation batches.
            
        Returns:
            Dictionary of averaged validation metrics.
        """
        self.model.eval()
        running_total = 0.0
        running_focal = 0.0
        running_dice = 0.0
        total_batches = 0

        for batch in dataloader:
            images = batch["image"].to(self.device)
            carriers = batch["carrier"].to(self.device)
            masks = batch["mask"].to(self.device)

            preds = self.model(images, carriers)
            loss, metrics = self.criterion(preds, masks, return_dict=True)

            running_total += loss.item()
            running_focal += metrics.get("loss_focal", 0.0)
            running_dice += metrics.get("loss_dice", 0.0)
            total_batches += 1

        n = max(1, total_batches)
        val_metrics = {
            "loss": running_total / n,
            "focal": running_focal / n,
            "dice": running_dice / n,
        }
        self.history["val_loss"].append(val_metrics["loss"])
        return val_metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 5,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Run the complete training curriculum.
        
        Args:
            train_loader: Training DataLoader.
            val_loader: Optional validation DataLoader.
            epochs: Total number of epochs to train.
            verbose: If True, prints epoch progress and metrics.
            
        Returns:
            Training history dictionary.
        """
        for epoch in range(1, epochs + 1):
            train_m = self.train_epoch(train_loader)

            val_str = ""
            if val_loader is not None:
                val_m = self.validate_epoch(val_loader)
                val_str = f" | Val Loss: {val_m['loss']:.4f}"

                # Check if this is the best checkpoint
                if val_m["loss"] < self.best_val_loss:
                    self.best_val_loss = val_m["loss"]
                    self.save_checkpoint(
                        epoch=epoch,
                        filename="best_detector.pth",
                        extra_info={"val_loss": val_m["loss"]},
                    )

            if self.scheduler is not None:
                self.scheduler.step()

            if verbose:
                print(
                    f"Epoch [{epoch:02d}/{epochs:02d}] "
                    f"Loss: {train_m['loss']:.4f} "
                    f"(Focal: {train_m['focal']:.4f}, Dice: {train_m['dice']:.4f}) "
                    f"LR: {train_m['lr']:.6f}"
                    f"{val_str}"
                )

        # Save latest checkpoint at the end
        self.save_checkpoint(epoch=epochs, filename="latest_checkpoint.pth")
        return self.history

    def save_checkpoint(
        self,
        epoch: int,
        filename: str = "checkpoint.pth",
        extra_info: Optional[dict] = None,
    ) -> Path:
        """
        Serialize model weights, optimizer, and training state to disk.
        """
        save_path = self.checkpoint_dir / filename
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "history": self.history,
            "best_val_loss": self.best_val_loss,
            "extra_info": extra_info or {},
        }
        torch.save(state, save_path)
        return save_path

    def load_checkpoint(self, filepath: Union[str, Path]) -> dict:
        """
        Restore model weights, optimizer, and training telemetry from disk.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)
        raw_state_dict = checkpoint.get("model_state_dict", checkpoint)

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

        if "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"]:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception:
                pass

        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.history = checkpoint.get("history", self.history)
        self.best_val_loss = checkpoint.get("best_val_loss", self.best_val_loss)

        return checkpoint

