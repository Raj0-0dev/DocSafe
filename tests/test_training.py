"""
Unit Tests for DocSafe Training Engine (Module 5: Losses & Optimization)
==========================================================================
"""

import os
from pathlib import Path
import shutil
import tempfile
import unittest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from docsafe.training.loss import (
    FocalLoss,
    DiceLoss,
    IoULoss,
    CompoundTamperLoss,
)
from docsafe.training.trainer import TamperDetectorTrainer
from docsafe.models.unet_detector import DocSafeUNetDetector


class TestTrainingLosses(unittest.TestCase):

    def test_focal_loss_numerical_properties(self):
        """Verify focal loss computes finite, positive values and down-weights easy negatives."""
        focal = FocalLoss(alpha=0.75, gamma=2.0, from_logits=True)

        target = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
        pred_easy = torch.tensor([[[[10.0, -10.0], [-10.0, -10.0]]]])
        loss_easy = focal(pred_easy, target)

        pred_hard = torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]])
        loss_hard = focal(pred_hard, target)

        pred_wrong = torch.tensor([[[[-10.0, 10.0], [10.0, 10.0]]]])
        loss_wrong = focal(pred_wrong, target)

        self.assertLess(loss_easy.item(), loss_hard.item())
        self.assertLess(loss_hard.item(), loss_wrong.item())
        self.assertGreater(loss_easy.item(), 0.0)
        self.assertTrue(torch.isfinite(loss_wrong))

    def test_focal_loss_from_probs(self):
        """Verify focal loss works correctly when fed raw probabilities."""
        focal_prob = FocalLoss(alpha=0.75, gamma=2.0, from_logits=False)
        
        target = torch.zeros((2, 1, 32, 32))
        target[:, :, 5:10, 5:10] = 1.0  # ~2.4% tampered pixels
        
        prob = torch.full_like(target, 0.01)
        prob[:, :, 5:10, 5:10] = 0.95
        
        loss = focal_prob(prob, target)
        self.assertGreater(loss.item(), 0.0)
        self.assertTrue(torch.isfinite(loss))

    def test_dice_loss_bounds(self):
        """Verify Dice loss satisfies 0 <= L_dice <= 1."""
        dice_fn = DiceLoss(smooth=1e-6, from_logits=False)

        target = torch.zeros((1, 1, 64, 64))
        target[:, :, 20:30, 20:30] = 1.0

        loss_perfect = dice_fn(target.clone(), target)
        self.assertAlmostEqual(loss_perfect.item(), 0.0, places=4)

        inverted = 1.0 - target
        loss_inverted = dice_fn(inverted, target)
        self.assertAlmostEqual(loss_inverted.item(), 1.0, places=3)

        clean_target = torch.zeros((2, 1, 64, 64))
        clean_pred = torch.zeros((2, 1, 64, 64))
        loss_clean = dice_fn(clean_pred, clean_target)
        self.assertAlmostEqual(loss_clean.item(), 0.0, places=4)

    def test_iou_loss(self):
        """Verify IoU loss behaves monotonically with spatial overlap."""
        iou_fn = IoULoss(smooth=1e-6, from_logits=False)

        target = torch.zeros((1, 1, 32, 32))
        target[:, :, 10:20, 10:20] = 1.0

        pred_100 = target.clone()
        self.assertAlmostEqual(iou_fn(pred_100, target).item(), 0.0, places=4)

        pred_50 = torch.zeros((1, 1, 32, 32))
        pred_50[:, :, 10:20, 15:25] = 1.0
        loss_50 = iou_fn(pred_50, target).item()
        self.assertAlmostEqual(loss_50, 0.6667, places=2)

    def test_compound_loss_backward(self):
        """Verify CompoundTamperLoss propagates stable gradients to model parameters."""
        criterion = CompoundTamperLoss(alpha=0.75, gamma=2.0, lambda_focal=1.0, lambda_dice=1.0)
        
        conv = nn.Conv2d(4, 1, kernel_size=1)
        features = torch.randn(2, 4, 32, 32, requires_grad=True)
        target = torch.zeros((2, 1, 32, 32))
        target[:, :, 12:18, 12:18] = 1.0

        pred_logits = conv(features)
        loss, metrics = criterion(pred_logits, target, return_dict=True)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.ndim, 0)
        self.assertIn("loss_total", metrics)
        self.assertIn("loss_focal", metrics)
        self.assertIn("loss_dice", metrics)

        loss.backward()

        self.assertIsNotNone(conv.weight.grad)
        self.assertTrue(torch.all(torch.isfinite(conv.weight.grad)))
        self.assertGreater(conv.weight.grad.abs().sum().item(), 0.0)

    def test_compound_loss_dimension_flexibility(self):
        """Verify handling of 3D vs 4D input tensors."""
        criterion = CompoundTamperLoss()
        pred = torch.randn(2, 64, 64)
        target = torch.zeros(2, 1, 64, 64)
        target[:, :, 10:20, 10:20] = 1.0

        loss = criterion(pred, target)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)


class MockDataset(torch.utils.data.Dataset):
    """Generates synthetic batch dictionaries for trainer testing."""
    def __init__(self, size=4, h=64, w=64):
        self.size = size
        self.h = h
        self.w = w

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img = torch.rand(3, self.h, self.w)
        carrier = torch.ones(1, self.h, self.w)
        mask = torch.zeros(1, self.h, self.w)
        mask[:, 10:20, 10:20] = 1.0
        return {"image": img, "carrier": carrier, "mask": mask}


def mock_collate_fn(batch):
    return {
        "image": torch.stack([x["image"] for x in batch]),
        "carrier": torch.stack([x["carrier"] for x in batch]),
        "mask": torch.stack([x["mask"] for x in batch]),
    }


class TestTrainerEngine(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        # Compact detector for lightning-fast testing
        self.model = DocSafeUNetDetector(
            in_fused_channels=8,
            encoder_dims=(16, 32, 64, 128),
            decoder_out_dim=16,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_trainer_step_weight_update(self):
        """Submodule 5.2: Test that a single train_step updates model weights."""
        trainer = TamperDetectorTrainer(
            model=self.model,
            device="cpu",
            checkpoint_dir=self.temp_dir,
        )
        
        batch = {
            "image": torch.rand(2, 3, 64, 64),
            "carrier": torch.ones(2, 1, 64, 64),
            "mask": torch.zeros(2, 1, 64, 64),
        }
        batch["mask"][:, :, 10:20, 10:20] = 1.0

        # Snapshot initial weight of head
        initial_param = next(self.model.seg_head.parameters()).clone().detach()

        loss_val, metrics = trainer.train_step(batch)

        updated_param = next(self.model.seg_head.parameters()).clone().detach()

        self.assertGreater(loss_val, 0.0)
        self.assertIn("loss_focal", metrics)
        self.assertIn("loss_dice", metrics)
        # Weights must be modified by AdamW step
        self.assertFalse(torch.equal(initial_param, updated_param))

    def test_trainer_epoch_metrics(self):
        """Submodule 5.2: Test train_epoch processes batches and returns averaged telemetry."""
        trainer = TamperDetectorTrainer(
            model=self.model,
            device="cpu",
            checkpoint_dir=self.temp_dir,
        )
        loader = DataLoader(MockDataset(size=4), batch_size=2, collate_fn=mock_collate_fn)

        metrics = trainer.train_epoch(loader)

        self.assertIn("loss", metrics)
        self.assertIn("focal", metrics)
        self.assertIn("dice", metrics)
        self.assertIn("lr", metrics)
        self.assertEqual(len(trainer.history["train_loss"]), 1)

    def test_trainer_validation_no_grad(self):
        """Submodule 5.2: Test validate_epoch computes loss without accumulating gradients."""
        trainer = TamperDetectorTrainer(
            model=self.model,
            device="cpu",
            checkpoint_dir=self.temp_dir,
        )
        loader = DataLoader(MockDataset(size=2), batch_size=2, collate_fn=mock_collate_fn)

        # Clear grads
        trainer.optimizer.zero_grad()
        val_m = trainer.validate_epoch(loader)

        self.assertIn("loss", val_m)
        self.assertIn("dice", val_m)
        # Verify no gradient buffers were created during validation
        for p in self.model.parameters():
            self.assertIsNone(p.grad)

    def test_checkpoint_save_and_load(self):
        """Submodule 5.2: Test serializing and deserializing training checkpoints."""
        trainer = TamperDetectorTrainer(
            model=self.model,
            device="cpu",
            checkpoint_dir=self.temp_dir,
        )

        ckpt_path = trainer.save_checkpoint(epoch=3, filename="test_ckpt.pth")
        self.assertTrue(ckpt_path.exists())

        # Modify model weight manually
        with torch.no_grad():
            for p in self.model.parameters():
                p.add_(1.0)

        # Reload from checkpoint
        loaded = trainer.load_checkpoint(ckpt_path)
        self.assertEqual(loaded["epoch"], 3)

    def test_fit_mini_curriculum(self):
        """Submodule 5.2: Test full training orchestration with best weight checkpointing."""
        trainer = TamperDetectorTrainer(
            model=self.model,
            device="cpu",
            checkpoint_dir=self.temp_dir,
        )
        train_loader = DataLoader(MockDataset(size=4), batch_size=2, collate_fn=mock_collate_fn)
        val_loader = DataLoader(MockDataset(size=2), batch_size=2, collate_fn=mock_collate_fn)

        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=2,
            verbose=False,
        )

        self.assertEqual(len(history["train_loss"]), 2)
        self.assertEqual(len(history["val_loss"]), 2)
        # Verify checkpoints were created
        self.assertTrue((Path(self.temp_dir) / "best_detector.pth").exists())
        self.assertTrue((Path(self.temp_dir) / "latest_checkpoint.pth").exists())


class TestMetricsEngine(unittest.TestCase):

    def test_compute_pixel_metrics_perfect_and_imperfect(self):
        """Submodule 5.3: Test pixel-level precision, recall, F1, and IoU calculations."""
        from docsafe.training.metrics import compute_pixel_metrics

        # Target has 100 positive pixels
        target = np.zeros((32, 32), dtype=np.float32)
        target[10:20, 10:20] = 1.0  # 100 pixels

        # Prediction: 50 true positives, 50 false positives, 50 false negatives
        pred = np.zeros((32, 32), dtype=np.float32)
        pred[10:20, 15:25] = 0.9  # 100 pixels, overlap with target is 10x5 = 50

        metrics = compute_pixel_metrics(pred, target, threshold=0.5)

        # TP=50, FP=50, FN=50, TN=1024 - 150 = 874
        self.assertEqual(metrics["tp"], 50)
        self.assertEqual(metrics["fp"], 50)
        self.assertEqual(metrics["fn"], 50)
        self.assertAlmostEqual(metrics["precision"], 50.0 / 100.0, places=3)
        self.assertAlmostEqual(metrics["recall"], 50.0 / 100.0, places=3)
        self.assertAlmostEqual(metrics["f1"], 50.0 / 100.0, places=3)
        self.assertAlmostEqual(metrics["iou"], 50.0 / 150.0, places=3)

    def test_compute_pixel_metrics_clean_negative_image(self):
        """Submodule 5.3: Test evaluation on clean pristine document without false alarms."""
        from docsafe.training.metrics import compute_pixel_metrics

        clean_target = np.zeros((64, 64), dtype=np.float32)
        clean_pred = np.zeros((64, 64), dtype=np.float32)

        metrics = compute_pixel_metrics(clean_pred, clean_target)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)
        self.assertEqual(metrics["iou"], 1.0)
        self.assertEqual(metrics["specificity"], 1.0)

    def test_compute_image_level_decision(self):
        """Submodule 5.3: Test image authenticity flag with small-cluster noise suppression."""
        from docsafe.training.metrics import compute_image_level_decision

        # 1. Clean image with 4 random noise speckle pixels (should NOT flag as tampered)
        clean_target = np.zeros((64, 64), dtype=np.float32)
        noise_pred = np.zeros((64, 64), dtype=np.float32)
        noise_pred[5, 5] = 0.9
        noise_pred[10, 12] = 0.8
        noise_pred[20, 25] = 0.7
        noise_pred[30, 35] = 0.6  # 4 pixels total < min_tampered_pixels (16)

        dec = compute_image_level_decision(noise_pred, clean_target, min_tampered_pixels=16)
        self.assertFalse(dec["is_tampered_pred"])
        self.assertFalse(dec["is_tampered_true"])
        self.assertTrue(dec["is_correct"])

        # 2. Tampered image with 25 pixels altered
        tampered_target = np.zeros((64, 64), dtype=np.float32)
        tampered_target[10:15, 10:15] = 1.0  # 25 pixels
        tampered_pred = np.zeros((64, 64), dtype=np.float32)
        tampered_pred[10:15, 10:15] = 0.95

        dec_t = compute_image_level_decision(tampered_pred, tampered_target, min_tampered_pixels=16)
        self.assertTrue(dec_t["is_tampered_pred"])
        self.assertTrue(dec_t["is_tampered_true"])
        self.assertTrue(dec_t["is_correct"])

    def test_tamper_metrics_evaluator_batch_streaming(self):
        """Submodule 5.3: Test streaming accumulator across mini-batches."""
        from docsafe.training.metrics import TamperMetricsEvaluator

        evaluator = TamperMetricsEvaluator(threshold=0.5, min_tampered_pixels=16)

        # Batch 1: 2 clean images
        b1_target = torch.zeros((2, 1, 32, 32))
        b1_pred = torch.zeros((2, 1, 32, 32))
        evaluator.update(b1_pred, b1_target)

        # Batch 2: 2 tampered images
        b2_target = torch.zeros((2, 1, 32, 32))
        b2_target[:, :, 8:16, 8:16] = 1.0  # 64 pixels each
        b2_pred = torch.zeros((2, 1, 32, 32))
        b2_pred[:, :, 8:16, 8:16] = 0.92
        evaluator.update(b2_pred, b2_target)

        results = evaluator.compute()

        self.assertEqual(results["total_images_evaluated"], 4)
        self.assertEqual(results["clean_fpr"], 0.0)      # 0 clean images falsely flagged
        self.assertEqual(results["tamper_tpr"], 1.0)     # 100% of tampered images detected
        self.assertAlmostEqual(results["pixel_f1"], 1.0, places=3)
        self.assertAlmostEqual(results["pixel_iou"], 1.0, places=3)

        # Verify reset clears internal state
        evaluator.reset()
        self.assertEqual(evaluator.clean_images_total, 0)
        self.assertEqual(evaluator.tampered_images_total, 0)


if __name__ == "__main__":
    unittest.main()
