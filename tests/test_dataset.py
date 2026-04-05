"""Tests for PiLoT dataset and loss functions."""

from __future__ import annotations

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Loss function tests
# ---------------------------------------------------------------------------

def test_barron_loss_l2():
    """Barron loss with alpha=2 should behave like L2."""
    from pilot.losses import BarronRobustLoss

    loss_fn = BarronRobustLoss(alpha=2.0, scale=1.0)
    x = torch.tensor([0.0, 1.0, 4.0])
    result = loss_fn(x)
    expected = 0.5 * x  # L2: 0.5 * x
    assert torch.allclose(result, expected, atol=1e-5)


def test_barron_loss_cauchy():
    """Barron loss with alpha=0 should behave like Cauchy."""
    from pilot.losses import BarronRobustLoss

    loss_fn = BarronRobustLoss(alpha=0.0, scale=1.0)
    x = torch.tensor([0.0, 1.0, 4.0])
    result = loss_fn(x)
    expected = torch.log1p(0.5 * x)
    assert torch.allclose(result, expected, atol=1e-5)


def test_barron_loss_differentiable():
    """Barron loss should be differentiable."""
    from pilot.losses import BarronRobustLoss

    loss_fn = BarronRobustLoss(alpha=0.0, scale=1.0)
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    loss = loss_fn(x).sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_photometric_cost_shape():
    """PhotometricCost should return scalar."""
    from pilot.losses import PhotometricCost

    cost_fn = PhotometricCost(huber_delta=1.0)
    q = torch.randn(2, 32, 16, 16)
    r = torch.randn(2, 32, 16, 16)
    unc = torch.sigmoid(torch.randn(2, 1, 16, 16))
    cost = cost_fn(q, r, unc)
    assert cost.ndim == 0  # scalar


def test_se3_motion_regularization():
    """SE3MotionRegularization should be zero for identical poses."""
    from pilot.losses import SE3MotionRegularization

    reg = SE3MotionRegularization(lam=1.0)
    T = torch.eye(4).unsqueeze(0)
    cost = reg(T, T)
    assert cost.item() < 1e-5


def test_reprojection_loss():
    """ReprojectionLoss should be zero for matching points."""
    from pilot.losses import ReprojectionLoss

    loss_fn = ReprojectionLoss(alpha=0.0, scale=1.0)
    pts = torch.randn(2, 50, 2)
    loss = loss_fn(pts, pts)
    assert loss.item() < 1e-5


def test_pilot_loss_combined():
    """PiLoTLoss should return dict with total, reproj, photo, motion."""
    from pilot.losses import PiLoTLoss

    loss_fn = PiLoTLoss()
    pts_pred = torch.randn(2, 50, 2)
    pts_gt = torch.randn(2, 50, 2)
    q_feat = torch.randn(2, 32, 8, 8)
    r_feat = torch.randn(2, 32, 8, 8)
    T_est = torch.eye(4).unsqueeze(0).expand(2, -1, -1).clone()
    T_pred = torch.eye(4).unsqueeze(0).expand(2, -1, -1).clone()

    losses = loss_fn(
        pts_pred=pts_pred,
        pts_gt=pts_gt,
        query_feat=q_feat,
        ref_feat=r_feat,
        T_est=T_est,
        T_pred=T_pred,
    )
    assert "total" in losses
    assert "reproj" in losses
    assert "photo" in losses
    assert "motion" in losses


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------

def test_depth_to_geo_anchors():
    """depth_to_geo_anchors should produce (B, N, 3) output."""
    from pilot.dataset import depth_to_geo_anchors

    B, H, W = 2, 64, 80
    depth = torch.ones(B, H, W) * 50.0  # 50m uniform depth
    T_ref = torch.eye(4).unsqueeze(0).expand(B, -1, -1)
    K = torch.tensor([
        [250, 0, 40],
        [0, 250, 32],
        [0, 0, 1],
    ], dtype=torch.float32).unsqueeze(0).expand(B, -1, -1)

    anchors = depth_to_geo_anchors(depth, T_ref, K, num_anchors=100)
    assert anchors.shape == (B, 100, 3)
    # Z values should be around 50m
    assert anchors[:, :, 2].mean().item() > 10.0


def test_perturb_pose():
    """perturb_pose should produce a different pose."""
    from pilot.dataset import perturb_pose

    T = np.eye(4, dtype=np.float32)
    T_noisy = perturb_pose(T, t_range=(5.0, 15.0), r_range=(5.0, 15.0))
    assert T_noisy.shape == (4, 4)
    assert not np.allclose(T, T_noisy)


# ---------------------------------------------------------------------------
# Evaluation tests
# ---------------------------------------------------------------------------

def test_compute_pose_error():
    """compute_pose_error should return zero for identical poses."""
    from pilot.evaluate import compute_pose_error

    T = torch.eye(4).unsqueeze(0)
    t_err, r_err = compute_pose_error(T, T)
    assert t_err.item() < 1e-5
    assert r_err.item() < 0.05  # acos numerical precision near identity


def test_compute_recall():
    """compute_recall should return 100% when all errors below threshold."""
    from pilot.evaluate import compute_recall

    errors = np.array([0.1, 0.5, 0.9, 1.5, 2.0])
    result = compute_recall(errors, [1.0, 3.0, 5.0])
    assert result["recall@1.0"] == pytest.approx(60.0)
    assert result["recall@3.0"] == pytest.approx(100.0)
    assert result["recall@5.0"] == pytest.approx(100.0)


def test_evaluator_accumulate():
    """PiLoTEvaluator should accumulate multiple batches."""
    from pilot.evaluate import PiLoTEvaluator

    evaluator = PiLoTEvaluator()
    T_gt = torch.eye(4).unsqueeze(0)

    for _ in range(5):
        # Small noise -> small error
        noise = torch.randn(1, 4, 4) * 0.001
        T_pred = T_gt + noise
        evaluator.update(T_pred, T_gt, elapsed_s=0.01)

    results = evaluator.compute()
    assert results.num_frames == 5
    assert results.median_t_error < 1.0  # should be very small
