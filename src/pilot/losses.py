"""PiLoT loss functions.

Paper: arXiv 2603.20778
- Barron's robust loss (Eq.3) for training
- Photometric cost (Eq.7) with uncertainty weighting
- SE(3) motion regularization (Eq.10)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pilot.model import se3_log

# ---------------------------------------------------------------------------
# Barron's General and Adaptive Robust Loss
# ---------------------------------------------------------------------------

class BarronRobustLoss(nn.Module):
    """Barron's robust loss function (CVPR 2019).

    rho(x, alpha, c) where:
      alpha = 2  -> L2 loss
      alpha = 1  -> Charbonnier (pseudo-Huber)
      alpha = 0  -> Cauchy (Lorentzian)
      alpha = -2 -> Geman-McClure
      alpha = -inf -> Welsch

    The loss is: (|alpha - 2| / alpha) * ((x/c)^2 / |alpha - 2| + 1)^(alpha/2) - 1)
    """

    def __init__(self, alpha: float = 0.0, scale: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute robust loss.

        Args:
            x: (...) squared residuals (non-negative).

        Returns:
            loss: (...) per-element loss values.
        """
        scaled = x / (self.scale ** 2 + 1e-8)
        alpha = self.alpha

        if abs(alpha - 2.0) < 1e-6:
            # L2
            return 0.5 * scaled
        elif abs(alpha) < 1e-6:
            # Cauchy / Lorentzian
            return torch.log1p(0.5 * scaled)
        elif abs(alpha + 2.0) < 1e-6:
            # Geman-McClure
            return 2.0 * scaled / (scaled + 4.0)
        else:
            # General case
            z = scaled / abs(alpha - 2.0) + 1.0
            t = z.clamp(min=1e-8) ** (alpha / 2.0)
            return (abs(alpha - 2.0) / alpha) * (t - 1.0)


# ---------------------------------------------------------------------------
# Photometric Cost with Uncertainty Weighting
# ---------------------------------------------------------------------------

class PhotometricCost(nn.Module):
    """Uncertainty-weighted feature-metric photometric cost (Eq.7).

    C_photo = sum_j rho_huber(w(j) * ||r_j||^2)

    where w(j) is the learned uncertainty weight and r_j is the feature
    residual between query and warped reference at anchor j.
    """

    def __init__(self, huber_delta: float = 1.0):
        super().__init__()
        self.huber_delta = huber_delta

    def huber(self, x: torch.Tensor) -> torch.Tensor:
        """Huber loss: quadratic for |x| < delta, linear otherwise."""
        delta = self.huber_delta
        abs_x = x.abs()
        quadratic = 0.5 * x ** 2
        linear = delta * (abs_x - 0.5 * delta)
        return torch.where(abs_x <= delta, quadratic, linear)

    def forward(
        self,
        query_feat: torch.Tensor,
        ref_feat: torch.Tensor,
        uncertainty: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute photometric cost.

        Args:
            query_feat: (B, C, H, W) query features.
            ref_feat: (B, C, H, W) reference features (warped to query frame).
            uncertainty: (B, 1, H, W) or None. Learned uncertainty weights.

        Returns:
            cost: scalar, mean photometric cost.
        """
        residual = query_feat - ref_feat  # (B, C, H, W)
        sq_residual = (residual ** 2).sum(dim=1, keepdim=True)  # (B, 1, H, W)

        if uncertainty is not None:
            # Weight by inverse uncertainty (higher uncertainty -> lower weight)
            weight = 1.0 / (uncertainty + 1e-6)
            weighted = weight * sq_residual
        else:
            weighted = sq_residual

        return self.huber(weighted).mean()


# ---------------------------------------------------------------------------
# SE(3) Motion Regularization
# ---------------------------------------------------------------------------

class SE3MotionRegularization(nn.Module):
    """SE(3) geodesic motion regularization (Eq.10).

    C_motion = lambda * ||log(T_pred^{-1} T_est)||^2

    Penalizes estimated poses that deviate significantly from the
    Kalman-predicted pose in the SE(3) Lie algebra.
    """

    def __init__(self, lam: float = 1.0):
        super().__init__()
        self.lam = lam

    def forward(
        self,
        T_est: torch.Tensor,
        T_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Compute motion regularization.

        Args:
            T_est: (B, 4, 4) estimated pose.
            T_pred: (B, 4, 4) predicted pose (from Kalman filter).

        Returns:
            cost: scalar, motion regularization cost.
        """
        T_pred_inv = torch.linalg.inv(T_pred)
        T_rel = T_pred_inv @ T_est
        xi = se3_log(T_rel)  # (B, 6)
        geodesic_sq = (xi ** 2).sum(dim=-1)  # (B,)
        return self.lam * geodesic_sq.mean()


# ---------------------------------------------------------------------------
# Reprojection Loss (Training)
# ---------------------------------------------------------------------------

class ReprojectionLoss(nn.Module):
    """Barron robust loss on reprojection error (Eq.3).

    L = sum_j rho_B(||p_j^q - p_tilde_j^q||^2)

    Used for training the feature network. p_j^q are GT 2D projections,
    p_tilde_j^q are estimated projections from the predicted pose.
    """

    def __init__(self, alpha: float = 0.0, scale: float = 1.0):
        super().__init__()
        self.barron = BarronRobustLoss(alpha=alpha, scale=scale)

    def forward(
        self,
        pts_pred: torch.Tensor,
        pts_gt: torch.Tensor,
    ) -> torch.Tensor:
        """Compute reprojection loss.

        Args:
            pts_pred: (B, N, 2) predicted 2D projections.
            pts_gt: (B, N, 2) ground truth 2D projections.

        Returns:
            loss: scalar, mean reprojection loss.
        """
        sq_error = ((pts_pred - pts_gt) ** 2).sum(dim=-1)  # (B, N)
        return self.barron(sq_error).mean()


# ---------------------------------------------------------------------------
# Combined PiLoT Loss
# ---------------------------------------------------------------------------

class PiLoTLoss(nn.Module):
    """Combined PiLoT training loss.

    L_total = L_reproj + lambda_photo * L_photo + lambda_motion * L_motion

    During training:
    - L_reproj: Barron robust on 2D reprojection error (primary)
    - L_photo: uncertainty-weighted feature residual
    - L_motion: SE(3) geodesic regularization
    """

    def __init__(
        self,
        barron_alpha: float = 0.0,
        barron_scale: float = 1.0,
        huber_delta: float = 1.0,
        motion_lambda: float = 1.0,
        photo_lambda: float = 0.1,
    ):
        super().__init__()
        self.reproj_loss = ReprojectionLoss(alpha=barron_alpha, scale=barron_scale)
        self.photo_cost = PhotometricCost(huber_delta=huber_delta)
        self.motion_reg = SE3MotionRegularization(lam=motion_lambda)
        self.photo_lambda = photo_lambda

    def forward(
        self,
        pts_pred: torch.Tensor,
        pts_gt: torch.Tensor,
        query_feat: torch.Tensor | None = None,
        ref_feat: torch.Tensor | None = None,
        uncertainty: torch.Tensor | None = None,
        T_est: torch.Tensor | None = None,
        T_pred: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute combined loss.

        Returns:
            Dictionary with 'total', 'reproj', 'photo', 'motion' keys.
        """
        losses = {}

        # Reprojection (always)
        losses["reproj"] = self.reproj_loss(pts_pred, pts_gt)
        total = losses["reproj"]

        # Photometric (if features provided)
        if query_feat is not None and ref_feat is not None:
            losses["photo"] = self.photo_cost(query_feat, ref_feat, uncertainty)
            total = total + self.photo_lambda * losses["photo"]
        else:
            losses["photo"] = torch.tensor(0.0, device=pts_pred.device)

        # Motion regularization (if poses provided)
        if T_est is not None and T_pred is not None:
            losses["motion"] = self.motion_reg(T_est, T_pred)
            total = total + losses["motion"]
        else:
            losses["motion"] = torch.tensor(0.0, device=pts_pred.device)

        losses["total"] = total
        return losses
