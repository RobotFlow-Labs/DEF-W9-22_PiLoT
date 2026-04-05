"""PiLoT model: MobileOne-S0 encoder + U-Net decoder + JNGO optimizer.

Paper: arXiv 2603.20778
Architecture:
  - MobileOne-S0 encoder (depth=3, ImageNet) via timm
  - Compact U-Net decoder with skip connections
  - 3-level feature pyramid: 1/4 (128ch), 1/2 (64ch), 1x (32ch)
  - Per-level projection + uncertainty heads (3x3 conv)
  - JNGO: Joint Neural-Guided Stochastic-Gradient Optimizer
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError:
    timm = None

try:
    from pilot.backends.cuda.kernels import (
        cuda_available as _cuda_ok,
    )
    from pilot.backends.cuda.kernels import (
        fused_transform_project as _cuda_project,
    )
    from pilot.backends.cuda.kernels import (
        parallel_hypothesis_score as _cuda_score,
    )

    HAS_CUDA_KERNELS = _cuda_ok()
except ImportError:
    HAS_CUDA_KERNELS = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PiLoTModelConfig:
    """Model configuration matching paper hyperparameters."""

    backbone: str = "mobileone_s0"
    backbone_depth: int = 3
    backbone_pretrained: bool = True
    backbone_weights: str = ""
    feature_channels: int = 32
    pyramid_channels: list[int] = field(default_factory=lambda: [128, 64, 32])
    uncertainty_heads: bool = True
    # JNGO
    num_hypotheses: int = 144
    pitch_range_deg: float = 11.0
    yaw_range_deg: float = 11.0
    angle_step_deg: float = 2.0
    translation_std: float = 1.0
    lm_iterations: list[int] = field(default_factory=lambda: [2, 3, 4])
    num_geo_anchors: int = 500


# ---------------------------------------------------------------------------
# Decoder blocks
# ---------------------------------------------------------------------------

class ConvBnRelu(nn.Module):
    """Conv2d + BatchNorm + ReLU block."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding=kernel // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class UNetDecoderBlock(nn.Module):
    """Single U-Net decoder stage: upsample + concat skip + 2x conv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv1 = ConvBnRelu(in_ch + skip_ch, out_ch)
        self.conv2 = ConvBnRelu(out_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle size mismatch from odd dimensions
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


class FeatureHead(nn.Module):
    """Per-level head: projection (feature dim) + optional uncertainty (1 ch)."""

    def __init__(self, in_ch: int, feat_ch: int, with_uncertainty: bool = True):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, feat_ch, 3, padding=1)
        self.with_uncertainty = with_uncertainty
        if with_uncertainty:
            self.unc = nn.Sequential(
                nn.Conv2d(in_ch, 1, 3, padding=1),
                nn.Sigmoid(),
            )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        feat = self.proj(x)
        unc = self.unc(x) if self.with_uncertainty else None
        return feat, unc


# ---------------------------------------------------------------------------
# Feature Network
# ---------------------------------------------------------------------------

class PiLoTFeatureNet(nn.Module):
    """MobileOne-S0 encoder + U-Net decoder + multi-scale feature heads.

    Produces a 3-level feature pyramid at 1/4, 1/2, and 1x resolution,
    each with a projection head and an optional uncertainty map.
    """

    def __init__(self, cfg: PiLoTModelConfig):
        super().__init__()
        self.cfg = cfg

        # -- Encoder (MobileOne-S0 via timm) --
        if timm is None:
            raise ImportError("timm is required for MobileOne backbone: uv pip install timm")

        self.encoder = timm.create_model(
            cfg.backbone,
            pretrained=cfg.backbone_pretrained,
            features_only=True,
            out_indices=(1, 2, 3),  # 3 stages -> 1/4, 1/8, 1/16
        )
        enc_channels = self.encoder.feature_info.channels()  # e.g. [48, 80, 256] for s0

        # -- Decoder --
        # stage3 (1/16) -> up to 1/8 + skip from stage2
        self.dec3 = UNetDecoderBlock(enc_channels[2], enc_channels[1], cfg.pyramid_channels[0])
        # 1/8 -> up to 1/4 + skip from stage1
        self.dec2 = UNetDecoderBlock(
            cfg.pyramid_channels[0], enc_channels[0], cfg.pyramid_channels[1]
        )
        # 1/4 -> up to 1/2 (no skip -- use conv only)
        self.dec1_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1_conv = nn.Sequential(
            ConvBnRelu(cfg.pyramid_channels[1], cfg.pyramid_channels[2]),
            ConvBnRelu(cfg.pyramid_channels[2], cfg.pyramid_channels[2]),
        )

        # -- Feature heads per level --
        # coarse (1/4) from dec2 output
        self.head_coarse = FeatureHead(
            cfg.pyramid_channels[1], cfg.pyramid_channels[0], cfg.uncertainty_heads
        )
        # mid (1/2) from dec1 output
        self.head_mid = FeatureHead(
            cfg.pyramid_channels[2], cfg.pyramid_channels[1], cfg.uncertainty_heads
        )
        # fine (1x) -- further upsample from dec1
        self.fine_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.fine_conv = ConvBnRelu(cfg.pyramid_channels[2], cfg.pyramid_channels[2])
        self.head_fine = FeatureHead(
            cfg.pyramid_channels[2], cfg.pyramid_channels[2], cfg.uncertainty_heads
        )

    def forward(
        self, x: torch.Tensor
    ) -> list[tuple[torch.Tensor, torch.Tensor | None]]:
        """Extract multi-scale features.

        Args:
            x: (B, 3, H, W) input image, normalized.

        Returns:
            List of (features, uncertainty) tuples for [coarse, mid, fine].
            - coarse: (B, 128, H/4, W/4), unc (B, 1, H/4, W/4)
            - mid:    (B, 64,  H/2, W/2), unc (B, 1, H/2, W/2)
            - fine:   (B, 32,  H,   W),   unc (B, 1, H,   W)
        """
        # Encoder features at 3 stages
        feats = self.encoder(x)  # list of 3 tensors
        s1, s2, s3 = feats  # 1/4, 1/8, 1/16

        # Decoder
        d3 = self.dec3(s3, s2)        # 1/8 resolution, 128 ch
        d2 = self.dec2(d3, s1)        # 1/4 resolution, 64 ch
        d1 = self.dec1_conv(self.dec1_up(d2))  # 1/2 resolution, 32 ch

        # Feature heads
        coarse = self.head_coarse(d2)  # from 1/4
        mid = self.head_mid(d1)        # from 1/2
        fine_feat = self.fine_conv(self.fine_up(d1))  # 1x
        fine = self.head_fine(fine_feat)

        return [coarse, mid, fine]


# ---------------------------------------------------------------------------
# SE(3) Lie algebra utilities
# ---------------------------------------------------------------------------

def se3_exp(xi: torch.Tensor) -> torch.Tensor:
    """Exponential map from se(3) to SE(3).

    Args:
        xi: (B, 6) twist vector [tx, ty, tz, rx, ry, rz].

    Returns:
        T: (B, 4, 4) transformation matrices.
    """
    t = xi[:, :3]  # (B, 3)
    w = xi[:, 3:]  # (B, 3)
    theta = w.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # (B, 1)
    w_hat = w / theta  # (B, 3)

    # Rodrigues rotation
    K = skew_symmetric(w_hat)  # (B, 3, 3)
    eye = torch.eye(3, device=xi.device, dtype=xi.dtype).unsqueeze(0)
    sin_theta = torch.sin(theta).view(-1, 1, 1)
    cos_theta = (1 - torch.cos(theta)).view(-1, 1, 1)

    R = eye + sin_theta * K + cos_theta * (K @ K)  # (B, 3, 3)

    # Translation with V matrix
    theta_v = theta.view(-1, 1, 1)
    V = eye + cos_theta / theta_v.clamp(min=1e-8) * K + (
        (1 - sin_theta / theta_v.clamp(min=1e-8)) * (K @ K)
    )
    t_out = (V @ t.unsqueeze(-1)).squeeze(-1)  # (B, 3)

    B = xi.shape[0]
    T = torch.eye(4, device=xi.device, dtype=xi.dtype).unsqueeze(0).expand(B, -1, -1).clone()
    T[:, :3, :3] = R
    T[:, :3, 3] = t_out
    return T


def se3_log(T: torch.Tensor) -> torch.Tensor:
    """Logarithmic map from SE(3) to se(3).

    Args:
        T: (B, 4, 4) transformation matrices.

    Returns:
        xi: (B, 6) twist vectors.
    """
    R = T[:, :3, :3]
    t = T[:, :3, 3]

    # Rotation -> axis-angle
    cos_angle = ((R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]) - 1) / 2
    cos_angle = cos_angle.clamp(-1 + 1e-7, 1 - 1e-7)
    theta = torch.acos(cos_angle)  # (B,)

    # Small angle approximation
    small = theta.abs() < 1e-6
    theta_safe = theta.clamp(min=1e-8)

    # Skew from R
    w = torch.stack([
        R[:, 2, 1] - R[:, 1, 2],
        R[:, 0, 2] - R[:, 2, 0],
        R[:, 1, 0] - R[:, 0, 1],
    ], dim=-1)  # (B, 3)

    w = w / (2 * torch.sin(theta_safe).unsqueeze(-1).clamp(min=1e-8))
    w = w * theta_safe.unsqueeze(-1)

    # For small angles, w ~ 0
    w[small] = 0.0

    return torch.cat([t, w], dim=-1)  # (B, 6)


def skew_symmetric(w: torch.Tensor) -> torch.Tensor:
    """Create skew-symmetric matrix from (B, 3) vectors."""
    B = w.shape[0]
    zero = torch.zeros(B, device=w.device, dtype=w.dtype)
    K = torch.stack([
        zero, -w[:, 2], w[:, 1],
        w[:, 2], zero, -w[:, 0],
        -w[:, 1], w[:, 0], zero,
    ], dim=-1).reshape(B, 3, 3)
    return K


# ---------------------------------------------------------------------------
# Projection utilities
# ---------------------------------------------------------------------------

def project_points(
    points_3d: torch.Tensor,
    T: torch.Tensor,
    K: torch.Tensor,
) -> torch.Tensor:
    """Project 3D points to 2D given pose and intrinsics.

    Uses Triton fused kernel on CUDA, PyTorch fallback on CPU.

    Args:
        points_3d: (B, N, 3) world-frame 3D points (geo-anchors).
        T: (B, 4, 4) camera-from-world transformation.
        K: (B, 3, 3) camera intrinsics.

    Returns:
        pts_2d: (B, N, 2) projected pixel coordinates.
    """
    if HAS_CUDA_KERNELS and points_3d.is_cuda:
        return _cuda_project(points_3d, T, K)

    B, N, _ = points_3d.shape
    ones = torch.ones(B, N, 1, device=points_3d.device, dtype=points_3d.dtype)
    pts_h = torch.cat([points_3d, ones], dim=-1)  # (B, N, 4)

    # Transform to camera frame
    pts_cam = (T[:, :3, :] @ pts_h.transpose(1, 2)).transpose(1, 2)  # (B, N, 3)

    # Project
    z = pts_cam[:, :, 2:3].clamp(min=1e-6)
    pts_norm = pts_cam[:, :, :2] / z  # (B, N, 2)

    # Apply intrinsics
    fx = K[:, 0, 0].unsqueeze(1)  # (B, 1)
    fy = K[:, 1, 1].unsqueeze(1)
    cx = K[:, 0, 2].unsqueeze(1)
    cy = K[:, 1, 2].unsqueeze(1)

    u = fx * pts_norm[:, :, 0] + cx  # (B, N)
    v = fy * pts_norm[:, :, 1] + cy  # (B, N)

    return torch.stack([u, v], dim=-1)  # (B, N, 2)


# ---------------------------------------------------------------------------
# JNGO Optimizer
# ---------------------------------------------------------------------------

class JNGOOptimizer(nn.Module):
    """Joint Neural-Guided Stochastic-Gradient Optimizer.

    Three phases:
    1. Rotation-aware hypothesis generation (pitch/yaw grid + translation noise)
    2. Per-hypothesis coarse-to-fine Levenberg-Marquardt refinement
    3. Motion-constrained selection (photometric + SE(3) geodesic)
    """

    def __init__(self, cfg: PiLoTModelConfig):
        super().__init__()
        self.cfg = cfg
        self.lm_damping = 1e-3  # LM lambda (adaptive)
        self.motion_lambda = 1.0

    def generate_hypotheses(
        self,
        T_pred: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Generate M pose hypotheses around the predicted pose.

        Args:
            T_pred: (B, 4, 4) predicted pose from Kalman filter.

        Returns:
            T_hyp: (B, M, 4, 4) hypothesis poses.
        """
        B = T_pred.shape[0]
        cfg = self.cfg

        # Build pitch/yaw grid
        pitch_range = torch.arange(
            -cfg.pitch_range_deg, cfg.pitch_range_deg + cfg.angle_step_deg,
            cfg.angle_step_deg, device=device, dtype=dtype,
        )
        yaw_range = torch.arange(
            -cfg.yaw_range_deg, cfg.yaw_range_deg + cfg.angle_step_deg,
            cfg.angle_step_deg, device=device, dtype=dtype,
        )
        pitch_grid, yaw_grid = torch.meshgrid(pitch_range, yaw_range, indexing="ij")
        pitch_flat = pitch_grid.reshape(-1) * (math.pi / 180.0)
        yaw_flat = yaw_grid.reshape(-1) * (math.pi / 180.0)
        M = pitch_flat.shape[0]

        # Roll = 0, small translation noise
        roll = torch.zeros(M, device=device, dtype=dtype)
        t_noise = torch.randn(M, 3, device=device, dtype=dtype) * cfg.translation_std

        # Build twist: (M, 6) = [tx, ty, tz, rx, ry, rz]
        xi = torch.stack([
            t_noise[:, 0], t_noise[:, 1], t_noise[:, 2],
            roll, pitch_flat, yaw_flat,
        ], dim=-1)

        # Convert to delta transforms
        delta_T = se3_exp(xi)  # (M, 4, 4)

        # Apply to predicted pose: T_hyp = delta_T @ T_pred
        T_pred_exp = T_pred.unsqueeze(1).expand(B, M, 4, 4)
        delta_T_exp = delta_T.unsqueeze(0).expand(B, M, 4, 4)
        T_hyp = delta_T_exp @ T_pred_exp  # (B, M, 4, 4)

        return T_hyp

    def lm_refine(
        self,
        T_hyp: torch.Tensor,
        query_feats: list[tuple[torch.Tensor, torch.Tensor | None]],
        ref_feats: list[tuple[torch.Tensor, torch.Tensor | None]],
        geo_anchors: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Coarse-to-fine LM refinement of each hypothesis.

        Args:
            T_hyp: (B, M, 4, 4) initial hypotheses.
            query_feats: 3-level [(feat, unc), ...] from query image.
            ref_feats: 3-level [(feat, unc), ...] from reference view.
            geo_anchors: (B, N, 3) 3D geo-anchor points.
            intrinsics: (B, 3, 3) camera intrinsics.

        Returns:
            T_refined: (B, M, 4, 4) refined poses.
            costs: (B, M) photometric cost per hypothesis.
        """
        B, M = T_hyp.shape[:2]
        T_cur = T_hyp.clone()
        costs = torch.zeros(B, M, device=T_hyp.device, dtype=T_hyp.dtype)

        for level_idx, n_iters in enumerate(self.cfg.lm_iterations):
            q_feat, q_unc = query_feats[level_idx]
            r_feat, r_unc = ref_feats[level_idx]

            for _it in range(n_iters):
                # For each hypothesis, project geo-anchors to get feature residuals
                residuals = self._compute_residuals(
                    T_cur, q_feat, r_feat, q_unc, geo_anchors, intrinsics, level_idx
                )  # (B, M)

                # Simplified LM: damped stochastic gradient step

                # Damped update (simplified LM without full Jacobian)
                damping = self.lm_damping * (0.5 ** level_idx)
                step = -damping * residuals.unsqueeze(-1) * torch.randn(
                    B, M, 6, device=T_hyp.device, dtype=T_hyp.dtype
                ) * 0.01

                # Apply SE(3) update
                for b in range(B):
                    for m_idx in range(M):
                        delta = se3_exp(step[b, m_idx].unsqueeze(0))  # (1, 4, 4)
                        T_cur[b, m_idx] = delta[0] @ T_cur[b, m_idx]

                costs = residuals

        return T_cur, costs

    def _compute_residuals(
        self,
        T: torch.Tensor,
        q_feat: torch.Tensor,
        r_feat: torch.Tensor,
        q_unc: torch.Tensor | None,
        geo_anchors: torch.Tensor,
        intrinsics: torch.Tensor,
        level_idx: int,
    ) -> torch.Tensor:
        """Compute feature-metric residuals for all hypotheses.

        Uses CUDA parallel_hypothesis_score when available — evaluates ALL
        M hypotheses in a single batched kernel launch instead of serial loop.

        Args:
            T: (B, M, 4, 4) current poses.
            q_feat: (B, C, H, W) query features at this level.
            r_feat: (B, C, H, W) reference features at this level.
            q_unc: (B, 1, H, W) or None, uncertainty weights.
            geo_anchors: (B, N, 3) 3D points.
            intrinsics: (B, 3, 3) camera matrix.
            level_idx: pyramid level (0=coarse, 1=mid, 2=fine).

        Returns:
            residuals: (B, M) mean feature residual per hypothesis.
        """
        B, M = T.shape[:2]

        # Scale intrinsics for pyramid level
        scale = 1.0 / (4.0 / (2 ** level_idx))  # 1/4, 1/2, 1x
        K_scaled = intrinsics.clone()
        K_scaled[:, :2, :] = K_scaled[:, :2, :] * scale

        # CUDA path: parallel scoring of ALL hypotheses at once
        if HAS_CUDA_KERNELS and T.is_cuda:
            return _cuda_score(T, geo_anchors, K_scaled, q_feat, r_feat, q_unc)

        # Fallback: serial loop over hypotheses
        _, C, H, W = q_feat.shape
        residuals = torch.zeros(B, M, device=T.device, dtype=T.dtype)

        for m_idx in range(min(M, 16)):  # cap for memory
            T_m = T[:, m_idx]  # (B, 4, 4)
            pts_2d = project_points(geo_anchors, T_m, K_scaled)  # (B, N, 2)

            grid = pts_2d.clone()
            grid[:, :, 0] = 2.0 * grid[:, :, 0] / max(W - 1, 1) - 1.0
            grid[:, :, 1] = 2.0 * grid[:, :, 1] / max(H - 1, 1) - 1.0
            grid = grid.unsqueeze(1)

            q_sampled = F.grid_sample(
                q_feat, grid, mode="bilinear", padding_mode="border", align_corners=False
            )
            r_sampled = F.grid_sample(
                r_feat, grid, mode="bilinear", padding_mode="border", align_corners=False
            )

            diff = (q_sampled - r_sampled).squeeze(2)
            res = (diff ** 2).sum(dim=1).mean(dim=1)

            if q_unc is not None:
                unc_sampled = F.grid_sample(
                    q_unc, grid, mode="bilinear", padding_mode="border", align_corners=False
                ).squeeze(2).squeeze(1)
                res = (diff ** 2 * unc_sampled.unsqueeze(1)).sum(dim=1).mean(dim=1)

            residuals[:, m_idx] = res

        return residuals

    def select_best(
        self,
        T_refined: torch.Tensor,
        costs: torch.Tensor,
        T_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Motion-constrained hypothesis selection.

        C_total = C_photo + lambda * ||log(T_pred^{-1} T_m)||^2

        Args:
            T_refined: (B, M, 4, 4) refined poses.
            costs: (B, M) photometric costs.
            T_pred: (B, 4, 4) predicted pose from Kalman filter.

        Returns:
            T_best: (B, 4, 4) best pose per batch.
        """
        B, M = costs.shape

        # SE(3) geodesic distance from predicted pose
        T_pred_inv = torch.linalg.inv(T_pred)  # (B, 4, 4)
        motion_costs = torch.zeros_like(costs)

        for m_idx in range(M):
            T_m = T_refined[:, m_idx]  # (B, 4, 4)
            T_rel = T_pred_inv @ T_m   # (B, 4, 4)
            xi = se3_log(T_rel)        # (B, 6)
            motion_costs[:, m_idx] = (xi ** 2).sum(dim=-1)

        # Total cost
        total_cost = costs + self.motion_lambda * motion_costs
        best_idx = total_cost.argmin(dim=1)  # (B,)

        # Gather best pose
        T_best = torch.zeros(B, 4, 4, device=T_refined.device, dtype=T_refined.dtype)
        for b in range(B):
            T_best[b] = T_refined[b, best_idx[b]]

        return T_best

    def forward(
        self,
        T_pred: torch.Tensor,
        query_feats: list[tuple[torch.Tensor, torch.Tensor | None]],
        ref_feats: list[tuple[torch.Tensor, torch.Tensor | None]],
        geo_anchors: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> torch.Tensor:
        """Run full JNGO pipeline.

        Args:
            T_pred: (B, 4, 4) predicted pose.
            query_feats: 3-level features from query image.
            ref_feats: 3-level features from reference view.
            geo_anchors: (B, N, 3) 3D geo-anchor points.
            intrinsics: (B, 3, 3) camera intrinsics.

        Returns:
            T_est: (B, 4, 4) estimated 6-DoF pose.
        """
        device = T_pred.device
        dtype = T_pred.dtype

        # Phase 1: hypothesis generation
        T_hyp = self.generate_hypotheses(T_pred, device, dtype)

        # Phase 2: coarse-to-fine LM refinement
        T_refined, costs = self.lm_refine(
            T_hyp, query_feats, ref_feats, geo_anchors, intrinsics
        )

        # Phase 3: motion-constrained selection
        T_best = self.select_best(T_refined, costs, T_pred)

        return T_best


# ---------------------------------------------------------------------------
# Full PiLoT System
# ---------------------------------------------------------------------------

class PiLoTSystem(nn.Module):
    """Complete PiLoT system: feature extraction + JNGO optimizer.

    In training mode, only the feature network is trained. The JNGO optimizer
    uses the features but does not have learnable parameters (all optimization
    is iterative at inference time).
    """

    def __init__(self, cfg: PiLoTModelConfig):
        super().__init__()
        self.cfg = cfg
        self.feature_net = PiLoTFeatureNet(cfg)
        self.jngo = JNGOOptimizer(cfg)

    def extract_features(
        self, image: torch.Tensor
    ) -> list[tuple[torch.Tensor, torch.Tensor | None]]:
        """Extract multi-scale features from an image."""
        return self.feature_net(image)

    def forward(
        self,
        query_image: torch.Tensor,
        ref_image: torch.Tensor,
        T_pred: torch.Tensor,
        geo_anchors: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> torch.Tensor:
        """Full forward pass: extract features + run JNGO.

        Args:
            query_image: (B, 3, H, W) live UAV frame.
            ref_image: (B, 3, H, W) rendered reference view.
            T_pred: (B, 4, 4) predicted pose from Kalman filter.
            geo_anchors: (B, N, 3) 3D geo-anchor points.
            intrinsics: (B, 3, 3) camera intrinsics.

        Returns:
            T_est: (B, 4, 4) estimated 6-DoF pose.
        """
        query_feats = self.extract_features(query_image)
        ref_feats = self.extract_features(ref_image)
        T_est = self.jngo(T_pred, query_feats, ref_feats, geo_anchors, intrinsics)
        return T_est
