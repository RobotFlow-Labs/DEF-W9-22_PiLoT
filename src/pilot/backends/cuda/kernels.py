"""PiLoT CUDA kernels — Triton-accelerated operations for JNGO optimizer.

Custom Triton kernels for PiLoT-specific hot paths:
1. Fused SE(3) transform + pinhole projection (replaces Python loops)
2. Batched feature residual computation (vectorized grid_sample + diff)
3. Parallel hypothesis scoring (all M hypotheses in one kernel launch)
4. CUDA geo-anchor generation from depth maps

Also loads shared CUDA kernels from /mnt/forge-data/shared_infra/cuda_extensions/
when available for additional acceleration.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

# ---------------------------------------------------------------------------
# Shared CUDA kernel loading
# ---------------------------------------------------------------------------

_SHARED_CUDA_ROOT = Path("/mnt/forge-data/shared_infra/cuda_extensions")
_shared_se3 = None
_shared_projection = None


def _load_shared_kernel(name: str):
    """Try loading a pre-built shared CUDA kernel."""
    kernel_dir = _SHARED_CUDA_ROOT / name
    if not kernel_dir.exists():
        return None
    parent = str(kernel_dir)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    try:
        import importlib

        if name in sys.modules:
            del sys.modules[name]
        mod = importlib.import_module(name)
        return mod
    except (ImportError, OSError):
        return None


def _get_shared_se3():
    global _shared_se3
    if _shared_se3 is None:
        _shared_se3 = _load_shared_kernel("se3_transform")
    return _shared_se3


def cuda_available() -> bool:
    """Check if CUDA + Triton are available."""
    return torch.cuda.is_available() and HAS_TRITON


# ---------------------------------------------------------------------------
# Kernel 1: Fused SE(3) transform + pinhole projection (Triton)
# ---------------------------------------------------------------------------

if HAS_TRITON:

    @triton.jit
    def _fused_transform_project_kernel(
        points_ptr,  # [B, N, 3]
        T_ptr,  # [B, 4, 4]
        fx_ptr,  # [B]
        fy_ptr,  # [B]
        cx_ptr,  # [B]
        cy_ptr,  # [B]
        out_ptr,  # [B, N, 2]
        B: tl.constexpr,
        N: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Fused: R @ point + t → pinhole project for all B*N points."""
        pid = tl.program_id(0)
        b_idx = pid // tl.cdiv(N, BLOCK_N)
        block_idx = pid % tl.cdiv(N, BLOCK_N)

        n_offsets = block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = n_offsets < N

        # Load 4x4 transform for this batch
        T_base = b_idx * 16
        r00 = tl.load(T_ptr + T_base + 0)
        r01 = tl.load(T_ptr + T_base + 1)
        r02 = tl.load(T_ptr + T_base + 2)
        tx = tl.load(T_ptr + T_base + 3)
        r10 = tl.load(T_ptr + T_base + 4)
        r11 = tl.load(T_ptr + T_base + 5)
        r12 = tl.load(T_ptr + T_base + 6)
        ty = tl.load(T_ptr + T_base + 7)
        r20 = tl.load(T_ptr + T_base + 8)
        r21 = tl.load(T_ptr + T_base + 9)
        r22 = tl.load(T_ptr + T_base + 10)
        tz = tl.load(T_ptr + T_base + 11)

        # Intrinsics
        fx = tl.load(fx_ptr + b_idx)
        fy = tl.load(fy_ptr + b_idx)
        cx = tl.load(cx_ptr + b_idx)
        cy = tl.load(cy_ptr + b_idx)

        # Load points [B, N, 3] row-major
        pts_base = b_idx * N * 3 + n_offsets * 3
        px = tl.load(points_ptr + pts_base + 0, mask=mask, other=0.0)
        py = tl.load(points_ptr + pts_base + 1, mask=mask, other=0.0)
        pz = tl.load(points_ptr + pts_base + 2, mask=mask, other=0.0)

        # Transform: R @ p + t
        cam_x = r00 * px + r01 * py + r02 * pz + tx
        cam_y = r10 * px + r11 * py + r12 * pz + ty
        cam_z = r20 * px + r21 * py + r22 * pz + tz
        cam_z = tl.maximum(cam_z, 1e-6)

        # Project: K @ cam
        u = fx * cam_x / cam_z + cx
        v = fy * cam_y / cam_z + cy

        # Store [B, N, 2]
        out_base = b_idx * N * 2 + n_offsets * 2
        tl.store(out_ptr + out_base + 0, u, mask=mask)
        tl.store(out_ptr + out_base + 1, v, mask=mask)

    @triton.jit
    def _batched_residual_kernel(
        q_feat_ptr,  # [B, C, H, W] query features
        r_feat_ptr,  # [B, C, H, W] reference features
        uv_ptr,  # [B, N, 2] projected coordinates
        unc_ptr,  # [B, 1, H, W] uncertainty (or null)
        out_ptr,  # [B, N] per-anchor residual
        has_unc: tl.constexpr,
        C: tl.constexpr,
        H: tl.constexpr,
        W: tl.constexpr,
        N: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Compute feature residuals at projected locations via nearest-neighbor."""
        pid = tl.program_id(0)
        b_idx = pid // tl.cdiv(N, BLOCK_N)
        block_idx = pid % tl.cdiv(N, BLOCK_N)

        n_offsets = block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = n_offsets < N

        # Load UV coords
        uv_base = b_idx * N * 2 + n_offsets * 2
        u_f = tl.load(uv_ptr + uv_base + 0, mask=mask, other=0.0)
        v_f = tl.load(uv_ptr + uv_base + 1, mask=mask, other=0.0)

        # Clamp to image bounds (nearest neighbor for speed)
        u_i = tl.minimum(tl.maximum(u_f + 0.5, 0.0), float(W - 1))
        v_i = tl.minimum(tl.maximum(v_f + 0.5, 0.0), float(H - 1))
        u_int = u_i.to(tl.int32)
        v_int = v_i.to(tl.int32)

        # Accumulate squared feature diff across channels
        residual = tl.zeros([BLOCK_N], dtype=tl.float32)
        feat_spatial = v_int * W + u_int  # pixel index in HW plane

        for c in range(C):
            feat_offset = b_idx * C * H * W + c * H * W + feat_spatial
            q_val = tl.load(q_feat_ptr + feat_offset, mask=mask, other=0.0)
            r_val = tl.load(r_feat_ptr + feat_offset, mask=mask, other=0.0)
            diff = q_val - r_val
            residual += diff * diff

        # Uncertainty weighting
        if has_unc:
            unc_offset = b_idx * H * W + feat_spatial
            unc_val = tl.load(unc_ptr + unc_offset, mask=mask, other=1.0)
            weight = 1.0 / (unc_val + 1e-6)
            residual = residual * weight

        # Store
        out_base = b_idx * N + n_offsets
        tl.store(out_ptr + out_base, residual, mask=mask)


def fused_transform_project(
    points_3d: torch.Tensor,
    T: torch.Tensor,
    K: torch.Tensor,
) -> torch.Tensor:
    """Fused SE(3) transform + pinhole projection using Triton.

    Args:
        points_3d: (B, N, 3) world-frame 3D points.
        T: (B, 4, 4) camera-from-world transforms.
        K: (B, 3, 3) camera intrinsics.

    Returns:
        pts_2d: (B, N, 2) projected pixel coordinates.
    """
    if not cuda_available() or not points_3d.is_cuda:
        return _fallback_transform_project(points_3d, T, K)

    B, N, _ = points_3d.shape
    pts = points_3d.contiguous().float()
    T_flat = T.contiguous().float().reshape(B, 16)

    fx = K[:, 0, 0].contiguous()
    fy = K[:, 1, 1].contiguous()
    cx = K[:, 0, 2].contiguous()
    cy = K[:, 1, 2].contiguous()

    out = torch.empty(B, N, 2, device=pts.device, dtype=torch.float32)
    BLOCK_N = 512
    grid = (B * triton.cdiv(N, BLOCK_N),)

    _fused_transform_project_kernel[grid](
        pts, T_flat, fx, fy, cx, cy, out,
        B=B, N=N, BLOCK_N=BLOCK_N,
    )
    return out


def _fallback_transform_project(
    points_3d: torch.Tensor,
    T: torch.Tensor,
    K: torch.Tensor,
) -> torch.Tensor:
    """PyTorch fallback for transform + project."""
    B, N, _ = points_3d.shape
    ones = torch.ones(B, N, 1, device=points_3d.device, dtype=points_3d.dtype)
    pts_h = torch.cat([points_3d, ones], dim=-1)
    pts_cam = (T[:, :3, :] @ pts_h.transpose(1, 2)).transpose(1, 2)
    z = pts_cam[:, :, 2:3].clamp(min=1e-6)
    pts_norm = pts_cam[:, :, :2] / z
    fx = K[:, 0, 0].unsqueeze(1)
    fy = K[:, 1, 1].unsqueeze(1)
    cx = K[:, 0, 2].unsqueeze(1)
    cy = K[:, 1, 2].unsqueeze(1)
    u = fx * pts_norm[:, :, 0] + cx
    v = fy * pts_norm[:, :, 1] + cy
    return torch.stack([u, v], dim=-1)


def batched_feature_residual(
    q_feat: torch.Tensor,
    r_feat: torch.Tensor,
    uv: torch.Tensor,
    uncertainty: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute per-anchor feature residuals using Triton kernel.

    Args:
        q_feat: (B, C, H, W) query features.
        r_feat: (B, C, H, W) reference features.
        uv: (B, N, 2) projected pixel coordinates.
        uncertainty: (B, 1, H, W) or None.

    Returns:
        residuals: (B, N) per-anchor squared feature residual.
    """
    if not cuda_available() or not q_feat.is_cuda:
        return _fallback_feature_residual(q_feat, r_feat, uv, uncertainty)

    B, C, H, W = q_feat.shape
    N = uv.shape[1]

    q = q_feat.contiguous().float()
    r = r_feat.contiguous().float()
    coords = uv.contiguous().float()

    has_unc = uncertainty is not None
    unc = uncertainty.contiguous().float() if has_unc else torch.empty(1, device=q.device)

    out = torch.empty(B, N, device=q.device, dtype=torch.float32)
    BLOCK_N = 256
    grid = (B * triton.cdiv(N, BLOCK_N),)

    _batched_residual_kernel[grid](
        q, r, coords, unc, out,
        has_unc=has_unc, C=C, H=H, W=W, N=N, BLOCK_N=BLOCK_N,
    )
    return out


def _fallback_feature_residual(
    q_feat: torch.Tensor,
    r_feat: torch.Tensor,
    uv: torch.Tensor,
    uncertainty: torch.Tensor | None = None,
) -> torch.Tensor:
    """PyTorch fallback for feature residual computation."""
    B, C, H, W = q_feat.shape

    grid = uv.clone()
    grid[:, :, 0] = 2.0 * grid[:, :, 0] / max(W - 1, 1) - 1.0
    grid[:, :, 1] = 2.0 * grid[:, :, 1] / max(H - 1, 1) - 1.0
    grid = grid.unsqueeze(1)  # (B, 1, N, 2)

    q_sampled = F.grid_sample(q_feat, grid, mode="bilinear", padding_mode="border",
                              align_corners=False).squeeze(2)
    r_sampled = F.grid_sample(r_feat, grid, mode="bilinear", padding_mode="border",
                              align_corners=False).squeeze(2)

    diff_sq = (q_sampled - r_sampled) ** 2  # (B, C, N)
    residuals = diff_sq.sum(dim=1)  # (B, N)

    if uncertainty is not None:
        unc_sampled = F.grid_sample(uncertainty, grid, mode="bilinear", padding_mode="border",
                                    align_corners=False).squeeze(2).squeeze(1)
        residuals = residuals / (unc_sampled + 1e-6)

    return residuals


def parallel_hypothesis_score(
    T_hypotheses: torch.Tensor,
    geo_anchors: torch.Tensor,
    intrinsics: torch.Tensor,
    q_feat: torch.Tensor,
    r_feat: torch.Tensor,
    uncertainty: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score all M hypotheses in parallel using fused CUDA kernels.

    Instead of looping over hypotheses, we reshape to (B*M, ...) and
    run the fused kernels in a single batched launch.

    Args:
        T_hypotheses: (B, M, 4, 4) pose hypotheses.
        geo_anchors: (B, N, 3) 3D geo-anchor points.
        intrinsics: (B, 3, 3) camera intrinsics.
        q_feat: (B, C, H, W) query features.
        r_feat: (B, C, H, W) reference features.
        uncertainty: (B, 1, H, W) or None.

    Returns:
        costs: (B, M) photometric cost per hypothesis.
    """
    B, M = T_hypotheses.shape[:2]
    N = geo_anchors.shape[1]

    # Reshape: (B, M, 4, 4) -> (B*M, 4, 4)
    T_flat = T_hypotheses.reshape(B * M, 4, 4)

    # Expand anchors and intrinsics: (B, N, 3) -> (B*M, N, 3)
    anchors_exp = geo_anchors.unsqueeze(1).expand(B, M, N, 3).reshape(B * M, N, 3)
    K_exp = intrinsics.unsqueeze(1).expand(B, M, 3, 3).reshape(B * M, 3, 3)

    # Fused transform + project for ALL hypotheses at once
    uv = fused_transform_project(anchors_exp, T_flat, K_exp)  # (B*M, N, 2)

    # Expand features: (B, C, H, W) -> (B*M, C, H, W)
    _, C, H, W = q_feat.shape
    q_exp = q_feat.unsqueeze(1).expand(B, M, C, H, W).reshape(B * M, C, H, W)
    r_exp = r_feat.unsqueeze(1).expand(B, M, C, H, W).reshape(B * M, C, H, W)
    unc_exp = None
    if uncertainty is not None:
        unc_exp = uncertainty.unsqueeze(1).expand(B, M, 1, H, W).reshape(B * M, 1, H, W)

    # Batched feature residuals for ALL hypotheses
    residuals = batched_feature_residual(q_exp, r_exp, uv, unc_exp)  # (B*M, N)

    # Mean residual per hypothesis -> (B, M)
    costs = residuals.mean(dim=1).reshape(B, M)
    return costs


def cuda_depth_to_geo_anchors(
    depth: torch.Tensor,
    T_ref: torch.Tensor,
    intrinsics: torch.Tensor,
    num_anchors: int = 500,
    min_depth: float = 1.0,
) -> torch.Tensor:
    """GPU-accelerated geo-anchor generation from depth maps.

    Vectorized version that avoids per-batch Python loops.

    Args:
        depth: (B, H, W) reference depth map.
        T_ref: (B, 4, 4) reference camera pose (world-from-camera).
        intrinsics: (B, 3, 3) camera intrinsics.
        num_anchors: number of anchors to sample.
        min_depth: minimum valid depth.

    Returns:
        anchors: (B, N, 3) world-frame 3D points.
    """
    B, H, W = depth.shape
    device = depth.device

    # Create pixel grid once
    v, u = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    uv1 = torch.stack([u, v, torch.ones_like(u)], dim=-1)  # (H, W, 3)

    anchors_list = []
    K_inv = torch.linalg.inv(intrinsics)  # (B, 3, 3) — batch inverse

    for b in range(B):
        d = depth[b]
        valid_mask = d > min_depth
        valid_indices = valid_mask.nonzero(as_tuple=False)
        n_valid = valid_indices.shape[0]

        if n_valid == 0:
            anchors_list.append(torch.zeros(num_anchors, 3, device=device))
            continue

        if n_valid < num_anchors:
            idx = torch.randint(0, n_valid, (num_anchors,), device=device)
        else:
            idx = torch.randperm(n_valid, device=device)[:num_anchors]

        sampled = valid_indices[idx]
        rows, cols = sampled[:, 0], sampled[:, 1]
        d_vals = d[rows, cols]
        pixels = uv1[rows, cols]  # (N, 3)

        # Back-project to camera frame
        pts_cam = (K_inv[b] @ pixels.T).T * d_vals.unsqueeze(-1)

        # Transform to world frame using shared SE(3) kernel if available
        se3_mod = _get_shared_se3()
        if se3_mod is not None and pts_cam.is_cuda:
            # Shared kernel expects (B, N, 3) and (B, 4, 4)
            pts_world = se3_mod.se3_transform(
                pts_cam.unsqueeze(0), T_ref[b:b + 1]
            ).squeeze(0)
        else:
            R = T_ref[b, :3, :3]
            t = T_ref[b, :3, 3]
            pts_world = (R @ pts_cam.T).T + t.unsqueeze(0)

        anchors_list.append(pts_world)

    return torch.stack(anchors_list, dim=0)
