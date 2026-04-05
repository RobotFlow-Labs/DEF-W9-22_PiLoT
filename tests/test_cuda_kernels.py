"""Tests for PiLoT CUDA/Triton kernels."""

from __future__ import annotations

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestCUDAKernels:
    """Test CUDA-accelerated operations match PyTorch fallbacks."""

    def test_fused_transform_project_matches_fallback(self):
        from pilot.backends.cuda.kernels import (
            _fallback_transform_project,
            fused_transform_project,
        )

        B, N = 4, 200
        pts = torch.randn(B, N, 3, device="cuda") + torch.tensor(
            [0, 0, 20], device="cuda", dtype=torch.float32
        )
        T = torch.eye(4, device="cuda").unsqueeze(0).expand(B, -1, -1).contiguous()
        K = torch.tensor(
            [[500, 0, 256], [0, 500, 192], [0, 0, 1]],
            dtype=torch.float32, device="cuda",
        ).unsqueeze(0).expand(B, -1, -1).contiguous()

        triton_out = fused_transform_project(pts, T, K)
        pytorch_out = _fallback_transform_project(pts, T, K)

        assert torch.allclose(triton_out, pytorch_out, atol=1e-3)

    def test_batched_feature_residual_shape(self):
        from pilot.backends.cuda.kernels import batched_feature_residual

        B, C, H, W, N = 2, 32, 64, 80, 100
        q = torch.randn(B, C, H, W, device="cuda")
        r = torch.randn(B, C, H, W, device="cuda")
        uv = torch.rand(B, N, 2, device="cuda") * torch.tensor(
            [W - 1, H - 1], device="cuda"
        )

        res = batched_feature_residual(q, r, uv)
        assert res.shape == (B, N)
        assert (res >= 0).all()

    def test_parallel_hypothesis_score_shape(self):
        from pilot.backends.cuda.kernels import parallel_hypothesis_score

        B, M, N, C, H, W = 2, 16, 50, 32, 32, 40
        T_hyp = torch.eye(4, device="cuda").unsqueeze(0).unsqueeze(0).expand(
            B, M, -1, -1
        ).contiguous()
        anchors = torch.randn(B, N, 3, device="cuda") + torch.tensor(
            [0, 0, 30], device="cuda", dtype=torch.float32
        )
        K = torch.tensor(
            [[250, 0, 20], [0, 250, 16], [0, 0, 1]],
            dtype=torch.float32, device="cuda",
        ).unsqueeze(0).expand(B, -1, -1).contiguous()
        q = torch.randn(B, C, H, W, device="cuda")
        r = torch.randn(B, C, H, W, device="cuda")

        costs = parallel_hypothesis_score(T_hyp, anchors, K, q, r)
        assert costs.shape == (B, M)

    def test_cuda_depth_to_geo_anchors(self):
        from pilot.backends.cuda.kernels import cuda_depth_to_geo_anchors

        B, H, W = 2, 64, 80
        depth = torch.ones(B, H, W, device="cuda") * 50.0
        T_ref = torch.eye(4, device="cuda").unsqueeze(0).expand(B, -1, -1)
        K = torch.tensor(
            [[250, 0, 40], [0, 250, 32], [0, 0, 1]],
            dtype=torch.float32, device="cuda",
        ).unsqueeze(0).expand(B, -1, -1).contiguous()

        anchors = cuda_depth_to_geo_anchors(depth, T_ref, K, num_anchors=100)
        assert anchors.shape == (B, 100, 3)
