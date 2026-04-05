"""Tests for PiLoT model architecture."""

from __future__ import annotations

import torch


def test_se3_exp_identity():
    """se3_exp(zeros) should produce identity matrix."""
    from pilot.model import se3_exp

    xi = torch.zeros(2, 6)
    T = se3_exp(xi)
    assert T.shape == (2, 4, 4)
    eye = torch.eye(4).unsqueeze(0).expand(2, -1, -1)
    assert torch.allclose(T, eye, atol=1e-5)


def test_se3_log_roundtrip():
    """se3_log(se3_exp(xi)) should recover xi for small angles."""
    from pilot.model import se3_exp, se3_log

    xi = torch.tensor([[0.1, 0.2, 0.3, 0.05, 0.1, 0.15]])
    T = se3_exp(xi)
    xi_recovered = se3_log(T)
    assert torch.allclose(xi[:, :3], xi_recovered[:, :3], atol=0.1)


def test_skew_symmetric():
    """skew_symmetric should produce antisymmetric matrix."""
    from pilot.model import skew_symmetric

    w = torch.tensor([[1.0, 2.0, 3.0]])
    K = skew_symmetric(w)
    assert K.shape == (1, 3, 3)
    # K + K^T = 0
    assert torch.allclose(K + K.transpose(-1, -2), torch.zeros(1, 3, 3), atol=1e-6)


def test_project_points_shape():
    """project_points should produce (B, N, 2) output."""
    from pilot.model import project_points

    B, N = 2, 100
    pts = torch.randn(B, N, 3) + torch.tensor([0.0, 0.0, 10.0])  # ensure positive z
    T = torch.eye(4).unsqueeze(0).expand(B, -1, -1)
    K = torch.tensor([
        [500, 0, 256],
        [0, 500, 192],
        [0, 0, 1],
    ], dtype=torch.float32).unsqueeze(0).expand(B, -1, -1)
    pts_2d = project_points(pts, T, K)
    assert pts_2d.shape == (B, N, 2)


def test_feature_net_output_shapes():
    """PiLoTFeatureNet should produce 3-level features with correct shapes."""
    from pilot.model import PiLoTFeatureNet, PiLoTModelConfig

    cfg = PiLoTModelConfig(
        backbone_pretrained=False,
        pyramid_channels=[128, 64, 32],
    )
    net = PiLoTFeatureNet(cfg)
    x = torch.randn(1, 3, 256, 320)

    with torch.no_grad():
        outputs = net(x)

    assert len(outputs) == 3  # coarse, mid, fine
    for feat, unc in outputs:
        assert feat.shape[0] == 1
        assert feat.ndim == 4
        if unc is not None:
            assert unc.shape[0] == 1
            assert unc.shape[1] == 1


def test_pilot_system_forward():
    """PiLoTSystem forward should accept images and return pose."""
    from pilot.model import PiLoTModelConfig, PiLoTSystem

    cfg = PiLoTModelConfig(
        backbone_pretrained=False,
        num_hypotheses=4,  # small for testing
        angle_step_deg=6.0,
        lm_iterations=[1, 1, 1],
        num_geo_anchors=10,
    )
    model = PiLoTSystem(cfg)

    B = 1
    H, W = 128, 160
    query = torch.randn(B, 3, H, W)
    ref = torch.randn(B, 3, H, W)
    T_pred = torch.eye(4).unsqueeze(0)
    anchors = torch.randn(B, 10, 3) + torch.tensor([0.0, 0.0, 50.0])
    K = torch.tensor([
        [250, 0, 80],
        [0, 250, 64],
        [0, 0, 1],
    ], dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        T_est = model(query, ref, T_pred, anchors, K)

    assert T_est.shape == (B, 4, 4)


def test_model_config_defaults():
    """PiLoTModelConfig should have reasonable defaults."""
    from pilot.model import PiLoTModelConfig

    cfg = PiLoTModelConfig()
    assert cfg.backbone == "mobileone_s0"
    assert cfg.feature_channels == 32
    assert cfg.num_hypotheses == 144
    assert len(cfg.pyramid_channels) == 3
    assert cfg.pyramid_channels == [128, 64, 32]
