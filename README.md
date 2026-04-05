<p align="center">
  <img src="assets/hero.png" alt="PiLoT Hero" width="100%">
</p>

# PiLoT — Neural Pixel-to-3D Registration for UAV Geo-localization

GNSS-denied UAV localization via neural pixel-to-3D map registration. Registers live UAV video frames against geo-referenced 3D maps for real-time 6-DoF pose estimation without GPS.

**Paper:** Cheng et al., [arXiv 2603.20778](https://arxiv.org/abs/2603.20778)

## Architecture

- **Encoder:** MobileOne-S0 (ImageNet pretrained) via timm
- **Decoder:** Compact U-Net with skip connections
- **Feature Pyramid:** 3 levels — 1/4 (128ch), 1/2 (64ch), 1x (32ch)
- **JNGO Optimizer:** 144 parallel pose hypotheses, coarse-to-fine Levenberg-Marquardt
- **CUDA Acceleration:** 4 custom Triton kernels (1.8-2.7x speedup)

## Quick Start

```bash
uv venv .venv --python python3.11
source .venv/bin/activate
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install -e ".[dev]"

# Run tests
uv run pytest tests/ -v

# Train
CUDA_VISIBLE_DEVICES=0 python scripts/train_cu.py --config configs/train_real.toml
```

## ANIMA Module

Part of the [ANIMA](https://robotflow-labs.github.io) defense suite by RobotFlow Labs.
