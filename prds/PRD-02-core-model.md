# PRD-02: Core Model

## Objective
Implement the PiLoT feature extraction network: MobileOne-S0 encoder + U-Net decoder
with 3-level feature pyramid and uncertainty heads.

## Architecture
```
Input Image (H x W x 3)
    |
    v
MobileOne-S0 Encoder (depth=3, ImageNet)
    |--- stage1 features (H/4 x W/4)
    |--- stage2 features (H/2 x W/2)
    |--- stage3 features (H x W)
    v
U-Net Decoder (skip connections)
    |
    +---> Coarse (H/4 x W/4 x 128) ---> proj_head + uncertainty_head
    +---> Mid    (H/2 x W/2 x 64)  ---> proj_head + uncertainty_head
    +---> Fine   (H x W x 32)      ---> proj_head + uncertainty_head
```

## Deliverables
- `src/pilot/model.py` -- PiLoTFeatureNet with:
  - MobileOne-S0 encoder via timm
  - U-Net decoder with skip connections
  - 3-level feature pyramid (128, 64, 32 channels)
  - Per-level projection head (3x3 conv -> feature_dim)
  - Per-level uncertainty head (3x3 conv -> 1, sigmoid activation)
- `src/pilot/model.py` -- JNGOOptimizer with:
  - Hypothesis generation (pitch/yaw grid + translation noise)
  - Per-hypothesis LM refinement on feature residuals
  - Motion-constrained selection
- `src/pilot/model.py` -- PiLoTSystem combining feature net + JNGO

## Acceptance Criteria
- Forward pass produces 3-level features + uncertainties
- JNGO optimizer accepts features + geo-anchors, outputs 6-DoF pose
- Shapes verified in unit tests
- VRAM < 12GB for batch_size=1 at 512px
