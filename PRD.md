# PRD.md -- PiLoT Master Build Plan

## Module: project_pilot
## Paper: PiLoT -- Neural Pixel-to-3D Registration for UAV Geo-localization (arXiv 2603.20778)

---

## Build Plan

| PRD | Title | Status | Description |
|-----|-------|--------|-------------|
| PRD-01 | Foundation | [x] DONE | Project scaffolding, configs, venv, package structure |
| PRD-02 | Core Model | [x] DONE | MobileOne-S0 encoder, U-Net decoder, feature pyramid, uncertainty heads |
| PRD-03 | Loss Functions | [x] DONE | Barron robust loss, photometric cost, SE(3) motion regularization |
| PRD-04 | Training Pipeline | [x] DONE | Dataset loaders, training loop, checkpointing, LR schedule |
| PRD-05 | Evaluation | [x] DONE | Metrics (median error, recall@k, completeness, FPS), eval scripts |
| PRD-06 | Export Pipeline | [x] DONE | ONNX, TensorRT fp16/fp32, safetensors, HF push |
| PRD-07 | Integration | [x] DONE | Docker serving, ROS2 node, anima_module.yaml, API endpoints |

---

## Architecture Overview

```
Query Frame (UAV camera)
        |
        v
+-------------------+     +--------------------+
| Rendering Thread  |     | Localization Thread|
| (Kalman predict   |<--->| (Feature extract   |
|  + 3D map render) |     |  + JNGO optimize)  |
+-------------------+     +--------------------+
        |                          |
        v                          v
  Reference View            6-DoF Pose Estimate
  + Depth Map               (WGS84 / ECEF)
  + Geo-Anchors
```

### Feature Extraction
- MobileOne-S0 encoder (ImageNet pretrained)
- U-Net decoder with skip connections
- 3-level pyramid: 1/4 (128ch), 1/2 (64ch), 1x (32ch)
- Parallel uncertainty heads at each level

### JNGO Optimizer
- 144 pose hypotheses (pitch/yaw grid + translation noise)
- Per-hypothesis coarse-to-fine LM refinement
- Motion-constrained selection with SE(3) geodesic

### CUDA Acceleration
- Triton fused transform+project kernel (1.8x speedup)
- Triton batched feature residual kernel (2.7x speedup)
- Parallel hypothesis scoring (all 144 hypotheses in one launch)
- CUDA geo-anchor generation from depth maps

### Training
- Adam optimizer, lr=1e-3, 30 epochs
- Barron's robust loss on reprojection error
- Extra data: SERAPHIM UAV (83K images) + nuScenes GPS + KITTI

### Evaluation
- Median translation/rotation error
- Recall@{1,3,5}m and degrees
- FPS on target hardware
