# PiLoT -- Neural Pixel-to-3D Registration for UAV Geo-localization

## Paper
- **Title**: PiLoT: Neural Pixel-to-3D Registration for UAV-based Ego and Target Geo-localization
- **Authors**: Xiaoya Cheng, Long Wang, Yan Liu, Xinyi Liu, Hanlin Tan, Yu Liu, Maojun Zhang, Shen Yan
- **ArXiv**: 2603.20778
- **Project Page**: https://nudt-sawlab.github.io/PiLoT/

## Summary
PiLoT registers live UAV video frames against a geo-referenced 3D map for GNSS-denied
localization. Instead of decoupled GNSS + VIO, it directly aligns neural features from the
query frame to rendered reference views of a photogrammetric 3D model. Key innovations:

1. **Dual-Thread Engine** -- decouples 3D map rendering from the localization pipeline;
   a Kalman filter predicts the next pose for the rendering thread while the localization
   thread optimizes the current frame, achieving >25 FPS on Jetson Orin.
2. **Million-Scale Synthetic Dataset** -- 1.1M+ images from 82 regions (AirSim + Cesium +
   Unreal Engine) with per-frame 6-DoF pose and pixel-wise metric depth.
3. **JNGO Optimizer** -- Joint Neural-Guided Stochastic-Gradient Optimizer that samples
   144 rotation-aware pose hypotheses and refines each with coarse-to-fine
   Levenberg-Marquardt on multi-scale feature pyramids.

## Architecture

### Backbone + Decoder
- **Encoder**: MobileOne-S0 (depth=3, ImageNet-initialized)
- **Decoder**: Compact U-Net with skip connections
- **Feature Pyramid**: 3 levels
  - Coarse (1/4 resolution): 128 channels
  - Mid (1/2 resolution): 64 channels
  - Fine (1x resolution): 32 channels
- Each level has parallel projection and uncertainty heads (3x3 conv)

### Dual-Thread Engine
- **Rendering Thread**: constant-velocity Kalman filter predicts reference pose, renders
  synthetic view from 3D map, back-projects depth-valid pixels to world-frame geo-anchors
- **Localization Thread**: extracts multi-scale features from query + reference, runs JNGO
  optimizer, returns estimated 6-DoF pose

### JNGO Optimizer (3 phases)
1. **Rotation-Aware Hypothesis Generation**: pitch/yaw in [-11, 11] deg at 2-deg steps
   (M=144 hypotheses), translation sampled from Kalman predictor N(0,1)m
2. **Neural-Guided Parallel LM Refinement**: per-hypothesis LM on coarse-to-fine pyramid;
   iterations = 2 (coarse), 3 (mid), 4 (fine); Jacobian via chain rule on feature gradients
3. **Motion-Constrained Selection**: minimize C_total = C_photo + lambda * geodesic(SE3)

## Loss Functions
- **Training (Eq.3)**: L = sum_j rho_B(||p_j^q - p_tilde_j^q||_2^2)
  Barron's robust loss on reprojection error
- **Photometric Cost (Eq.7)**: C_photo = sum_j rho(w_l(j) * ||r_{j,m}^l||_2^2)
  Huber robust loss with learned uncertainty weighting
- **Total Cost (Eq.10)**: C_total = C_photo + lambda * ||log(T_pred^{-1} T_m')||_2^2
  Photometric alignment + SE(3) motion regularization

## Hyperparameters
| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Epochs | 30 |
| Image resolution (train) | 1600x1200 |
| Image resolution (inference) | 512px short side |
| Feature channel width C | 32 |
| Geo-anchors N | 500 |
| Pose hypotheses M | 144 |
| Pitch/yaw range | +/-11 deg, 2-deg step |
| LM iterations (coarse) | 2 |
| LM iterations (mid) | 3 |
| LM iterations (fine) | 4 |
| Translation noise | N(0,1) meters |
| Training GPUs (paper) | 8x RTX 4090 |
| Mixed precision | bf16 |
| Seed | 42 |

## Datasets

### Training
- **PiLoT Synthetic**: 1.1M+ images, 82 regions, 650+ km trajectories
  - AirSim + Cesium + Unreal Engine pipeline
  - Weather: Sunny, Cloudy, Rainy, Foggy, Snowy
  - Time: Day, Sunset, Night
  - Altitude: sub-800m, camera pitch 20-90 deg
  - GT: 6-DoF poses (WGS84 + ECEF), pixel-wise metric depth

### Evaluation
| Dataset | Frames | Type | GT |
|---------|--------|------|-----|
| SynthCity-6 | 54k | Synthetic | 6-DoF + depth |
| UAVScenes | 51.6k | Real | 6-DoF poses |
| UAVD4L-2yr | 7.2k | Real | RTK-GPS (cm) |
| UAVD4L-SynTarget | 6k | Synthetic | Target 3D loc |

## Evaluation Metrics
- Median translation error (m)
- Median rotation error (deg)
- Recall@k (k in {1, 3, 5} meters and degrees)
- Completeness (%)
- FPS

## Model Requirements
- MobileOne-S0 pretrained weights (ImageNet)
- 3D photogrammetric map tiles (Google 3D Tiles / Cesium)
- AirSim simulator (for synthetic data generation)

## VRAM Estimate
- MobileOne-S0 + U-Net decoder: ~50M params -> ~200MB fp32
- Feature maps at 512px: moderate
- 144 parallel hypotheses: main memory consumer
- Estimated single-GPU VRAM: 8-12GB (fits L4 23GB)
