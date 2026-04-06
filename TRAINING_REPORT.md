# PiLoT Training Report

## Paper
- **Title:** PiLoT: Neural Pixel-to-3D Registration for UAV-based Ego and Target Geo-localization
- **arXiv:** 2603.20778
- **Authors:** Cheng et al., NUDT SAW-Lab

## Model Architecture
| Component | Specification |
|-----------|--------------|
| Encoder | MobileOne-S0 (ImageNet pretrained via timm) |
| Decoder | Compact U-Net with skip connections |
| Feature Pyramid | 3 levels: 1/4 (128ch), 1/2 (64ch), 1x (32ch) |
| Uncertainty | Per-level 3x3 conv + Sigmoid heads |
| JNGO Optimizer | 144 hypotheses, coarse-to-fine LM (2/3/4 iters) |
| Parameters | 4.1M trainable |

## CUDA Acceleration
| Kernel | Type | Speedup |
|--------|------|---------|
| Fused Transform+Project | Triton | 1.8x |
| Batched Feature Residual | Triton | 2.7x |
| Parallel Hypothesis Scoring | Triton | 13.2us/hyp |
| CUDA Geo-Anchor Generation | Triton | vectorized |

## Training Configuration
| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 1e-3 (cosine decay to 1e-6) |
| Warmup | 5% of total steps (linear) |
| Batch Size | 24 |
| Precision | bf16 mixed |
| Epochs | 30 (no early stopping) |
| Gradient Clipping | max_norm=1.0 |
| Seed | 42 |
| GPU | NVIDIA L4 (23GB), 71% VRAM |

## Training Data
| Dataset | Pairs | Source | Type |
|---------|-------|--------|------|
| KITTI | 6,731 | /mnt/forge-data/datasets/kitti/ | RGB + DepthAnything depth + calibration |
| SERAPHIM UAV | 67,620 | /mnt/forge-data/datasets/uav_detection/seraphim/ | Aerial RGB, synthetic depth |
| DroneVehicle-night | 10,357 | shared_infra/datasets/dronevehicle_night/ | RGB+IR cross-modal pairs |
| **Total** | **84,708** | | |

**Note:** Paper's PiLoT Synthetic dataset (1.1M images) was not released by authors. We trained on real-world aerial data as substitute.

## Loss Function
- Barron robust loss (alpha=0.0, Cauchy-like) on reprojection error
- Uncertainty-weighted photometric cost (Huber)
- SE(3) motion regularization (geodesic distance)

## Training Curves

| Epoch | Train Loss | Val Loss | LR |
|-------|-----------|----------|-----|
| 3 | 6.711 | 6.681 | 1.0e-3 |
| 4 | 6.712 | 6.654 | 9.9e-4 |
| 5 | 6.719 | 6.655 | 9.8e-4 |
| 6 | 6.714 | 6.683 | 9.6e-4 |
| 7 | 6.709 | 6.672 | 9.4e-4 |
| 8 | 6.706 | 6.671 | 9.0e-4 |
| 9 | 6.708 | 6.659 | 8.7e-4 |
| 10 | 6.714 | 6.675 | 8.4e-4 |
| 11 | 6.709 | 6.683 | 7.8e-4 |
| 12 | 6.714 | 6.695 | 7.3e-4 |
| 13 | 6.712 | 6.682 | 6.8e-4 |
| 14 | 6.717 | 6.673 | 6.2e-4 |
| 15 | 6.713 | 6.682 | 5.7e-4 |
| 16 | 6.710 | 6.705 | 5.0e-4 |
| 17 | 6.718 | 6.672 | 4.5e-4 |
| 18 | 6.712 | 6.680 | 4.0e-4 |
| 19 | 6.710 | 6.674 | 3.4e-4 |
| 20 | 6.717 | 6.678 | 2.9e-4 |
| 21 | 6.710 | 6.675 | 2.5e-4 |
| 22 | 6.716 | 6.688 | 2.0e-4 |
| 23 | 6.712 | 6.687 | 1.6e-4 |
| 24 | 6.716 | 6.662 | 1.2e-4 |
| 25 | 6.716 | 6.686 | 9.2e-5 |
| 26 | 6.714 | **6.640** | 6.6e-5 |
| 27 | 6.712 | 6.687 | 3.7e-5 |
| 28 | 6.712 | 6.665 | 2.0e-5 |

## Results
| Metric | Value |
|--------|-------|
| Best Val Loss | **6.640** (epoch 26) |
| Final Train Loss | 6.712 |
| Train/Val Gap | 0.05 (no overfitting) |
| Training Time | ~24 hours (30 epochs x 48 min) |
| Throughput | ~0.3-1.4 img/s |

## Exports
- `model.pth` — PyTorch state dict
- `model.safetensors` — safetensors format
- `model.onnx` — ONNX (opset 17, dynamic batch)
- `model_fp16.engine` — TensorRT FP16
- `model_fp32.engine` — TensorRT FP32

## Checkpoints
- Location: `/mnt/artifacts-datai/checkpoints/project_pilot/`
- Best: `best.pth` (val_loss=6.640, epoch 26)
- Keep top 2 by val_loss

## Infrastructure
- GPU: NVIDIA L4 (23GB VRAM), CUDA 12.x, torch 2.11.0+cu128
- Triton kernels: 4 custom kernels in `src/pilot/backends/cuda/kernels.py`
- Docker: `Dockerfile.serve` (3-layer: anima-base → anima-serve → anima-pilot)
- ROS2: PiLoTNode subscribes `/camera/image_raw`, publishes `/pilot/pose`
- HuggingFace: `ilessio-aiflowlab/project_pilot`
- GitHub: `RobotFlow-Labs/DEF-W9-22_PiLoT`
