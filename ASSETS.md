# ASSETS.md -- PiLoT Asset Inventory

## Pretrained Models

| Asset | Size (est.) | Source | Local Path | Status |
|-------|-------------|--------|------------|--------|
| MobileOne-S0 (ImageNet) | ~20MB | apple/ml-mobileone (HF/GitHub) | /mnt/forge-data/models/mobileone_s0.pth | NOT DOWNLOADED |
| PiLoT official weights | ~200MB | nudt-sawlab (if released) | /mnt/forge-data/models/pilot/ | NOT DOWNLOADED |

## Datasets

### Training Dataset
| Asset | Size (est.) | Source | Local Path | Status |
|-------|-------------|--------|------------|--------|
| PiLoT Synthetic (1.1M images) | ~500GB+ | Authors / project page | /mnt/forge-data/datasets/pilot_synthetic/ | NOT DOWNLOADED |

### Evaluation Datasets
| Asset | Size (est.) | Source | Local Path | Status |
|-------|-------------|--------|------------|--------|
| SynthCity-6 (54k frames) | ~50GB | Authors | /mnt/forge-data/datasets/synthcity6/ | NOT DOWNLOADED |
| UAVScenes (51.6k frames) | ~40GB | Authors | /mnt/forge-data/datasets/uavscenes/ | NOT DOWNLOADED |
| UAVD4L-2yr (7.2k frames) | ~10GB | Authors | /mnt/forge-data/datasets/uavd4l_2yr/ | NOT DOWNLOADED |
| UAVD4L-SynTarget (6k frames) | ~8GB | Authors | /mnt/forge-data/datasets/uavd4l_syntarget/ | NOT DOWNLOADED |

## Shared Infrastructure (already on disk)

| Asset | Path | Relevant |
|-------|------|----------|
| DINOv2 ViT-B/14 | /mnt/forge-data/models/dinov2_vitb14_pretrain.pth | Optional (feature comparison) |
| SERAPHIM UAV data | /mnt/forge-data/datasets/uav_detection/seraphim/ | Potential eval supplement |
| Fused image preprocess kernel | /mnt/forge-data/shared_infra/cuda_extensions/fused_image_preprocess/ | Use for data loading |
| SE(3) transform kernel | /mnt/forge-data/shared_infra/cuda_extensions/ | Use for pose transforms |

## Software Dependencies
| Package | Version | Purpose |
|---------|---------|---------|
| torch | >=2.2 (cu128) | Core framework |
| torchvision | >=0.17 (cu128) | Image transforms |
| timm | >=1.0 | MobileOne backbone |
| numpy | >=1.24 | Array ops |
| opencv-python | >=4.8 | Image I/O, undistortion |
| scipy | >=1.11 | Rotation utilities |
| pyproj | >=3.6 | WGS84/ECEF coordinate transforms |
| open3d | >=0.18 | 3D map loading, rendering |
| pyntcloud | >=0.3 | Point cloud utilities |
| tensorboard | >=2.14 | Training visualization |
| safetensors | >=0.4 | Model serialization |
| onnx | >=1.15 | Export |
| ruff | >=0.3 | Linting |
| pytest | >=7.0 | Testing |

## 3D Map Infrastructure (Optional -- for full pipeline)
| Asset | Purpose | Notes |
|-------|---------|-------|
| Cesium ion API key | 3D Tiles access | Required for Google 3D Tiles |
| AirSim | Synthetic data generation | Unreal Engine plugin |
| Unreal Engine 5 | Rendering | For synthetic dataset |

## Notes
- The PiLoT synthetic dataset is very large (~500GB+). For initial development, we can
  create a small synthetic subset or use the evaluation datasets only.
- MobileOne-S0 weights are available from Apple's ml-mobileone repo or via timm.
- The JNGO optimizer is custom CUDA code -- we implement it in PyTorch first, then
  optimize with CUDA kernels from shared infra (SE(3) transform, etc.).
- DO NOT download anything without checking disk space first.
