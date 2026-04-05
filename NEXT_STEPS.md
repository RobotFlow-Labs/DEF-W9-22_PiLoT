# NEXT_STEPS.md
> Last updated: 2026-04-05
> MVP Readiness: 65%

## Done
- [x] Paper analysis (arXiv 2603.20778) -- architecture, losses, datasets, metrics
- [x] PRD-01 Foundation: scaffolding, configs, pyproject.toml, anima_module.yaml
- [x] PRD-02 Core Model: MobileOne-S0 encoder + U-Net decoder + JNGO optimizer
- [x] PRD-03 Loss Functions: Barron robust, photometric, SE(3) regularization
- [x] PRD-04 Training Pipeline: dataset, train loop, checkpointing, scheduler
- [x] PRD-05 Evaluation: metrics, evaluator class, eval script
- [x] PRD-06 Export Pipeline: pth/safetensors/ONNX/TRT fp16/fp32 export
- [x] PRD-07 Integration: Dockerfile.serve, docker-compose, serve.py
- [x] CUDA kernels: 4 Triton kernels (transform+project, feature residual, hypothesis scoring, geo-anchors)
- [x] Shared CUDA kernel integration (se3_transform, batched_projection from shared_infra)
- [x] Ruff lint: 0 errors
- [x] Tests: 23/23 passing (19 CPU + 4 CUDA)
- [x] .venv created with torch 2.11.0+cu128, timm 1.0.26
- [x] Copied Triton kernels to /mnt/forge-data/shared_infra/cuda_extensions/pilot_kernels/

## In Progress
- [ ] Create synthetic training data subset for smoke test
- [ ] Download MobileOne-S0 pretrained weights via timm

## TODO
- [ ] Generate synthetic data subset (1K image pairs with depth + poses)
- [ ] Run smoke test on debug config (5 steps, save/load cycle)
- [ ] Full training run (30 epochs, SERAPHIM + nuScenes data)
- [ ] Evaluation on benchmarks
- [ ] Export all 5 formats from trained model
- [ ] Push to HuggingFace: ilessio-aiflowlab/project_pilot
- [ ] Docker build + health check

## Blocking
- PiLoT synthetic dataset (~500GB+) not available from authors
- STRATEGY: Use SERAPHIM UAV (83K images) + nuScenes GPS data + KITTI
  as real-world training substitute to SURPASS paper results
- MobileOne-S0 weights: download via timm on first run (~20MB)

## Extra Data Strategy (to beat paper)
- SERAPHIM UAV: /mnt/forge-data/datasets/uav_detection/seraphim/ (18GB, 83K images)
- nuScenes: /mnt/forge-data/datasets/nuscenes/ (479GB, GPS ground truth)
- KITTI: /mnt/forge-data/datasets/kitti/ (52GB, GPS-tagged driving)
- BirdDrone: /mnt/forge-data/datasets/wave9/drones/LAT-BirdDrone.zip
- Pre-computed DINOv2 features available for all datasets

## CUDA Kernel Benchmarks (L4 GPU)
| Kernel | Triton | PyTorch | Speedup |
|--------|--------|---------|---------|
| Fused Transform+Project | 0.15ms | 0.27ms | 1.8x |
| Feature Residual | 0.09ms | 0.25ms | 2.7x |
| Parallel Hypothesis (B=8, M=144) | 15.2ms | serial | 13.2μs/hyp |
