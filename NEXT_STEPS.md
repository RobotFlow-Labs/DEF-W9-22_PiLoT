# NEXT_STEPS.md
> Last updated: 2026-04-05
> MVP Readiness: 75%

## Done
- [x] Paper analysis (arXiv 2603.20778)
- [x] PRD-01 through PRD-07: all implemented
- [x] 4 custom Triton CUDA kernels (1.8-2.7x speedup)
- [x] Shared kernel integration (se3_transform, batched_projection)
- [x] Multi-dataset adapter: KITTI (6.7K) + SERAPHIM (67.6K) + DroneVehicle (10.4K) = 84.7K pairs
- [x] Ruff lint: 0 errors | Tests: 23/23 passing
- [x] .venv: torch 2.11.0+cu128, timm 1.0.26
- [x] 7 focused git commits
- [x] Smoke test: loss decreasing, checkpoint save/load verified
- [x] Training launched: GPU 7, bs=24, 77% VRAM, bf16, 30 epochs
- [x] Auto-monitor set up: will trigger export pipeline when training completes
- [x] Triton kernels copied to /mnt/forge-data/shared_infra/cuda_extensions/pilot_kernels/

## In Progress
- [ ] Training: epoch 1/30, loss=6.6 and declining (GPU 7, 100% util)
  - PID in /mnt/artifacts-datai/logs/project_pilot/train.pid
  - Log: /mnt/artifacts-datai/logs/project_pilot/train_20260405_0747.log
  - Monitor: /mnt/artifacts-datai/logs/project_pilot/monitor.log

## TODO (automated — will run when training completes)
- [ ] Export: pth + safetensors + ONNX + TRT FP16 + TRT FP32
- [ ] Push to HuggingFace: ilessio-aiflowlab/project_pilot
- [ ] Git push to origin main
- [ ] Generate TRAINING_REPORT.md

## TODO (manual — future sessions)
- [ ] Evaluate on official benchmarks when released
- [ ] Swap in PiLoT synthetic dataset when authors release it
- [ ] Docker build + integration test
- [ ] ROS2 node test

## CUDA Kernel Benchmarks (L4 GPU)
| Kernel | Triton | PyTorch | Speedup |
|--------|--------|---------|---------|
| Fused Transform+Project | 0.15ms | 0.27ms | 1.8x |
| Feature Residual | 0.09ms | 0.25ms | 2.7x |
| Parallel Hypothesis (B=8, M=144) | 15.2ms | serial | 13.2μs/hyp |

## Training Config
- Model: MobileOne-S0 + U-Net decoder + JNGO (4.1M params)
- Data: 84,708 real-world pairs (3 datasets)
- Optimizer: Adam, lr=1e-3, cosine + 5% warmup
- Precision: bf16, grad clip=1.0
- Checkpoint: save every 500 steps, keep best 2
- Early stopping: patience=10
