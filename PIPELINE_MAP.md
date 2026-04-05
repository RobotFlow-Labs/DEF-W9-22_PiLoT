# PIPELINE_MAP.md -- PiLoT Paper Reproduction Pipeline

## Paper: PiLoT (arXiv 2603.20778)

| Step | Section | Task | Status |
|------|---------|------|--------|
| 1 | §3.1 | MobileOne-S0 encoder + U-Net decoder | [x] DONE |
| 2 | §3.1 | 3-level feature pyramid (128/64/32 ch) | [x] DONE |
| 3 | §3.1 | Uncertainty heads (per-level 3x3 conv) | [x] DONE |
| 4 | §3.2 | JNGO: rotation-aware hypothesis gen (M=144) | [x] DONE |
| 5 | §3.2 | JNGO: coarse-to-fine LM refinement | [x] DONE |
| 6 | §3.2 | JNGO: motion-constrained selection | [x] DONE |
| 7 | §3.3 | Barron robust loss (Eq.3) | [x] DONE |
| 8 | §3.3 | Photometric cost with uncertainty (Eq.7) | [x] DONE |
| 9 | §3.3 | SE(3) motion regularization (Eq.10) | [x] DONE |
| 10 | §4.1 | Kalman filter pose prediction | [x] DONE |
| 11 | §4.1 | Geo-anchor back-projection from depth | [x] DONE |
| 12 | §4.2 | Training loop (Adam, lr=1e-3, 30 epochs) | [x] DONE |
| 13 | §4.2 | bf16 mixed precision + gradient clipping | [x] DONE |
| 14 | §4.3 | Evaluation metrics (median error, recall@k) | [x] DONE |
| 15 | - | Export: pth/safetensors/ONNX/TRT | [x] DONE |
| 16 | - | CUDA Triton kernels (4 custom) | [x] DONE |
| 17 | §4.2 | Training on real UAV data | [ ] TODO |
| 18 | §4.3 | Benchmark evaluation | [ ] TODO |
| 19 | - | HuggingFace push | [ ] TODO |
