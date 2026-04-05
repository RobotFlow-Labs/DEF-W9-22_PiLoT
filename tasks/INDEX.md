# Task Index -- PiLoT Module

## PRD-01: Foundation
- [x] T01.1: Create pyproject.toml with hatchling, torch cu128, ruff
- [x] T01.2: Create src/pilot/ package with __init__.py
- [x] T01.3: Create configs/paper.toml and configs/debug.toml
- [x] T01.4: Create anima_module.yaml
- [x] T01.5: Create CLAUDE.md, ASSETS.md, PRD.md
- [x] T01.6: Create test stubs (tests/test_model.py, tests/test_dataset.py)
- [x] T01.7: Create script entry points (scripts/train.py, scripts/evaluate.py)
- [x] T01.8: Create Dockerfile.serve + docker-compose.serve.yml
- [x] T01.9: Create .env.serve

## PRD-02: Core Model
- [x] T02.1: Implement MobileOne-S0 encoder wrapper via timm
- [x] T02.2: Implement U-Net decoder with skip connections
- [x] T02.3: Implement 3-level feature pyramid (128, 64, 32 channels)
- [x] T02.4: Implement projection + uncertainty heads per level
- [x] T02.5: Implement PiLoTFeatureNet combining encoder + decoder + heads
- [x] T02.6: Implement hypothesis generation (pitch/yaw grid + translation)
- [x] T02.7: Implement per-hypothesis LM refinement
- [x] T02.8: Implement motion-constrained selection
- [x] T02.9: Implement JNGOOptimizer combining all 3 phases
- [x] T02.10: Implement PiLoTSystem (feature net + JNGO)
- [ ] T02.11: Unit test forward pass shapes and VRAM usage

## PRD-03: Loss Functions
- [x] T03.1: Implement Barron's robust loss (differentiable, parameterized)
- [x] T03.2: Implement photometric cost with uncertainty weighting
- [x] T03.3: Implement SE(3) log map (numerically stable)
- [x] T03.4: Implement SE(3) motion regularization loss
- [x] T03.5: Implement combined PiLoTLoss
- [ ] T03.6: Unit test all losses with known inputs

## PRD-04: Training Pipeline
- [x] T04.1: Implement PiLoTSyntheticDataset (image pairs + poses + depth)
- [x] T04.2: Implement data augmentation (Fourier noise, photometric jitter)
- [x] T04.3: Implement config loading (TOML -> Pydantic)
- [x] T04.4: Implement training loop with AMP, gradient clipping
- [x] T04.5: Implement CheckpointManager (top-k by val_loss)
- [x] T04.6: Implement EarlyStopping
- [x] T04.7: Implement WarmupCosineScheduler
- [x] T04.8: Implement resume-from-checkpoint
- [x] T04.9: Create scripts/train.py CLI
- [ ] T04.10: Smoke test: 2-epoch run on debug config
- [ ] T04.11: Create scripts/find_batch_size.py

## PRD-05: Evaluation
- [x] T05.1: Implement pose error computation (translation + rotation)
- [x] T05.2: Implement recall@k metrics
- [x] T05.3: Implement completeness metric
- [x] T05.4: Implement FPS measurement
- [x] T05.5: Implement PiLoTEvaluator class
- [x] T05.6: Create scripts/evaluate.py CLI
- [ ] T05.7: Test evaluation pipeline on synthetic data

## PRD-06: Export Pipeline
- [ ] T06.1: Implement export_safetensors
- [ ] T06.2: Implement export_onnx with dynamic axes
- [ ] T06.3: Implement export_trt via shared toolkit
- [ ] T06.4: Create scripts/export.py CLI
- [ ] T06.5: Verify round-trip accuracy

## PRD-07: Integration
- [x] T07.1: Create Dockerfile.serve (3-layer)
- [x] T07.2: Create docker-compose.serve.yml
- [x] T07.3: Create .env.serve
- [x] T07.4: Implement src/pilot/serve.py (PiLoTNode)
- [ ] T07.5: Test Docker build
- [ ] T07.6: Test API endpoints
- [ ] T07.7: Push to HuggingFace
