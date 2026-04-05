# PRD-04: Training Pipeline

## Objective
Implement data loading, training loop, checkpointing, and LR scheduling
matching the paper's setup.

## Training Protocol (Paper)
- Optimizer: Adam, lr=1e-3
- Epochs: 30
- Precision: bf16 mixed precision
- Data: consecutive frames from trajectories with pose noise
- Augmentation: Fourier noise + photometric jitter
- Pose noise: 5-15m translation, 5-15 deg rotation

## Deliverables

### Dataset (`src/pilot/dataset.py`)
- `PiLoTSyntheticDataset`: loads image pairs + 6-DoF poses + depth maps
- `PiLoTEvalDataset`: loads query images + GT poses for evaluation
- Augmentation pipeline: Fourier noise, photometric jitter, pose perturbation
- DataLoader with configurable workers, pin_memory

### Training Loop (`src/pilot/train.py`)
- Config-driven (TOML -> Pydantic settings)
- Adam optimizer with cosine LR + 5% warmup
- bf16 mixed precision via torch.amp
- Gradient clipping max_norm=1.0
- CheckpointManager (keep top 2 by val_loss)
- EarlyStopping (patience=10)
- NaN detection with auto-halt
- Resume from checkpoint support
- nohup-compatible logging

### Scripts
- `scripts/train.py`: CLI entry point with --config, --resume, --device
- `scripts/find_batch_size.py`: auto batch size detection

## Acceptance Criteria
- Training loop runs on debug config for 2 epochs without error
- Checkpoint save/load/resume cycle verified
- LR schedule matches warmup + cosine decay
- Logging outputs to /mnt/artifacts-datai/logs/project_pilot/
- batch_size="auto" triggers GPU memory probing
