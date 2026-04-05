"""PiLoT training loop.

Config-driven training with:
- Adam optimizer, cosine LR + warmup
- bf16 mixed precision
- Gradient clipping
- Checkpoint management (top-k by val_loss)
- Early stopping
- Resume from checkpoint
"""

from __future__ import annotations

import math
import os
import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from pilot.dataset import PiLoTSyntheticDataset, depth_to_geo_anchors
from pilot.losses import PiLoTLoss
from pilot.model import PiLoTModelConfig, PiLoTSystem, project_points
from pilot.utils import seed_everything

# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class WarmupCosineScheduler:
    """Linear warmup + cosine decay learning rate scheduler."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-7,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            scale = self.current_step / max(self.warmup_steps, 1)
        else:
            progress = (self.current_step - self.warmup_steps) / max(
                self.total_steps - self.warmup_steps, 1
            )
            scale = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs, strict=False):
            pg["lr"] = max(self.min_lr, base_lr * scale)

    def state_dict(self) -> dict:
        return {"current_step": self.current_step}

    def load_state_dict(self, state: dict):
        self.current_step = state["current_step"]


# ---------------------------------------------------------------------------
# Checkpoint Manager
# ---------------------------------------------------------------------------

class CheckpointManager:
    """Keep top-k checkpoints by metric, auto-delete older ones."""

    def __init__(
        self,
        save_dir: str,
        keep_top_k: int = 2,
        metric: str = "val_loss",
        mode: str = "min",
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_top_k = keep_top_k
        self.mode = mode
        self.history: list[tuple[float, Path]] = []

    def save(
        self,
        state: dict,
        metric_value: float,
        step: int,
    ) -> Path:
        path = self.save_dir / f"checkpoint_step{step:06d}.pth"
        torch.save(state, path)
        self.history.append((metric_value, path))

        # Sort: best first
        self.history.sort(
            key=lambda x: x[0], reverse=(self.mode == "max")
        )

        # Prune
        while len(self.history) > self.keep_top_k:
            _, old_path = self.history.pop()
            old_path.unlink(missing_ok=True)

        # Save best
        best_val, best_path = self.history[0]
        best_dst = self.save_dir / "best.pth"
        shutil.copy2(best_path, best_dst)

        return path


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Stop training when metric stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = float("inf") if mode == "min" else float("-inf")
        self.counter = 0

    def step(self, metric: float) -> bool:
        if self.mode == "min":
            improved = metric < self.best - self.min_delta
        else:
            improved = metric > self.best + self.min_delta

        if improved:
            self.best = metric
            self.counter = 0
            return False
        self.counter += 1
        if self.counter >= self.patience:
            print(f"[EARLY STOP] No improvement for {self.patience} epochs. Stopping.")
            return True
        return False


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def build_model(cfg: dict) -> PiLoTSystem:
    """Build PiLoT model from config dict."""
    model_cfg = PiLoTModelConfig(
        backbone=cfg["model"]["backbone"],
        backbone_depth=cfg["model"]["backbone_depth"],
        backbone_pretrained=cfg["model"]["backbone_pretrained"],
        backbone_weights=cfg["model"].get("backbone_weights", ""),
        feature_channels=cfg["model"]["feature_channels"],
        pyramid_channels=cfg["model"]["pyramid_channels"],
        uncertainty_heads=cfg["model"]["uncertainty_heads"],
        num_hypotheses=cfg["jngo"]["num_hypotheses"],
        pitch_range_deg=cfg["jngo"]["pitch_range_deg"],
        yaw_range_deg=cfg["jngo"]["yaw_range_deg"],
        angle_step_deg=cfg["jngo"]["angle_step_deg"],
        translation_std=cfg["jngo"]["translation_std"],
        lm_iterations=[
            cfg["jngo"]["lm_iterations_coarse"],
            cfg["jngo"]["lm_iterations_mid"],
            cfg["jngo"]["lm_iterations_fine"],
        ],
        num_geo_anchors=cfg["jngo"]["num_geo_anchors"],
    )
    return PiLoTSystem(model_cfg)


def build_loss(cfg: dict) -> PiLoTLoss:
    """Build loss from config dict."""
    loss_cfg = cfg.get("loss", {})
    return PiLoTLoss(
        barron_alpha=loss_cfg.get("barron_alpha", 0.0),
        barron_scale=loss_cfg.get("barron_scale", 1.0),
        motion_lambda=loss_cfg.get("motion_lambda", 1.0),
    )


def train(cfg: dict, resume: str | None = None, device: str = "cuda"):
    """Run training loop.

    Args:
        cfg: parsed TOML config dictionary.
        resume: path to checkpoint to resume from.
        device: 'cuda' or 'cpu'.
    """
    seed_everything(cfg["training"]["seed"])

    # Paths
    ckpt_dir = cfg["checkpoint"]["output_dir"]
    log_dir = cfg["logging"]["log_dir"]
    tb_dir = cfg["logging"]["tensorboard_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    # Model
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] {n_params / 1e6:.1f}M trainable parameters")

    # Loss
    criterion = build_loss(cfg)

    # Optimizer
    lr = cfg["training"]["learning_rate"]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Dataset
    data_cfg = cfg["data"]
    train_dataset = PiLoTSyntheticDataset(
        root=data_cfg["train_path"],
        split="train",
        image_size=(data_cfg["image_width"], data_cfg["image_height"]),
        augment=True,
        pose_noise_t=tuple(cfg["training"]["augmentation"]["pose_noise_translation_m"]),
        pose_noise_r=tuple(cfg["training"]["augmentation"]["pose_noise_rotation_deg"]),
    )
    val_dataset = PiLoTSyntheticDataset(
        root=data_cfg.get("val_path", data_cfg["train_path"]),
        split="val",
        image_size=(data_cfg["image_width"], data_cfg["image_height"]),
        augment=False,
    )

    batch_size = cfg["training"]["batch_size"]
    if batch_size == "auto":
        batch_size = 4  # placeholder; use find_batch_size.py
        print(f"[BATCH] auto mode -> defaulting to {batch_size} (run find_batch_size.py)")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=data_cfg["pin_memory"],
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=data_cfg["pin_memory"],
    )

    # Scheduler
    epochs = cfg["training"]["epochs"]
    total_steps = epochs * max(len(train_loader), 1)
    warmup_steps = int(total_steps * cfg["scheduler"]["warmup_ratio"])
    min_lr = cfg["scheduler"]["min_lr"]
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps, min_lr)

    # Checkpoint / early stopping
    ckpt_mgr = CheckpointManager(
        ckpt_dir,
        keep_top_k=cfg["checkpoint"]["keep_top_k"],
        metric=cfg["checkpoint"]["metric"],
        mode=cfg["checkpoint"]["mode"],
    )
    early_stop = EarlyStopping(
        patience=cfg["early_stopping"]["patience"],
        min_delta=cfg["early_stopping"]["min_delta"],
    ) if cfg["early_stopping"]["enabled"] else None

    # AMP
    use_amp = cfg["training"]["precision"] in ("fp16", "bf16")
    amp_dtype = torch.bfloat16 if cfg["training"]["precision"] == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    # TensorBoard
    writer = SummaryWriter(tb_dir)

    # Resume
    start_epoch = 0
    global_step = 0
    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("step", 0)
        print(f"[RESUME] from epoch {start_epoch}, step {global_step}")

    # Print config summary
    print(f"[CONFIG] {cfg.get('_source', 'unknown')}")
    print(f"[BATCH] batch_size={batch_size}")
    print(f"[GPU] device={device}")
    print(f"[DATA] train={len(train_dataset)}, val={len(val_dataset)}")
    print(f"[TRAIN] {epochs} epochs, lr={lr}, optimizer=Adam")
    save_n = cfg["checkpoint"]["save_every_n_steps"]
    keep_k = cfg["checkpoint"]["keep_top_k"]
    print(f"[CKPT] save every {save_n} steps, keep best {keep_k}")

    # Training loop
    max_grad_norm = cfg["training"]["max_grad_norm"]
    save_every = cfg["checkpoint"]["save_every_n_steps"]
    num_anchors = cfg["jngo"]["num_geo_anchors"]

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            query_img = batch["query_image"].to(device)
            ref_img = batch["ref_image"].to(device)
            T_query = batch["T_query"].to(device)
            T_init = batch["T_init"].to(device)
            T_ref = batch["T_ref"].to(device)
            depth = batch["depth"].to(device)
            intrinsics = batch["intrinsics"].to(device)

            # Generate geo-anchors from depth
            geo_anchors = depth_to_geo_anchors(depth, T_ref, intrinsics, num_anchors)

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                # Forward pass (feature extraction only for training)
                query_feats = model.extract_features(query_img)
                ref_feats = model.extract_features(ref_img)

                # Compute reprojection targets
                pts_gt = project_points(geo_anchors, T_query, intrinsics)
                pts_pred = project_points(geo_anchors, T_init, intrinsics)

                # Loss
                loss_dict = criterion(
                    pts_pred=pts_pred,
                    pts_gt=pts_gt,
                    query_feat=query_feats[0][0],  # coarse level
                    ref_feat=ref_feats[0][0],
                    uncertainty=query_feats[0][1],
                )
                loss = loss_dict["total"]

            # NaN check
            if torch.isnan(loss):
                print("[FATAL] Loss is NaN -- stopping training")
                print("[FIX] Reduce lr, check data, check gradient clipping")
                return

            # Backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            # Log
            if global_step % 10 == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/lr", lr_now, global_step)
                print(
                    f"[Epoch {epoch + 1}/{epochs}] step={global_step} "
                    f"loss={loss.item():.4f} lr={lr_now:.2e}"
                )

            # Checkpoint
            if global_step % save_every == 0:
                state = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "step": global_step,
                    "config": cfg,
                }
                ckpt_mgr.save(state, epoch_loss / max(batch_idx + 1, 1), global_step)

        # Epoch summary
        avg_train_loss = epoch_loss / max(len(train_loader), 1)
        elapsed = time.time() - t0
        print(
            f"[Epoch {epoch + 1}/{epochs}] train_loss={avg_train_loss:.4f} "
            f"time={elapsed:.1f}s"
        )
        writer.add_scalar("epoch/train_loss", avg_train_loss, epoch)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                query_img = batch["query_image"].to(device)
                ref_img = batch["ref_image"].to(device)
                T_query = batch["T_query"].to(device)
                T_init = batch["T_init"].to(device)
                T_ref = batch["T_ref"].to(device)
                depth = batch["depth"].to(device)
                intrinsics = batch["intrinsics"].to(device)

                geo_anchors = depth_to_geo_anchors(depth, T_ref, intrinsics, num_anchors)
                query_feats = model.extract_features(query_img)
                ref_feats = model.extract_features(ref_img)
                pts_gt = project_points(geo_anchors, T_query, intrinsics)
                pts_pred = project_points(geo_anchors, T_init, intrinsics)

                loss_dict = criterion(
                    pts_pred=pts_pred,
                    pts_gt=pts_gt,
                    query_feat=query_feats[0][0],
                    ref_feat=ref_feats[0][0],
                    uncertainty=query_feats[0][1],
                )
                val_loss_sum += loss_dict["total"].item()
                val_count += 1

        val_loss = val_loss_sum / max(val_count, 1)
        print(f"[Epoch {epoch + 1}/{epochs}] val_loss={val_loss:.4f}")
        writer.add_scalar("epoch/val_loss", val_loss, epoch)

        # Save checkpoint
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "step": global_step,
            "config": cfg,
        }
        ckpt_mgr.save(state, val_loss, global_step)

        # Early stopping
        if early_stop and early_stop.step(val_loss):
            break

    writer.close()
    print("[DONE] Training complete.")
