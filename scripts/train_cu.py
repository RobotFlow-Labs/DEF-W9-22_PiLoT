#!/usr/bin/env python3
"""PiLoT CUDA-accelerated training on multi-dataset.

Uses Triton kernels for projection, feature residuals, and hypothesis scoring.
Trains on KITTI + SERAPHIM + DroneVehicle real-world data.

Usage:
    CUDA_VISIBLE_DEVICES=7 uv run python scripts/train_cu.py --config configs/paper.toml
    CUDA_VISIBLE_DEVICES=7 uv run python scripts/train_cu.py --config configs/debug.toml
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pilot.dataset import PiLoTSyntheticDataset, depth_to_geo_anchors
from pilot.dataset_multi import build_multi_dataset
from pilot.losses import PiLoTLoss
from pilot.model import PiLoTModelConfig, PiLoTSystem, project_points
from pilot.train import CheckpointManager, EarlyStopping, WarmupCosineScheduler
from pilot.utils import load_config, seed_everything


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


def train_cuda(cfg: dict, resume: str | None = None, device: str = "cuda"):
    """CUDA-accelerated training loop with multi-dataset support."""
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

    # Check CUDA kernels
    try:
        from pilot.backends.cuda.kernels import cuda_available
        print(f"[CUDA] Triton kernels: {'ACTIVE' if cuda_available() else 'FALLBACK'}")
    except ImportError:
        print("[CUDA] Triton kernels: NOT AVAILABLE")

    # Loss
    loss_cfg = cfg.get("loss", {})
    criterion = PiLoTLoss(
        barron_alpha=loss_cfg.get("barron_alpha", 0.0),
        barron_scale=loss_cfg.get("barron_scale", 1.0),
        motion_lambda=loss_cfg.get("motion_lambda", 1.0),
    )

    # Optimizer
    lr = cfg["training"]["learning_rate"]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Dataset — try multi-dataset first, fallback to synthetic
    data_cfg = cfg["data"]
    img_size = (data_cfg["image_width"], data_cfg["image_height"])
    batch_size = cfg["training"]["batch_size"]
    if batch_size == "auto":
        batch_size = 16  # reasonable default for L4
        print(f"[BATCH] auto -> {batch_size}")

    try:
        train_dataset = build_multi_dataset("train", image_size=img_size, augment=True)
        val_dataset = build_multi_dataset("val", image_size=img_size, augment=False)
    except RuntimeError:
        print("[DATA] Multi-dataset failed, falling back to synthetic")
        train_dataset = PiLoTSyntheticDataset(
            root=data_cfg["train_path"], split="train", image_size=img_size, augment=True,
        )
        val_dataset = PiLoTSyntheticDataset(
            root=data_cfg.get("val_path", data_cfg["train_path"]),
            split="val", image_size=img_size, augment=False,
        )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=data_cfg["num_workers"], pin_memory=data_cfg["pin_memory"],
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=data_cfg["num_workers"], pin_memory=data_cfg["pin_memory"],
    )

    # Scheduler
    epochs = cfg["training"]["epochs"]
    total_steps = epochs * max(len(train_loader), 1)
    warmup_steps = int(total_steps * cfg["scheduler"]["warmup_ratio"])
    min_lr = cfg["scheduler"]["min_lr"]
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps, min_lr)

    # Checkpoint / early stopping
    ckpt_mgr = CheckpointManager(
        ckpt_dir, keep_top_k=cfg["checkpoint"]["keep_top_k"],
        metric=cfg["checkpoint"]["metric"], mode=cfg["checkpoint"]["mode"],
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
    from torch.utils.tensorboard import SummaryWriter
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
            query_img = batch["query_image"].to(device)
            ref_img = batch["ref_image"].to(device)
            T_query = batch["T_query"].to(device)
            T_init = batch["T_init"].to(device)
            T_ref = batch["T_ref"].to(device)
            depth = batch["depth"].to(device)
            intrinsics = batch["intrinsics"].to(device)

            # CUDA-accelerated geo-anchor generation
            geo_anchors = depth_to_geo_anchors(depth, T_ref, intrinsics, num_anchors)

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                # Feature extraction (uses Triton projection internally)
                query_feats = model.extract_features(query_img)
                ref_feats = model.extract_features(ref_img)

                # Reprojection targets (uses Triton fused kernel)
                pts_gt = project_points(geo_anchors, T_query, intrinsics)
                pts_pred = project_points(geo_anchors, T_init, intrinsics)

                loss_dict = criterion(
                    pts_pred=pts_pred,
                    pts_gt=pts_gt,
                    query_feat=query_feats[0][0],
                    ref_feat=ref_feats[0][0],
                    uncertainty=query_feats[0][1],
                )
                loss = loss_dict["total"]

            if torch.isnan(loss):
                print("[FATAL] Loss is NaN -- stopping training")
                return

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % 10 == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/lr", lr_now, global_step)
                throughput = batch_size * 10 / (time.time() - t0) if batch_idx > 0 else 0
                print(
                    f"[Epoch {epoch + 1}/{epochs}] step={global_step} "
                    f"loss={loss.item():.4f} lr={lr_now:.2e} "
                    f"throughput={throughput:.1f} img/s"
                )

            if global_step % save_every == 0:
                state = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch, "step": global_step, "config": cfg,
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
                    pts_pred=pts_pred, pts_gt=pts_gt,
                    query_feat=query_feats[0][0], ref_feat=ref_feats[0][0],
                    uncertainty=query_feats[0][1],
                )
                val_loss_sum += loss_dict["total"].item()
                val_count += 1

        val_loss = val_loss_sum / max(val_count, 1)
        print(f"[Epoch {epoch + 1}/{epochs}] val_loss={val_loss:.4f}")
        writer.add_scalar("epoch/val_loss", val_loss, epoch)

        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch, "step": global_step, "config": cfg,
        }
        ckpt_mgr.save(state, val_loss, global_step)

        if early_stop and early_stop.step(val_loss):
            break

    writer.close()
    print("[DONE] Training complete.")


def main():
    parser = argparse.ArgumentParser(description="PiLoT CUDA training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size

    print(f"[PiLoT-CUDA] Training with config: {args.config}")
    print(f"[PiLoT-CUDA] Device: {args.device}")

    train_cuda(cfg, resume=args.resume, device=args.device)


if __name__ == "__main__":
    main()
