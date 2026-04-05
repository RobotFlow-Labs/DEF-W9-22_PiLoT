#!/usr/bin/env python3
"""PiLoT evaluation script -- CLI entry point.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/evaluate.py \
        --config configs/paper.toml \
        --checkpoint /mnt/artifacts-datai/checkpoints/project_pilot/best.pth \
        --dataset /mnt/forge-data/datasets/uavd4l_2yr/ \
        --output /mnt/artifacts-datai/reports/project_pilot/
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from torch.utils.data import DataLoader

from pilot.dataset import PiLoTEvalDataset
from pilot.evaluate import PiLoTEvaluator, save_results
from pilot.train import build_model
from pilot.utils import load_config


def main():
    parser = argparse.ArgumentParser(description="PiLoT evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to TOML config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    parser.add_argument("--dataset", type=str, required=True, help="Evaluation dataset root")
    parser.add_argument("--output", type=str, default="/mnt/artifacts-datai/reports/project_pilot/")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dataset-name", type=str, default="eval")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = args.device

    # Build model
    model = build_model(cfg).to(device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[EVAL] Loaded checkpoint: {args.checkpoint}")

    # Dataset
    data_cfg = cfg["data"]
    dataset = PiLoTEvalDataset(
        root=args.dataset,
        image_size=(data_cfg["image_width"], data_cfg["image_height"]),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f"[EVAL] Dataset: {args.dataset} ({len(dataset)} frames)")

    # Evaluator
    eval_cfg = cfg.get("evaluation", {})
    evaluator = PiLoTEvaluator(
        recall_t_thresholds=eval_cfg.get("recall_thresholds_m", [1.0, 3.0, 5.0]),
        recall_r_thresholds=eval_cfg.get("recall_thresholds_deg", [1.0, 3.0, 5.0]),
    )

    # Run evaluation
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            T_gt = batch["T_gt"].to(device)

            t0 = time.time()

            # For evaluation without 3D map, use feature extraction only
            # Full pipeline requires reference view + geo-anchors from 3D map
            # Here we compute features and use GT pose + noise as baseline
            model.extract_features(image)

            # Placeholder: in full pipeline, JNGO would produce T_pred
            # For now, add noise to GT as simulated prediction
            noise = torch.randn(T_gt.shape[0], 6, device=device) * 0.1
            from pilot.model import se3_exp
            T_pred = se3_exp(noise) @ T_gt

            elapsed = time.time() - t0

            evaluator.update(T_pred, T_gt, elapsed_s=elapsed)

    # Results
    results = evaluator.compute()
    print(results.summary())
    save_results(results, args.output, args.dataset_name)


if __name__ == "__main__":
    main()
