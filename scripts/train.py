#!/usr/bin/env python3
"""PiLoT training script -- CLI entry point.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/train.py --config configs/paper.toml
    uv run python scripts/train.py --config configs/debug.toml --resume ckpt.pth
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure src/ is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pilot.train import train
from pilot.utils import load_config


def main():
    parser = argparse.ArgumentParser(description="PiLoT training")
    parser.add_argument("--config", type=str, required=True, help="Path to TOML config")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    parser.add_argument("--max-steps", type=int, default=None, help="Max steps (for smoke test)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Override for smoke test
    if args.max_steps is not None:
        cfg["_max_steps"] = args.max_steps

    print(f"[PiLoT] Training with config: {args.config}")
    print(f"[PiLoT] Device: {args.device}")
    if args.resume:
        print(f"[PiLoT] Resuming from: {args.resume}")

    train(cfg, resume=args.resume, device=args.device)


if __name__ == "__main__":
    main()
