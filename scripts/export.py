#!/usr/bin/env python3
"""PiLoT export script — all 5 formats.

Usage:
    uv run python scripts/export.py \
        --config configs/paper.toml \
        --checkpoint /mnt/artifacts-datai/checkpoints/project_pilot/best.pth \
        --output /mnt/artifacts-datai/exports/project_pilot/
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pilot.export import export_all


def main():
    parser = argparse.ArgumentParser(description="PiLoT export pipeline")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str,
                        default="/mnt/artifacts-datai/exports/project_pilot/")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=512)
    args = parser.parse_args()

    results = export_all(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_dir=args.output,
        input_height=args.height,
        input_width=args.width,
    )

    print(f"\n[DONE] Exported {len(results)} formats to {args.output}")


if __name__ == "__main__":
    main()
