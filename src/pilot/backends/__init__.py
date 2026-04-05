"""PiLoT backends — auto-detect CUDA > CPU."""

from __future__ import annotations

import torch


def get_backend() -> str:
    """Detect best available backend."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
