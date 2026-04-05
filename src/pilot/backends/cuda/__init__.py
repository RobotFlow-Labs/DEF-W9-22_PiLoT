"""PiLoT CUDA backend — Triton + shared CUDA kernels."""

from pilot.backends.cuda.kernels import (
    batched_feature_residual,
    cuda_available,
    cuda_depth_to_geo_anchors,
    fused_transform_project,
    parallel_hypothesis_score,
)

__all__ = [
    "cuda_available",
    "fused_transform_project",
    "batched_feature_residual",
    "parallel_hypothesis_score",
    "cuda_depth_to_geo_anchors",
]
