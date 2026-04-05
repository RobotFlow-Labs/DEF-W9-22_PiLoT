"""PiLoT evaluation: pose error metrics, recall, completeness, FPS.

Paper metrics:
- Median translation error (m)
- Median rotation error (deg)
- Recall@{1,3,5}m and {1,3,5}deg
- Completeness (%)
- FPS
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Pose error computation
# ---------------------------------------------------------------------------

def compute_pose_error(
    T_pred: torch.Tensor,
    T_gt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute translation and rotation error between predicted and GT poses.

    Args:
        T_pred: (B, 4, 4) predicted poses.
        T_gt: (B, 4, 4) ground truth poses.

    Returns:
        t_error: (B,) translation error in meters.
        r_error: (B,) rotation error in degrees.
    """
    # Translation error: Euclidean distance
    t_pred = T_pred[:, :3, 3]
    t_gt = T_gt[:, :3, 3]
    t_error = (t_pred - t_gt).norm(dim=-1)  # (B,)

    # Rotation error: geodesic angle
    R_pred = T_pred[:, :3, :3]
    R_gt = T_gt[:, :3, :3]
    R_rel = R_gt.transpose(-1, -2) @ R_pred  # (B, 3, 3)

    # Angle from trace: cos(theta) = (trace(R) - 1) / 2
    trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
    cos_angle = ((trace - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7)
    r_error = torch.acos(cos_angle) * (180.0 / torch.pi)  # (B,) in degrees

    return t_error, r_error


def compute_recall(
    errors: np.ndarray,
    thresholds: list[float],
) -> dict[str, float]:
    """Compute recall at multiple thresholds.

    Args:
        errors: (N,) array of errors.
        thresholds: list of threshold values.

    Returns:
        dict mapping threshold -> recall percentage.
    """
    results = {}
    for t in thresholds:
        recall = (errors <= t).sum() / max(len(errors), 1) * 100.0
        results[f"recall@{t}"] = float(recall)
    return results


def compute_completeness(valid_mask: np.ndarray) -> float:
    """Compute completeness: percentage of frames with valid pose estimate.

    Args:
        valid_mask: (N,) boolean array, True if pose was successfully estimated.

    Returns:
        completeness percentage.
    """
    return float(valid_mask.sum() / max(len(valid_mask), 1) * 100.0)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

@dataclass
class EvalResults:
    """Container for evaluation results."""

    median_t_error: float = 0.0
    median_r_error: float = 0.0
    mean_t_error: float = 0.0
    mean_r_error: float = 0.0
    recall_t: dict[str, float] = field(default_factory=dict)
    recall_r: dict[str, float] = field(default_factory=dict)
    completeness: float = 0.0
    fps: float = 0.0
    num_frames: int = 0

    def to_dict(self) -> dict:
        return {
            "median_translation_error_m": self.median_t_error,
            "median_rotation_error_deg": self.median_r_error,
            "mean_translation_error_m": self.mean_t_error,
            "mean_rotation_error_deg": self.mean_r_error,
            "recall_translation": self.recall_t,
            "recall_rotation": self.recall_r,
            "completeness_pct": self.completeness,
            "fps": self.fps,
            "num_frames": self.num_frames,
        }

    def summary(self) -> str:
        lines = [
            "=== PiLoT Evaluation Results ===",
            f"Frames: {self.num_frames}",
            f"Median translation error: {self.median_t_error:.3f} m",
            f"Median rotation error: {self.median_r_error:.3f} deg",
            f"Mean translation error: {self.mean_t_error:.3f} m",
            f"Mean rotation error: {self.mean_r_error:.3f} deg",
        ]
        for k, v in self.recall_t.items():
            lines.append(f"Translation {k}: {v:.1f}%")
        for k, v in self.recall_r.items():
            lines.append(f"Rotation {k}: {v:.1f}%")
        lines.append(f"Completeness: {self.completeness:.1f}%")
        lines.append(f"FPS: {self.fps:.1f}")
        return "\n".join(lines)


class PiLoTEvaluator:
    """Accumulates pose predictions and computes metrics."""

    def __init__(
        self,
        recall_t_thresholds: list[float] | None = None,
        recall_r_thresholds: list[float] | None = None,
    ):
        self.recall_t_thresholds = recall_t_thresholds or [1.0, 3.0, 5.0]
        self.recall_r_thresholds = recall_r_thresholds or [1.0, 3.0, 5.0]
        self.t_errors: list[float] = []
        self.r_errors: list[float] = []
        self.valid: list[bool] = []
        self.timings: list[float] = []

    def reset(self):
        self.t_errors.clear()
        self.r_errors.clear()
        self.valid.clear()
        self.timings.clear()

    def update(
        self,
        T_pred: torch.Tensor,
        T_gt: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        elapsed_s: float = 0.0,
    ):
        """Add batch of predictions.

        Args:
            T_pred: (B, 4, 4) predicted poses.
            T_gt: (B, 4, 4) ground truth poses.
            valid_mask: (B,) bool, which predictions are valid.
            elapsed_s: time for this batch in seconds.
        """
        t_err, r_err = compute_pose_error(T_pred, T_gt)
        B = T_pred.shape[0]

        if valid_mask is None:
            valid_mask = torch.ones(B, dtype=torch.bool)

        for i in range(B):
            self.t_errors.append(t_err[i].item())
            self.r_errors.append(r_err[i].item())
            self.valid.append(valid_mask[i].item())

        if elapsed_s > 0:
            self.timings.append(elapsed_s / B)

    def compute(self) -> EvalResults:
        """Compute all metrics from accumulated predictions."""
        t_arr = np.array(self.t_errors)
        r_arr = np.array(self.r_errors)
        v_arr = np.array(self.valid, dtype=bool)

        # Filter valid only for error metrics
        t_valid = t_arr[v_arr] if v_arr.any() else t_arr
        r_valid = r_arr[v_arr] if v_arr.any() else r_arr

        results = EvalResults(
            median_t_error=float(np.median(t_valid)) if len(t_valid) > 0 else 0.0,
            median_r_error=float(np.median(r_valid)) if len(r_valid) > 0 else 0.0,
            mean_t_error=float(np.mean(t_valid)) if len(t_valid) > 0 else 0.0,
            mean_r_error=float(np.mean(r_valid)) if len(r_valid) > 0 else 0.0,
            recall_t=compute_recall(t_valid, self.recall_t_thresholds) if len(t_valid) > 0 else {},
            recall_r=compute_recall(r_valid, self.recall_r_thresholds) if len(r_valid) > 0 else {},
            completeness=compute_completeness(v_arr),
            fps=1.0 / np.mean(self.timings) if self.timings else 0.0,
            num_frames=len(self.t_errors),
        )
        return results


def save_results(results: EvalResults, output_dir: str, dataset_name: str = "eval"):
    """Save evaluation results to JSON and markdown."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = out / f"{dataset_name}_results.json"
    with open(json_path, "w") as f:
        json.dump(results.to_dict(), f, indent=2)

    # Markdown
    md_path = out / f"{dataset_name}_report.md"
    with open(md_path, "w") as f:
        f.write(f"# PiLoT Evaluation Report -- {dataset_name}\n\n")
        f.write(results.summary())
        f.write("\n")

    print(f"[EVAL] Results saved to {json_path}")
    print(f"[EVAL] Report saved to {md_path}")
