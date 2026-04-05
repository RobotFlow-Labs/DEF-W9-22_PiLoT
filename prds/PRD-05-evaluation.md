# PRD-05: Evaluation

## Objective
Implement evaluation metrics and scripts matching the paper's benchmarks.

## Metrics
| Metric | Description | Paper Table |
|--------|-------------|-------------|
| Median translation error (m) | Median of per-frame position errors | Table 1-3 |
| Median rotation error (deg) | Median of per-frame orientation errors | Table 1-3 |
| Recall@1m | % of frames within 1m error | Table 1-3 |
| Recall@3m | % of frames within 3m error | Table 1-3 |
| Recall@5m | % of frames within 5m error | Table 1-3 |
| Recall@1deg | % of frames within 1 deg error | Table 1-3 |
| Recall@3deg | % of frames within 3 deg error | Table 1-3 |
| Recall@5deg | % of frames within 5 deg error | Table 1-3 |
| Completeness (%) | % of frames with valid pose estimate | Table 1 |
| FPS | Inference throughput | Table 1 |

## Deliverables

### Evaluation Module (`src/pilot/evaluate.py`)
- `compute_pose_error(pred_pose, gt_pose)`: translation + rotation error
- `compute_recall(errors, thresholds)`: recall at each threshold
- `compute_completeness(valid_mask)`: ratio of valid predictions
- `evaluate_dataset(model, dataset, config)`: full evaluation pipeline
- `PiLoTEvaluator` class with metric accumulation and reporting

### Scripts
- `scripts/evaluate.py`: CLI with --checkpoint, --dataset, --config, --device
- Outputs JSON metrics + human-readable report to /mnt/artifacts-datai/reports/

## Acceptance Criteria
- All metrics computable on synthetic test data
- Results saved as JSON + markdown report
- FPS measurement includes full pipeline (feature extraction + JNGO)
- Evaluation uses test split only (never train/val)
