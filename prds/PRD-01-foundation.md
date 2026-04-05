# PRD-01: Foundation

## Objective
Set up project scaffolding, package structure, configs, venv, and CI-ready tooling.

## Deliverables
- [x] `pyproject.toml` with hatchling backend, torch cu128, ruff config
- [x] `src/pilot/__init__.py` package skeleton
- [x] `configs/paper.toml` and `configs/debug.toml`
- [x] `anima_module.yaml` module manifest
- [x] `CLAUDE.md`, `ASSETS.md`, `PRD.md`
- [x] `.env.serve` template
- [x] Test infrastructure: `tests/test_model.py`, `tests/test_dataset.py`
- [x] Scripts: `scripts/train.py`, `scripts/evaluate.py`

## Acceptance Criteria
- `uv sync` succeeds in isolated .venv
- `uv run pytest tests/ -x` passes (placeholder tests)
- `ruff check src/` passes with zero errors
- All config TOML files parse without error

## Notes
- Torch cu128 index URL in pyproject.toml
- Line length 100, ruff selects E,F,I,B,UP,N,C4
