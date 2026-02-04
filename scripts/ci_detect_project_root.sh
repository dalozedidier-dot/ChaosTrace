#!/usr/bin/env bash
set -euo pipefail

echo "[ci] pwd: $(pwd)"
echo "[ci] listing root:"
ls -la

# Find first pyproject.toml (max depth 4) and set project root
PROJECT_ROOT="$(find . -maxdepth 4 -name pyproject.toml -print -quit || true)"

if [[ -z "${PROJECT_ROOT}" ]]; then
  echo "[ci][ERROR] No pyproject.toml found within maxdepth=4."
  echo "[ci] Showing tree (depth 3):"
  command -v tree >/dev/null 2>&1 && tree -L 3 || find . -maxdepth 3 -print
  exit 2
fi

PROJECT_DIR="$(dirname "${PROJECT_ROOT}")"
PROJECT_DIR="${PROJECT_DIR#./}"

if [[ -z "${PROJECT_DIR}" ]]; then
  PROJECT_DIR="."
fi

echo "[ci] Detected pyproject.toml at: ${PROJECT_ROOT}"
echo "[ci] Using project dir: ${PROJECT_DIR}"

cd "${PROJECT_DIR}"

echo "[ci] In project dir: $(pwd)"
echo "[ci] Install deps"
python -m pip install -U pip
pip install -e ".[dev]"

echo "[ci] Lint (ruff)"
python -m ruff check .

echo "[ci] Tests"
python -m pytest -q --cov=chaostrace --cov-report=term-missing || python -m pytest -q

echo "[ci] Smoke sweep"
mkdir -p _ci_out/smoke
python -m chaostrace.cli.run_sweep \
  --input test_data/sample_timeseries.csv \
  --out _ci_out/smoke \
  --runs "${RUNS:-3}" \
  --seed "${SEED:-7}"

echo "[ci] Done"
