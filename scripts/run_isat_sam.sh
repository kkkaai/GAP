#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-2026.6.6 realsense/seggpt-pre}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="$REPO_ROOT/$DATA_DIR"

if ! command -v isat-sam >/dev/null 2>&1; then
  echo "isat-sam is not installed in the current environment."
  echo "Install it with:"
  echo "  /home/w/miniconda3/envs/gap/bin/python -m pip install -r requirements/isat-sam-stage.txt"
  exit 1
fi

if [[ ! -d "$TARGET_DIR" ]]; then
  echo "Data directory not found: $TARGET_DIR"
  exit 1
fi

if [[ ! -f "$TARGET_DIR/isat.yaml" ]]; then
  cp "$REPO_ROOT/config/isat_prosthetic_hand.yaml" "$TARGET_DIR/isat.yaml"
fi

cd "$TARGET_DIR"
exec isat-sam
