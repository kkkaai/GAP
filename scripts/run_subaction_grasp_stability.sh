#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${STABILITY_API_KEY:-}" ]]; then
  echo "STABILITY_API_KEY is not set. Export your Stability API key first."
  echo "Example: export STABILITY_API_KEY='sk-...'"
  exit 1
fi

RGB_PATH="${1:-src/prosthetic_grasp/coffeecup.png}"
OUTPUT_DIR="${2:-outputs/run}"
CONFIG_PATH="${3:-config/default.toml}"

PYTHONPATH=src python -m prosthetic_grasp.apps.run_subaction_grasp \
  --rgb "${RGB_PATH}" \
  --config "${CONFIG_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --phase3-mode api \
  --phase3-model-name stability-erase \
  --phase4-mode api \
  --phase4-model-name stability-inpaint \
  --phase5-body-detector regnety
