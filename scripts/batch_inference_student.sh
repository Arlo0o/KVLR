#!/usr/bin/env bash
set -euo pipefail

# Batch KVLR-fast student inference from the same CSV task format used by the teacher.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="${CONFIG:-configs/diffusion/inference/768px_action_student_4step_budgeted.py}"
CHECKPOINT="${CHECKPOINT:-./checkpoints/kvlr_fast/merged_model.safetensors}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/inference/kvlr_fast}"

exec bash scripts/batch_inference_action.sh \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --outputs "$OUTPUT_DIR" \
  "$@"
