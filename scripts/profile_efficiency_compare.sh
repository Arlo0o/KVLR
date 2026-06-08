#!/usr/bin/env bash
set -euo pipefail

# Compare wall-clock inference cost for the KVLR teacher and KVLR-fast student.
# The wrapper uses the regular inference entrypoint so the measured path matches reproduction.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

TASKS_CSV="${TASKS_CSV:-./configs/inference_tasks_example.csv}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs/efficiency_profile}"

TEACHER_CONFIG="${TEACHER_CONFIG:-configs/diffusion/inference/768px_action_teacher_full.py}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-./checkpoints/kvlr/merged_model.safetensors}"
STUDENT_CONFIG="${STUDENT_CONFIG:-configs/diffusion/inference/768px_action_student_4step_budgeted.py}"
STUDENT_CHECKPOINT="${STUDENT_CHECKPOINT:-./checkpoints/kvlr_fast/merged_model.safetensors}"

mkdir -p "$OUTPUT_ROOT"

run_profile() {
  local name="$1"
  local config="$2"
  local checkpoint="$3"
  local output_dir="$4"
  local json_path="$5"

  local start end elapsed status
  start="$(date +%s)"
  set +e
  PYTHON_BIN="$PYTHON_BIN" NPROC_PER_NODE="$NPROC_PER_NODE" \
    bash scripts/batch_inference_action.sh \
      --tasks "$TASKS_CSV" \
      --config "$config" \
      --checkpoint "$checkpoint" \
      --outputs "$output_dir"
  status="$?"
  set -e
  end="$(date +%s)"
  elapsed="$((end - start))"

  "$PYTHON_BIN" - "$json_path" "$name" "$config" "$checkpoint" "$output_dir" "$elapsed" "$status" <<'PY'
import json
import os
import sys

path, name, config, checkpoint, output_dir, elapsed, status = sys.argv[1:]
payload = {
    "name": name,
    "config": config,
    "checkpoint": checkpoint,
    "output_dir": output_dir,
    "elapsed_seconds": int(elapsed),
    "exit_status": int(status),
    "num_videos": len([f for f in os.listdir(output_dir) if f.endswith(".mp4")]) if os.path.isdir(output_dir) else 0,
}
os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
with open(path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
print(json.dumps(payload, indent=2))
PY
  return "$status"
}

run_profile "KVLR teacher" "$TEACHER_CONFIG" "$TEACHER_CHECKPOINT" "$OUTPUT_ROOT/teacher" "$OUTPUT_ROOT/teacher_profile.json"
run_profile "KVLR-fast student" "$STUDENT_CONFIG" "$STUDENT_CHECKPOINT" "$OUTPUT_ROOT/student" "$OUTPUT_ROOT/student_profile.json"

"$PYTHON_BIN" scripts/diffusion/summarize_efficiency_profiles.py \
  "$OUTPUT_ROOT/teacher_profile.json" \
  "$OUTPUT_ROOT/student_profile.json" \
  --output "$OUTPUT_ROOT/summary.json"
