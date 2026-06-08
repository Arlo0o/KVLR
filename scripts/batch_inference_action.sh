#!/usr/bin/env bash
set -euo pipefail

# Batch KVLR teacher inference from a CSV task file.
# CSV columns: task_name,video_id,frame_num,image_path,prompt,save_file,checkpoint_override

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

TASKS_CSV="${TASKS_CSV:-./configs/inference_tasks_example.csv}"
CONFIG="${CONFIG:-configs/diffusion/inference/768px_action.py}"
CHECKPOINT="${CHECKPOINT:-./checkpoints/kvlr/merged_model.safetensors}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/inference/kvlr}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${CUDA_VISIBLE_DEVICES:+$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
CONTINUE_ON_ERROR=true
EXTRA_INFERENCE_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tasks) TASKS_CSV="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --outputs) OUTPUT_DIR="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --nproc-per-node) NPROC_PER_NODE="$2"; shift 2 ;;
    --stop-on-error) CONTINUE_ON_ERROR=false; shift ;;
    --) shift; EXTRA_INFERENCE_ARGS=("$@"); break ;;
    -h|--help)
      echo "Usage: bash scripts/batch_inference_action.sh --tasks tasks.csv --checkpoint merged_model.safetensors"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

for required in "$TASKS_CSV" "$CONFIG" "$CHECKPOINT"; do
  if [[ ! -f "$required" ]]; then
    echo "Required file not found: $required" >&2
    exit 1
  fi
done

adjust_frame_num() {
  local input="$1"
  local k=$(( (input - 1) / 4 ))
  echo $(( k * 4 + 1 ))
}

mkdir -p "$OUTPUT_DIR"
TOTAL_TASKS=$(grep -v '^\s*#' "$TASKS_CSV" | grep -v '^\s*$' | awk 'NR>1' | wc -l)
echo "Running $TOTAL_TASKS inference tasks with $NPROC_PER_NODE process(es)."

TASK_IDX=0
SUCCESS_COUNT=0
FAILED_COUNT=0

while IFS=$'\t' read -r task_name video_id frame_num image_path prompt save_file ckpt_override; do
  TASK_IDX=$((TASK_IDX + 1))
  echo "[$TASK_IDX/$TOTAL_TASKS] $task_name ($video_id)"

  if [[ ! -f "$image_path" ]]; then
    echo "  skip: reference image not found: $image_path"
    FAILED_COUNT=$((FAILED_COUNT + 1))
    continue
  fi

  CURRENT_CKPT="${ckpt_override:-$CHECKPOINT}"
  ADJUSTED_FRAMES="$(adjust_frame_num "$frame_num")"
  FINAL_SAVE_FILE="${save_file:-${OUTPUT_DIR}/${video_id}.mp4}"
  SAVE_DIR="${OUTPUT_DIR}/tmp_${video_id}_$(date +%s)"
  mkdir -p "$SAVE_DIR" "$(dirname "$FINAL_SAVE_FILE")"

  CMD=(
    "$PYTHON_BIN" -m torch.distributed.run
    --nproc_per_node "$NPROC_PER_NODE"
    --standalone
    scripts/diffusion/inference_action.py
    "$CONFIG"
    --cond_type "i2v_head"
    --prompt "$prompt"
    --ref "$image_path"
    --ckpt_path "$CURRENT_CKPT"
    --save_dir "$SAVE_DIR"
    --sampling_option.num_frames "$ADJUSTED_FRAMES"
    "${EXTRA_INFERENCE_ARGS[@]}"
  )

  printf '  '; printf '%q ' "${CMD[@]}"; echo
  if "${CMD[@]}"; then
    generated_video="$(find "$SAVE_DIR" -type f -name '*.mp4' | head -n 1)"
    if [[ -n "$generated_video" ]]; then
      mv "$generated_video" "$FINAL_SAVE_FILE"
      echo "  saved: $FINAL_SAVE_FILE"
    else
      echo "  warning: no generated mp4 found in $SAVE_DIR"
    fi
    rm -rf "$SAVE_DIR"
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
  else
    FAILED_COUNT=$((FAILED_COUNT + 1))
    [[ "$CONTINUE_ON_ERROR" == "true" ]] || break
  fi
done < <("$PYTHON_BIN" - "$TASKS_CSV" <<'PY'
import csv
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    lines = (line for line in f if line.strip() and not line.lstrip().startswith("#"))
    for row in csv.DictReader(lines):
        task = (row.get("task_name") or "").strip()
        if not task:
            continue
        print("\t".join([
            task,
            (row.get("video_id") or task).strip(),
            (row.get("frame_num") or "17").strip(),
            (row.get("image_path") or "").strip(),
            (row.get("prompt") or "").strip(),
            (row.get("save_file") or "").strip(),
            (row.get("checkpoint_override") or "").strip(),
        ]))
PY
)

echo "Done: total=$TOTAL_TASKS success=$SUCCESS_COUNT failed=$FAILED_COUNT"
[[ "$FAILED_COUNT" -eq 0 ]]
