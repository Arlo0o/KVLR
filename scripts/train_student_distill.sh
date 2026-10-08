#!/usr/bin/env bash
set -euo pipefail

# Action-adaptive few-step student distillation.
# TRAIN_STAGE: 4step_distill | 4step_budgeted | 2step_budgeted

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

TRAIN_STAGE="${TRAIN_STAGE:-4step_budgeted}"
CONFIG_4STEP_DISTILL="${CONFIG_4STEP_DISTILL:-configs/diffusion/train/student_4step.py}"
CONFIG_4STEP_BUDGETED="${CONFIG_4STEP_BUDGETED:-configs/diffusion/train/student_4step_budgeted.py}"
CONFIG_2STEP_BUDGETED="${CONFIG_2STEP_BUDGETED:-configs/diffusion/train/student_2step_budgeted.py}"

DATASET="${KASA_ACTION_CSV:-./data/kasa/annotations/surgical_rarp_action_filtered.csv}"
STUDENT_INIT_WEIGHTS="${STUDENT_INIT_WEIGHTS:-${STAGE2_CHECKPOINT:-./checkpoints/stage2_i2v}}"
TEACHER_CKPT="${TEACHER_CKPT:-./checkpoints/kvlr/merged_model.safetensors}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-./outputs/student_distill}"
OUTPUTS_DIR="${OUTPUTS_DIR:-${OUTPUTS_ROOT}/${TRAIN_STAGE}_$(date +%Y%m%d_%H%M%S)}"

NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${CUDA_VISIBLE_DEVICES:+$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
GRAD_CKPT_BUFFER_SIZE="${GRAD_CKPT_BUFFER_SIZE:-0}"

DISTILL_WEIGHT="${DISTILL_WEIGHT:-1.0}"
ROUTE_WEIGHT="${ROUTE_WEIGHT:-0.5}"
CTRL_WEIGHT="${CTRL_WEIGHT:-0.5}"
FLOW_WEIGHT="${FLOW_WEIGHT:-0.0}"
TARGET_ACTIVE_RATIO="${TARGET_ACTIVE_RATIO:-0.50}"
TARGET_REFRESH_RATIO="${TARGET_REFRESH_RATIO:-0.50}"
FULL_RATIO="${FULL_RATIO:-0.20}"
LIGHT_RATIO="${LIGHT_RATIO:-0.30}"
LIGHT_SCALE="${LIGHT_SCALE:-0.35}"
REUSE_FALLBACK_SCALE="${REUSE_FALLBACK_SCALE:-0.0}"

case "$TRAIN_STAGE" in
  4step_distill)
    STUDENT_CONFIG="$CONFIG_4STEP_DISTILL"
    NUM_DISTILL_STEPS="${NUM_DISTILL_STEPS:-4}"
    BUDGET_WEIGHT="${BUDGET_WEIGHT:-0.0}"
    TEMP_WEIGHT="${TEMP_WEIGHT:-0.0}"
    ;;
  4step_budgeted)
    STUDENT_CONFIG="$CONFIG_4STEP_BUDGETED"
    NUM_DISTILL_STEPS="${NUM_DISTILL_STEPS:-4}"
    BUDGET_WEIGHT="${BUDGET_WEIGHT:-0.05}"
    TEMP_WEIGHT="${TEMP_WEIGHT:-0.1}"
    ;;
  2step_budgeted)
    STUDENT_CONFIG="$CONFIG_2STEP_BUDGETED"
    NUM_DISTILL_STEPS="${NUM_DISTILL_STEPS:-2}"
    BUDGET_WEIGHT="${BUDGET_WEIGHT:-0.05}"
    TEMP_WEIGHT="${TEMP_WEIGHT:-0.1}"
    ;;
  *)
    echo "Unknown TRAIN_STAGE=$TRAIN_STAGE" >&2
    exit 2
    ;;
esac

for required in "$DATASET" "$STUDENT_INIT_WEIGHTS" "$TEACHER_CKPT"; do
  if [[ ! -e "$required" ]]; then
    echo "Required file or directory not found: $required" >&2
    exit 1
  fi
done

CMD=(
  python -m torch.distributed.run
  --nnodes "$NNODES"
  --node_rank "$NODE_RANK"
  --nproc_per_node "$NPROC_PER_NODE"
  --master_addr "$MASTER_ADDR"
  --master_port "$MASTER_PORT"
  scripts/diffusion/train.py
  "$STUDENT_CONFIG"
  --dataset.data-path "$DATASET"
  --model.from_pretrained "$STUDENT_INIT_WEIGHTS"
  --model.num_distill_steps "$NUM_DISTILL_STEPS"
  --adaptive_exec.full_ratio "$FULL_RATIO"
  --adaptive_exec.light_ratio "$LIGHT_RATIO"
  --adaptive_exec.light_scale "$LIGHT_SCALE"
  --adaptive_exec.reuse_fallback_scale "$REUSE_FALLBACK_SCALE"
  --distillation.enabled True
  --distillation.teacher_ckpt "$TEACHER_CKPT"
  --distillation.num_steps "$NUM_DISTILL_STEPS"
  --loss_weights.flow "$FLOW_WEIGHT"
  --loss_weights.distill "$DISTILL_WEIGHT"
  --loss_weights.route "$ROUTE_WEIGHT"
  --loss_weights.ctrl "$CTRL_WEIGHT"
  --loss_weights.budget "$BUDGET_WEIGHT"
  --loss_weights.temp "$TEMP_WEIGHT"
  --budget.target_active_ratio "$TARGET_ACTIVE_RATIO"
  --budget.target_refresh_ratio "$TARGET_REFRESH_RATIO"
  --outputs "$OUTPUTS_DIR"
  --grad_ckpt_buffer_size "$GRAD_CKPT_BUFFER_SIZE"
  --batch_size "$BATCH_SIZE"
  --grad_accum_steps "$GRAD_ACCUM_STEPS"
  --grad_clip "$GRAD_CLIP"
)

printf '%q ' "${CMD[@]}"
echo
"${CMD[@]}"
