#!/usr/bin/env bash
set -euo pipefail

# Stage 3: Kinematic-to-Visual Action Routing fine-tuning.
# Required:
#   KASA_ACTION_CSV      CSV with action annotation paths.
#   STAGE2_CHECKPOINT   Stage-2 I2V checkpoint.
# Optional:
#   KASA_ACTION_ROOT     Root containing label_results/*_results directories.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="${STAGE3_CONFIG:-configs/diffusion/train/stage3_action.py}"
DATASET="${KASA_ACTION_CSV:-./data/kasa/annotations/surgical_rarp_action_filtered.csv}"
INIT_WEIGHTS="${STAGE2_CHECKPOINT:-./checkpoints/stage2_i2v}"
OUTPUTS_DIR="${OUTPUTS_DIR:-./outputs/stage3_action}"

NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${CUDA_VISIBLE_DEVICES:+$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

GRAD_CKPT_BUFFER_SIZE="${GRAD_CKPT_BUFFER_SIZE:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"

if [[ ! -f "$DATASET" ]]; then
  echo "Missing KASA action CSV: $DATASET" >&2
  echo "Set KASA_ACTION_CSV or place the file under ./data/kasa/annotations/." >&2
  exit 1
fi

if [[ ! -e "$INIT_WEIGHTS" ]]; then
  echo "Missing Stage-2 checkpoint: $INIT_WEIGHTS" >&2
  echo "Set STAGE2_CHECKPOINT to a checkpoint from the anonymous model link." >&2
  exit 1
fi

CMD=(
  python -m torch.distributed.run
  --nnodes "$NNODES"
  --node_rank "$NODE_RANK"
  --nproc_per_node "$NPROC_PER_NODE"
  --master_addr "$MASTER_ADDR"
  --master_port "$MASTER_PORT"
  scripts/diffusion/train.py
  "$CONFIG"
  --dataset.data-path "$DATASET"
  --model.from_pretrained "$INIT_WEIGHTS"
  --outputs "$OUTPUTS_DIR"
  --grad_ckpt_buffer_size "$GRAD_CKPT_BUFFER_SIZE"
  --batch_size "$BATCH_SIZE"
  --grad_accum_steps "$GRAD_ACCUM_STEPS"
  --grad_clip "$GRAD_CLIP"
)

printf '%q ' "${CMD[@]}"
echo
"${CMD[@]}"
