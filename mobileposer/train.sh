#!/usr/bin/env sh
set -eu

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

RUN_NAME="${RUN_NAME:-diffusion_smoke_train}"
SAVE_DIR="${SAVE_DIR:-checkpoints}"
DEVICE="${DEVICE:-0}"
EPOCHS="${EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LR="${LR:-1e-3}"
WINDOW_LENGTH="${WINDOW_LENGTH:-125}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-1000}"
MODEL_DIM="${MODEL_DIM:-256}"
NUM_LAYERS="${NUM_LAYERS:-6}"
NUM_HEADS="${NUM_HEADS:-8}"
FF_DIM="${FF_DIM:-1024}"
DROPOUT="${DROPOUT:-0.1}"
BETA_START="${BETA_START:-1e-4}"
BETA_END="${BETA_END:-2e-2}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-1.0}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-1.0}"

conda run -n mobileposer python train_diffusion.py \
  --accelerator gpu \
  --device "$DEVICE" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --lr "$LR" \
  --save-dir "$SAVE_DIR" \
  --run-name "$RUN_NAME" \
  --limit-train-batches "$LIMIT_TRAIN_BATCHES" \
  --limit-val-batches "$LIMIT_VAL_BATCHES" \
  --window-length "$WINDOW_LENGTH" \
  --diffusion-steps "$DIFFUSION_STEPS" \
  --model-dim "$MODEL_DIM" \
  --num-layers "$NUM_LAYERS" \
  --num-heads "$NUM_HEADS" \
  --ff-dim "$FF_DIM" \
  --dropout "$DROPOUT" \
  --beta-start "$BETA_START" \
  --beta-end "$BETA_END"
