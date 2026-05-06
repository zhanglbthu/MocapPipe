#!/usr/bin/env sh
set -eu

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

RUN_DIR="${RUN_DIR:-checkpoints/diffusion_smoke_train}"
DATASET="${DATASET:-imuposer}"
COMBO="${COMBO:-lw_rp}"
NUM_STEPS="${NUM_STEPS:-2}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
WINDOW_LENGTH="${WINDOW_LENGTH:-125}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-1000}"
MODEL_DIM="${MODEL_DIM:-256}"
NUM_LAYERS="${NUM_LAYERS:-6}"
NUM_HEADS="${NUM_HEADS:-8}"
FF_DIM="${FF_DIM:-1024}"
DROPOUT="${DROPOUT:-0.1}"
BETA_START="${BETA_START:-1e-4}"
BETA_END="${BETA_END:-2e-2}"

set -- \
  conda run -n mobileposer python evaluate_diffusion.py \
  --run-dir "$RUN_DIR" \
  --dataset "$DATASET" \
  --combo "$COMBO" \
  --num-steps "$NUM_STEPS" \
  --window-length "$WINDOW_LENGTH" \
  --diffusion-steps "$DIFFUSION_STEPS" \
  --model-dim "$MODEL_DIM" \
  --num-layers "$NUM_LAYERS" \
  --num-heads "$NUM_HEADS" \
  --ff-dim "$FF_DIM" \
  --dropout "$DROPOUT" \
  --beta-start "$BETA_START" \
  --beta-end "$BETA_END"

if [ -n "$MAX_SAMPLES" ]; then
  set -- "$@" --max-samples "$MAX_SAMPLES"
fi

"$@"
