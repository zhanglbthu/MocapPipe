#!/usr/bin/env sh
set -eu

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

INPUT_DIR="${INPUT_DIR:-data/eval/diffusionposer/imuposer/lw_rp/diffusion_full_train/steps_100}"
SEQUENCE="${SEQUENCE:-1}"
FPS="${FPS:-30}"
STRIDE="${STRIDE:-1}"
MAX_FRAMES="${MAX_FRAMES:-}"
IMAGE_WIDTH="${IMAGE_WIDTH:-1920}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-1080}"
FACE_STRIDE="${FACE_STRIDE:-1}"
SUBJECT_SPACING="${SUBJECT_SPACING:-1.5}"
BATCH_SIZE="${BATCH_SIZE:-256}"
VISUALIZE_TRAN="${VISUALIZE_TRAN:-0}"
DISPLAY_ID="${DISPLAY_ID:-1}"

XVFB_PID=""
cleanup() {
  if [ -n "$XVFB_PID" ] && kill -0 "$XVFB_PID" 2>/dev/null; then
    kill "$XVFB_PID" || true
  fi
}
trap cleanup EXIT

if ! xdpyinfo -display ":${DISPLAY_ID}.0" >/dev/null 2>&1; then
  Xvfb ":${DISPLAY_ID}" -screen 0 "${IMAGE_WIDTH}x${IMAGE_HEIGHT}x24" >/tmp/mobileposer_xvfb.log 2>&1 &
  XVFB_PID=$!
  sleep 2
fi

set -- \
  conda run -n mobileposer python visualize.py \
  --input-dir "$INPUT_DIR" \
  --sequence "$SEQUENCE" \
  --fps "$FPS" \
  --stride "$STRIDE" \
  --image-width "$IMAGE_WIDTH" \
  --image-height "$IMAGE_HEIGHT" \
  --face-stride "$FACE_STRIDE" \
  --subject-spacing "$SUBJECT_SPACING" \
  --batch-size "$BATCH_SIZE"

if [ -n "$MAX_FRAMES" ]; then
  set -- "$@" --max-frames "$MAX_FRAMES"
fi

if [ "$VISUALIZE_TRAN" = "1" ]; then
  set -- "$@" --visualize-tran
fi

DISPLAY=":${DISPLAY_ID}.0" "$@"
