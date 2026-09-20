#!/usr/bin/env bash
# Real training run on an NVIDIA machine.
#
#   scripts/train_gpu.sh                              # plain task, size12m, 2e6 steps
#   SIZE=size25m scripts/train_gpu.sh --task carnav_full
#   LOGDIR=logdir/plain_12m scripts/train_gpu.sh      # re-running resumes from ckpt
#
# Pull results back to the laptop with:
#   rsync -av --exclude 'replay/' box:path/to/dreamer-car-nav/logdir/ logdir/
set -euo pipefail
cd "$(dirname "$0")/.."

SIZE="${SIZE:-size12m}"
LOGDIR="${LOGDIR:-logdir/gpu_${SIZE}_$(date +%Y%m%dT%H%M%S)}"

exec python -m carnav_dreamer.train \
  --configs carnav "$SIZE" \
  --logdir "$LOGDIR" \
  "$@"
