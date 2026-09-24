#!/usr/bin/env bash
# Real training run on an NVIDIA machine.
#
#   scripts/train_gpu.sh                              # plain task -> size1m, 2e6 steps
#   scripts/train_gpu.sh --task carnav_full           # full task -> size12m (see below)
#   SIZE=size25m scripts/train_gpu.sh --task carnav_full
#   LOGDIR=logdir/plain_12m scripts/train_gpu.sh      # re-running resumes from ckpt
#
# SIZE defaults by task if not set explicitly, per the guidance in
# carnav_dreamer/configs.yaml (size1m for plain/lights, size12m for
# traffic/full -- the latter is reasoning from task structure, not measured;
# override with SIZE= to run the ablation).
#
# Pull results back to the laptop with:
#   rsync -av --exclude 'replay/' box:path/to/dreamer-car-nav/logdir/ logdir/
set -euo pipefail
cd "$(dirname "$0")/.."

TASK="carnav_plain"
args=("$@")
for i in "${!args[@]}"; do
  if [ "${args[$i]}" = "--task" ]; then
    TASK="${args[$((i + 1))]}"
    break
  fi
done

case "$TASK" in
  carnav_plain|carnav_lights) DEFAULT_SIZE=size1m ;;
  *)                          DEFAULT_SIZE=size12m ;;
esac

SIZE="${SIZE:-$DEFAULT_SIZE}"
LOGDIR="${LOGDIR:-logdir/gpu_${TASK#carnav_}_${SIZE}_$(date +%Y%m%dT%H%M%S)}"

echo "task=$TASK size=$SIZE logdir=$LOGDIR"
exec python -m carnav_dreamer.train \
  --configs carnav "$SIZE" \
  --logdir "$LOGDIR" \
  "$@"
