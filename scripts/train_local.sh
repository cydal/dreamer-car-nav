#!/usr/bin/env bash
# Local (Apple silicon, CPU JAX) training run: pipeline validation, not a
# benchmark number. size1m is the tier upstream uses for vector-only DMC.
#
#   scripts/train_local.sh                      # plain task, 3e5 steps
#   scripts/train_local.sh --task carnav_lights # any extra flags pass through
#
# Re-running with the same --logdir resumes from its checkpoint.
set -euo pipefail
cd "$(dirname "$0")/.."

LOGDIR="${LOGDIR:-logdir/local_$(date +%Y%m%dT%H%M%S)}"
STEPS="${STEPS:-3e5}"

exec python -m carnav_dreamer.train \
  --configs carnav carnav_mac size1m \
  --run.steps "$STEPS" \
  --logdir "$LOGDIR" \
  "$@"
