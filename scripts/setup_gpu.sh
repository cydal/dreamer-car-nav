#!/usr/bin/env bash
# One-shot environment bootstrap for a Linux NVIDIA box.
#
#   git clone <this repo> dreamer-car-nav && cd dreamer-car-nav
#   scripts/setup_gpu.sh            # creates conda env `dreamer-carnav`
#   conda activate dreamer-carnav
#   scripts/train_gpu.sh
#
# Assumes conda (or mamba/micromamba via $CONDA) is on PATH and the NVIDIA
# driver is installed; CUDA libraries come with the jax[cuda12] wheels.
set -euo pipefail
cd "$(dirname "$0")/.."

CONDA="${CONDA:-conda}"
ENV_NAME="${ENV_NAME:-dreamer-carnav}"

if ! $CONDA env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  $CONDA create -y -n "$ENV_NAME" python=3.11
fi

# Run the rest inside the env without requiring `conda activate` in a script.
PY="$($CONDA run -n "$ENV_NAME" python -c 'import sys; print(sys.executable)')"
"$PY" -m pip install -U pip
"$PY" -m pip install -r requirements-cuda.txt      # jax[cuda12] + rl-env3d from git
PATH="$(dirname "$PY"):$PATH" scripts/fetch_upstream.sh

"$PY" - <<'EOF'
import jax, carnav, dreamerv3.main
print("JAX devices:", jax.devices())
env = carnav.make(traffic=False, traffic_lights=False)
print("CarNav plain task vector_dim:", env.vector_dim)
EOF
echo "Done. Activate with: conda activate $ENV_NAME"
