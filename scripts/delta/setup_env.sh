#!/usr/bin/env bash
# One-time RackZero environment setup on NCSA Delta (run on a LOGIN node,
# which has internet access; compute nodes are for jobs only).
#
#   cd /path/to/Scrabble            # your clone of this repository
#   RACKZERO_ENV=/path/with/space/rackzero-env bash scripts/delta/setup_env.sh
#
# RACKZERO_ENV must be on a filesystem with enough quota (PyTorch + CUDA
# wheels are ~5 GB). Run `quota` on Delta to see your home/projects/work
# directories and choose one; the default below is only a placeholder.
#
# Installs (user space, no admin rights, no Delta modules required):
#   Miniforge (conda-forge) -> Python 3.11 + OpenJDK 17 (JPype needs a JDK)
#   PyTorch CUDA 12.4 wheels (A100 = sm_80) and this project in editable mode
#   the Java rules-engine JAR (python -m scrabble_gym.build)
set -euo pipefail

RACKZERO_ENV="${RACKZERO_ENV:-$HOME/rackzero-env}"
MINIFORGE="${MINIFORGE:-$(dirname "$RACKZERO_ENV")/miniforge3}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "repository : $REPO"
echo "miniforge  : $MINIFORGE"
echo "environment: $RACKZERO_ENV"

if [ ! -x "$MINIFORGE/bin/conda" ]; then
  installer="$(mktemp -d)/Miniforge3.sh"
  curl -fsSL -o "$installer" \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
  bash "$installer" -b -p "$MINIFORGE"
fi
# shellcheck disable=SC1091
source "$MINIFORGE/etc/profile.d/conda.sh"

if [ ! -d "$RACKZERO_ENV" ]; then
  conda create -y -p "$RACKZERO_ENV" -c conda-forge python=3.11 openjdk=17
fi
conda activate "$RACKZERO_ENV"

python -m pip install --upgrade pip
python -m pip install torch --index-url "$TORCH_INDEX"
python -m pip install -e "$REPO[learning]" psutil
cd "$REPO"
python -m scrabble_gym.build

python - <<'EOF'
import torch, jpype, yaml, numpy
print("torch", torch.__version__, "built for CUDA", torch.version.cuda)
print("numpy", numpy.__version__, "jpype", jpype.__version__)
EOF
java -version
echo
echo "Setup complete. In every new shell or job:"
echo "  source $MINIFORGE/etc/profile.d/conda.sh && conda activate $RACKZERO_ENV"
echo "WOW24: copy your licensed WOW24v3.zip into $REPO (it is not redistributed)."
