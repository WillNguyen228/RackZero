#!/usr/bin/env bash
# RackZero training job for one A100 on NCSA Delta.
#
# Submit from the repository root. Account, partition and time are given on the
# command line, so nothing site-specific is hard-coded here:
#
#   sbatch --account=<GPU_ACCOUNT> --partition=gpuA100x4 --time=08:00:00 \
#          --export=ALL,RACKZERO_ACTIVATE="<activate command>" \
#          scripts/delta/train.sbatch configs/gpu.yaml [--set key=value ...]
#
# Resume an existing run (same script; the run directory holds the config):
#
#   sbatch --account=<GPU_ACCOUNT> --partition=gpuA100x4 --time=08:00:00 \
#          --export=ALL,RACKZERO_ACTIVATE="<activate command>" \
#          scripts/delta/train.sbatch --resume runs/<run-dir> [--set iterations=80]
#
# RACKZERO_ACTIVATE is the command that activates your environment, e.g.
#   "source /path/miniforge3/etc/profile.d/conda.sh && conda activate /path/rackzero-env"
#
#SBATCH --job-name=rackzero
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --gpus-per-node=1
#SBATCH --output=runs/slurm-%j.out
set -euo pipefail

if [ -n "${RACKZERO_ACTIVATE:-}" ]; then
  eval "$RACKZERO_ACTIVATE"
fi
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p runs

echo "job ${SLURM_JOB_ID:-?} on $(hostname), ${SLURM_CPUS_PER_TASK:-?} CPUs"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0))"

# Self-play workers: one per allocated core, leaving one for the trainer.
WORKERS=$(( ${SLURM_CPUS_PER_TASK:-16} > 1 ? ${SLURM_CPUS_PER_TASK:-16} - 1 : 1 ))

# Independent GPU utilization log (the trainer also records nvidia-smi samples
# per phase in metrics.jsonl under "hardware").
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used \
  --format=csv -l 30 > "runs/gpu-${SLURM_JOB_ID:-local}.csv" &
MONITOR=$!
trap 'kill $MONITOR 2>/dev/null || true' EXIT

if [ "${1:-}" = "--resume" ]; then
  RUN="$2"; shift 2
  python train.py --resume "$RUN" --set selfplay.workers=$WORKERS "$@"
else
  CONFIG="$1"; shift
  python train.py --config "$CONFIG" --set selfplay.workers=$WORKERS "$@"
fi
