#!/usr/bin/env bash
# Validate the GPU training pipeline INSIDE a GPU allocation (srun --pty or sbatch).
# Runs from the repository root; activate the environment first:
#
#   source <miniforge>/etc/profile.d/conda.sh && conda activate <env>
#   bash scripts/delta/gpu_check.sh            # ~10-20 minutes on one A100
#
# Every step prints its exact command. Outputs go to runs/delta-check-<jobid>/.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
OUT="runs/delta-check-${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT"
step() { echo; echo "+ $*"; "$@"; }

echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  cpus ${SLURM_CPUS_PER_TASK:-?}"
step nvidia-smi
step python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print('bf16', torch.cuda.is_bf16_supported())"
step java -version

# 1. Correctness: the RackZero test suites (CPU; Java engine; no GPU needed).
step python -m unittest discover -s tests/python -p "test_rackzero*.py"

# 2. Smoke training on the GPU: bf16 training, CUDA RNG checkpointing, resume.
step python train.py --config configs/smoke.yaml --run-dir "$OUT/smoke-cuda" \
  --set runtime.device=cuda --set runtime.precision=bf16 --set logging.hardware_interval=1
step python train.py --resume "$OUT/smoke-cuda" --set iterations=2
step python evaluate.py --checkpoint "$OUT/smoke-cuda" --games 4 --opponents greedy random --device cuda

# 3. Component throughput: training (bf16 and fp32) and batched inference.
if [ -f WOW24v3.zip ]; then DICT=WOW24v3.zip; else DICT=tiny; fi
step python scripts/benchmark.py training inference --dictionary "$DICT" --device cuda \
  --precision bf16 --output "$OUT/benchmark.jsonl"
step python scripts/benchmark.py training --dictionary "$DICT" --device cuda \
  --precision fp32 --sizes medium --output "$OUT/benchmark.jsonl"
step python scripts/benchmark.py inference --dictionary "$DICT" --device cpu \
  --batch-sizes 1 --output "$OUT/benchmark.jsonl"
if [ "$DICT" = WOW24v3.zip ]; then
  step python scripts/benchmark.py movegen search --dictionary "$DICT" \
    --config configs/medium.yaml --output "$OUT/benchmark.jsonl"
  # 4. A short real GPU training iteration (WOW24, medium network, parallel workers).
  WORKERS=$(( ${SLURM_CPUS_PER_TASK:-4} > 1 ? ${SLURM_CPUS_PER_TASK:-4} - 1 : 1 ))
  step python train.py --config configs/gpu.yaml --run-dir "$OUT/gpu-iteration" \
    --set iterations=1 --set selfplay.games_per_iteration=$((2 * WORKERS)) \
    --set selfplay.workers=$WORKERS --set training.min_replay=1 \
    --set evaluation.every=1 --set evaluation.games=8 --set arena.every=1 --set arena.games=8
fi
echo
echo "Done. Results: $OUT  (metrics.jsonl / benchmark.jsonl / environment.json)"
