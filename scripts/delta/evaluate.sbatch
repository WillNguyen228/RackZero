#!/usr/bin/env bash
# Large evaluation of one RackZero run's checkpoints on NCSA Delta.
#
#   sbatch --account=<GPU_ACCOUNT> --partition=gpuA100x4 --time=03:00:00 \
#          --export=ALL,RACKZERO_ACTIVATE="<activate command>" \
#          scripts/delta/evaluate.sbatch runs/<run-dir> [iterations...] [-- reference]
#
# Defaults: iterations "0006 0012 0016 0018 0020", reference iteration 0002.
# For each iteration: GAMES games vs greedy and vs heuristic (seat-balanced
# pairs, fixed seed EVAL_SEED, different from the in-training seed 1000).
# Then a head-to-head match of the last iteration vs the reference.
# Results are appended to <run-dir>/eval-<GAMES>.jsonl; summaries go to the
# Slurm log runs/eval-<jobid>.out.
#
# Evaluation is CPU-bound (search + move generation run in CPU workers), so the
# networks are evaluated on the CPU; the GPU allocation is used for its cores.
#
#SBATCH --job-name=rackzero-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --gpus-per-node=1
#SBATCH --output=runs/eval-%j.out
set -euo pipefail

if [ -n "${RACKZERO_ACTIVATE:-}" ]; then
  eval "$RACKZERO_ACTIVATE"
fi
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

RUN="${1:?usage: evaluate.sbatch runs/<run-dir> [iterations...] [-- reference]}"; shift
ITERS=(); REF=0002
while [ $# -gt 0 ]; do
  if [ "$1" = "--" ]; then REF="$2"; shift 2; else ITERS+=("$1"); shift; fi
done
[ ${#ITERS[@]} -gt 0 ] || ITERS=(0006 0012 0016 0018 0020)
GAMES="${GAMES:-200}"
EVAL_SEED="${EVAL_SEED:-5000}"
H2H_SEED="${H2H_SEED:-7000}"
WORKERS=$(( ${SLURM_CPUS_PER_TASK:-16} > 1 ? ${SLURM_CPUS_PER_TASK:-16} - 1 : 1 ))
OUT="$RUN/eval-$GAMES.jsonl"
CK="$RUN/checkpoints"

for it in "${ITERS[@]}" "$REF"; do
  [ -f "$CK/iteration_$it.pt" ] || { echo "missing $CK/iteration_$it.pt" >&2; exit 1; }
done
echo "job ${SLURM_JOB_ID:-local} on $(hostname): $RUN, iterations ${ITERS[*]}, reference $REF, $GAMES games, $WORKERS workers"

for it in "${ITERS[@]}"; do
  echo; echo "== iteration $it vs greedy, heuristic"
  python evaluate.py --checkpoint "$CK/iteration_$it.pt" --opponents greedy heuristic \
    --games "$GAMES" --seed "$EVAL_SEED" --workers "$WORKERS" --device cpu --output "$OUT"
done

LAST="${ITERS[${#ITERS[@]}-1]}"
echo; echo "== head-to-head: iteration $LAST vs iteration $REF"
python evaluate.py --checkpoint "$CK/iteration_$LAST.pt" --opponent search \
  --opponent-checkpoint "$CK/iteration_$REF.pt" \
  --games "$GAMES" --seed "$H2H_SEED" --workers "$WORKERS" --device cpu --output "$OUT"

echo; echo "Done. Results: $OUT"
