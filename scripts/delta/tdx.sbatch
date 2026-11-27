#!/usr/bin/env bash
# tdx on NCSA Delta: TD(lambda) self-play FROM SCRATCH (no Quackle warm start),
# small 32x4 network, pure Q play. Same settings as runs/tdx_launch.sh on
# rcs-gpu2, but workers = allocated cores - 1. Updated 2026-11-27 with the
# rcs-gpu2 experiments:
#   - exact endgame playouts (bag empty) kept: the biggest measured gain
#     (+4 to +5 points vs Quackle's static player with no training)
#   - pre-endgame CRN lookahead dropped (crn_bag=0): +0.8 points, within noise,
#     and the same lookahead hurt in mid-game (-9 to -13); it cost ~45% of self-play
#   - 32 candidates (mixed shortlist) instead of 16: Quackle's move is in the
#     16-list 88.5% of the time, in the 32-list 93.4%
#   - luck-adjusted TD targets (8 resampled draws): td4's Q became calibrated
#     (overestimate ~0 vs td-q's +1.9) and picked the best moves in rollouts
#
# Self-play is CPU-bound (local CPU inference); the GPU only does the training
# steps. Games/hour scales with --cpus-per-task.
#
# New run (submit from the repository root):
#   sbatch --account=$ACCT --partition=gpuA100x4 --time=48:00:00 --cpus-per-task=32 --mem=96g --gres=gpu:1 \
#          --export=ALL,RACKZERO_ACTIVATE="$ACTIVATE" scripts/delta/tdx.sbatch
#
# Continue after the time limit (picks up from checkpoints/latest.pt):
#   sbatch ... scripts/delta/tdx.sbatch --resume runs/tdx-<date>
#
# Smoke test (a few minutes): add --export=...,ITERS=2,GAMES=64,STEPS=20,CKPT=1,NAME=tdxsmoke
#
#SBATCH --job-name=rackzero-tdx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96g
#SBATCH --gpus-per-node=1
#SBATCH --output=runs/tdx-%j.out
set -euo pipefail

if [ -n "${RACKZERO_ACTIVATE:-}" ]; then
  eval "$RACKZERO_ACTIVATE"
fi
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p runs

WORKERS=$(( ${SLURM_CPUS_PER_TASK:-16} > 1 ? ${SLURM_CPUS_PER_TASK:-16} - 1 : 1 ))
echo "job ${SLURM_JOB_ID:-local} on $(hostname), $WORKERS self-play workers, start $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

if [ "${1:-}" = "--resume" ]; then
  RUN="$2"; shift 2
  exec python train.py --resume "$RUN" --set selfplay.workers=$WORKERS "$@"
fi

NAME=${NAME:-tdx}; ITERS=${ITERS:-100}; GAMES=${GAMES:-3000}; STEPS=${STEPS:-360}
exec python train.py --config configs/gpu.yaml --set name=$NAME --set output_root=runs \
  --set iterations=$ITERS --set runtime.precision=bf16 \
  --set model.channels=32 --set model.blocks=4 --set model.latent=128 \
  --set model.rack_dim=32 --set model.belief_dim=32 --set model.move_dim=64 \
  --set model.policy_hidden=128 --set model.value_hidden=128 --set model.q_head=true \
  --set search.candidates=32 --set search.shortlist=mixed --set search.root_selection=q \
  --set search.q_prior_weight=0.0 --set search.q_temperature=2.0 --set search.root_noise_fraction=0 \
  --set search.endgame_playout=true --set search.endgame_replies=2 \
  --set search.crn_bag=0 \
  --set selfplay.workers=$WORKERS --set selfplay.inference=local \
  --set selfplay.games_per_iteration=$GAMES \
  --set selfplay.full_search_probability=1.0 --set selfplay.luck_samples=8 \
  --set selfplay.td_lambda=0.7 --set selfplay.explore_epsilon=0.1 \
  --set replay.capacity=500000 --set training.samples_per_position=2 \
  --set training.batch_size=512 --set training.learning_rate=0.001 \
  --set training.lr_schedule=cosine --set training.lr_min_factor=0.05 \
  --set training.steps_per_iteration=$STEPS \
  --set training.value_target_q_weight=0 --set training.q_loss_weight=1.0 \
  --set evaluation.every=0 --set arena.mode=off --set checkpoint.every=${CKPT:-5} \
  "$@"
