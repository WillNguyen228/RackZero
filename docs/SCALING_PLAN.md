# RackZero scaling experiment plan

The aim is to find which changes make RackZero a **stronger Scrabble player**
per unit of compute. Bigger is not assumed to be better. Every experiment
changes **one factor** relative to a fixed medium baseline and is judged on
playing strength against fixed opponents, with confidence intervals.

## 1. Baseline

The baseline is `configs/gpu.yaml`: the medium network on one A100. All
experiments start from it with `--set` overrides, so each run records exactly
what changed (`config.yaml` in the run directory; `scripts/compare_runs.py`
prints the differing settings).

| Setting | Baseline value |
|---|---|
| network | `resnet`, 64 channels x 6 blocks, latent 256 (0.80M parameters) |
| search | `ismcts`, 32 simulations, depth 2, 16 candidates (top scores), 32 particles, root Dirichlet noise α 0.3 / ε 0.25 |
| self-play | 64 games/iteration, temperature 1.0 for 8 plies, then the most-visited move |
| replay / training | 200k positions; 8 gradient samples per new position (≈60 steps x batch 256 per iteration); Adam lr 1e-3, wd 1e-4, constant LR |
| arena | `monitor` (always accept; candidate vs previous every 2 iterations, 40 games) |
| evaluation | every 2 iterations, 40 games each vs `greedy` and `heuristic`, seed 1000 |

Runs started before 2026-11-25 (for example `gpu-20261124-235040`) used no
root noise and temperature 1.0 for the whole game. Treat the change as row 11
and do not mix those runs into comparisons against the new baseline.

Before any comparison, run the baseline **with three training seeds**
(`--set seed=1`, `2`, `3`). The spread between them is the noise floor that
any "improvement" has to exceed.

## 2. Measuring strength

* **Primary metric:** half-draw win rate against `greedy` and `heuristic` on
  the fixed evaluation seeds, with 95% Wilson intervals. Secondary metrics are
  the mean score difference (95% interval), mean game length, and the
  per-seat breakdown.
* **Final comparisons** use a separate, larger evaluation of each run's final
  checkpoint, so they are not limited by the in-training budget:

  ```bash
  python evaluate.py --checkpoint runs/<run> --opponents greedy heuristic random \
      --games 400 --workers 15 --output runs/<run>/final-eval.jsonl
  python evaluate.py --checkpoint runs/<run> --opponent search \
      --opponent-checkpoint runs/<baseline-run> --games 400 --workers 15
  ```

  Required sample sizes (half-draw win rate near 50%, 95% interval half-width):
  ±10% → 97 games, ±7% → 196, ±5% → 385 (`rackzero.stats.games_for_margin`).
  Use at least 200 games before calling a difference real, and prefer the
  head-to-head match (`--opponent search --opponent-checkpoint`), which
  measures the difference directly.
* **Cost metrics** come from `metrics.jsonl`: games/hour, seconds per decision,
  training samples/s, GPU utilization and peak memory. Also note the wall-clock
  hours to reach a given win rate.
* Do not compare in-training evaluations from runs with different
  `evaluation.seed` or `evaluation.games`.

## 3. One-factor experiments

Each row is one `--set` change applied to the baseline. Use a fixed training
budget, either the same number of iterations or the same wall-clock hours, and
state which one was used. Run the rows in order: the early rows are cheap and
inform the later ones.

| # | Factor | Values (baseline in bold) | Override | Hypothesis / what to watch |
|---|---|---|---|---|
| 1 | Self-play workers | 4, 8, **15**, 31 | `selfplay.workers=N` | Throughput only. Results are bit-identical by design (tested), so compare games/hour and CPU utilization, not strength. Pick the knee of the curve. |
| 2 | Inference mode / batch | **local**, server with batch 16/32/64 | `selfplay.inference=server`, `selfplay.inference_batch_size=N` | Throughput only. The server lost locally (mean batch 1.6); test it on Linux with 31 workers and the large network. |
| 3 | MCTS simulations | 16, **32**, 64, 128 | `search.simulations=N` | Strength vs cost per decision. Move-generation time grows linearly with simulations. |
| 4 | Network width | 32, **64**, 128 channels | `model.channels=N` | Capacity. With a small replay buffer, wider networks may overfit (watch samples_per_new_position). |
| 5 | Network depth | 3, **6**, 12 blocks | `model.blocks=N` | Receptive field: 6 blocks already cover the 15x15 board. |
| 6 | Particle count | 8, **32**, 128 | `search.particles=N` with `search.method=belief` | Particles matter mainly for the `belief` method and the belief features. Watch `belief_resets`. |
| 7 | Games per iteration | 32, **64**, 128 | `selfplay.games_per_iteration=N` | Data freshness vs number of iterations. |
| 8 | Training batch size | 128, **256**, 512 | `training.batch_size=N` (steps adapt automatically: the replay ratio stays fixed) | GPU efficiency vs optimization quality. |
| 8b | Replay ratio | 2, 4, **8**, 16 samples per new position | `training.samples_per_position=N` | Too high memorizes (value loss collapses; watch `samples_per_new_position` and the baseline curve). Too low wastes self-play data. |
| 9 | Replay capacity | 50k, **200k**, 800k | `replay.capacity=N` | Staleness vs diversity. |
| 10 | Search method | **ismcts**, belief, pimc, neural_determinization, stochastic | `search.method=M` | Use `evaluate.py --compare-methods` on one trained checkpoint first; it is cheap. |
| 11 | Exploration | off (old runs), **noise α=0.3/ε=0.25 + temperature 1 → 0 after 8 plies** | `search.root_noise_fraction=0 selfplay.temperature_moves=null` reproduces the old setting | Standard AlphaZero exploration; now the baseline. |
| 12 | Value target | **scalar**, wdl | `model.value_head=wdl` | Draws are rare in Scrabble, so expect little difference. |
| 13 | Architecture | **resnet**, cnn, legacy_mlp | `model.arch=A` | Does the spatial, move-pooling policy beat the original MLP? |
| 14 | Promotion | **monitor**, gate (80 games, wilson) | `arena.mode=gate arena.games=80 arena.criterion=wilson` | Gating with small arenas mostly rejects for noise. |
| 15 | LR schedule | **constant**, cosine with warmup | `training.lr_schedule=cosine training.warmup_steps=500` | |
| 16 | Gumbel root search | **puct**, gumbel | `search.root_selection=gumbel` | Gumbel AlphaZero keeps improving the policy with few simulations. Compare at 16 and 32 simulations. |
| 17 | Reanalyse | **0**, 0.25, 0.5 | `reanalyse.fraction=F` | Costs about F × self-play search time (CPU), not GPU time. Judge strength per wall-clock hour. |
| 18 | Playout cap randomization | **off**, p=0.25 full 64 / fast 16 | `selfplay.full_search_probability=0.25 selfplay.full_simulations=64 selfplay.fast_simulations=16` | Same average cost; policy targets from 2x deeper searches. |
| 19 | Value target | **outcome**, 0.5 outcome + 0.5 search value | `training.value_target_q_weight=0.5` | Lower-variance value target. Watch value loss and margin. |
| 20 | Shortlist | **score**, mixed | `search.shortlist=mixed` | Quackle's move was in the top-16-by-score shortlist 84.3% of the time and in the mixed shortlist 88.3% (4,000 games). |
| 21 | Warm start | **none**, Quackle pre-training | `--set init_from=<pretrained.pt>` | The largest expected effect early on. Compare strength after the same number of self-play iterations. |

After rows 3–5, run `configs/large.yaml` once; it changes the network size
and batch only. `configs/warm.yaml` combines rows 18–21. Because it changes
several factors at once, run rows 18–21 one at a time as well before
attributing its result to any single one of them.

### Example commands

```bash
# Baseline seeds
for s in 1 2 3; do
  sbatch --account=$ACCT --partition=gpuA100x4 --time=12:00:00 \
    --export=ALL,RACKZERO_ACTIVATE="$ACTIVATE" \
    scripts/delta/train.sbatch configs/gpu.yaml --set seed=$s --set name=base-s$s
done
# One factor: simulations
for n in 16 64 128; do
  sbatch --account=$ACCT --partition=gpuA100x4 --time=12:00:00 \
    --export=ALL,RACKZERO_ACTIVATE="$ACTIVATE" \
    scripts/delta/train.sbatch configs/gpu.yaml --set search.simulations=$n --set name=sims$n
done
python scripts/compare_runs.py runs/base-s* runs/sims*
```

## 4. Record for every experiment

`compare_runs.py` collects most of this automatically from each run directory.

| Field | Source |
|---|---|
| configuration and changed setting | `config.yaml`; `compare_runs.py` "Settings that differ" |
| hardware and software | `environment.json` (GPU, CUDA, torch, git commit, Slurm job) |
| training time | `metrics.jsonl` `seconds`, `counters.wall_seconds` |
| GPU utilization and memory | `hardware.training.gpu_util_mean`, `training.peak_gpu_memory_mb`, `runs/gpu-<job>.csv` |
| throughput | `selfplay.games_per_hour`, `decision_seconds_mean`, `training.samples_per_second` |
| strength | `baselines.<opponent>.win_rate`, `win_rate_ci95`, `mean_score_difference`, and the final 400-game evaluation |
| improvement vs baseline | head-to-head `evaluate.py --opponent search --opponent-checkpoint` |

## 5. Expected bottlenecks (measured)

Profiling (docs/A100_TRAINING.md §7) shows self-play is CPU-bound. For the
medium configuration, a search decision takes 1.55 s: 66% of it is move
generation, 22% network inference and 9% simulator transitions. Training on
the GPU takes seconds per iteration, while self-play takes minutes. In
practice:

* games/hour scales with **CPU workers**, not with the GPU. Request as many
  cores as the GPU allocation allows.
* `search.simulations` and `search.depth` multiply self-play cost almost
  linearly. Network size adds roughly 5 ms (medium) to 20 ms (large) per CPU
  inference call.
* the A100 is mostly idle during self-play. Its value is fast training of large
  networks and large batches, and possibly batched inference for large
  networks (row 2).
