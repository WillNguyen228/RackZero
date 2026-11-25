# RackZero audit, architecture upgrade and scaling report

Date: 2026-11-24. Scope: the RackZero Python pipeline (`python/rackzero`),
its configurations, tests and documentation. The Java rules engine, the
dictionary, the tile distribution, scoring and termination rules were **not**
changed, and neither were the training targets.

## 1. Audit of the starting state

All suites were run against the working tree as found (before any change):

| Check | Result |
|---|---|
| `test_rackzero*.py` | 27/28 pass, **1 error**: `test_pass_leaves_rack_hypotheses_unchanged` used `belief.known/unknown/bag_only`, which the rewritten belief no longer has |
| `test_environment.py` / `test_dqn.py` / `test_quackle.py` | 17 / 18 / 13 pass |
| Smoke training (`configs/smoke.yaml`) | completes: 2 natural games, 54 positions, loss 2.998, arena 0.5 (rejected), greedy 0/2 |
| Smoke evaluation | vs greedy 0/4 (margin −68.8); vs random 4/4 (+158.0) |

Findings:

1. **Belief system (recently rewritten): mathematically correct.** I
   re-derived every update of the new Rao-Blackwellized filter. They are
   exact under the default opponent model: own draws (multivariate
   hypergeometric over the groups' bag parts), own exchanges (draw before
   return), opponent placements, exchanges and passes, and the importance
   weight when an opponent likelihood is used. The stale parts were one
   test and the documentation, which still described the old "repair
   proposal" filter. Both are fixed. Particle degeneracy is still possible
   but rare (measured: 3 resets in 1,558 updates at 8 particles, none at 32
   or 128).
2. **Tile accounting, exchanges, draws, passes and hidden information:**
   correct. Tile conservation is checked after every belief update and
   simulator step. Exchanges reveal only their count. The draw-before-return
   rule is modeled. The coordinator never hands one player's view to the
   other.
3. **The bottleneck is CPU move generation, not the network.** Profiling a
   WOW24 search decision showed ~94% of the time in Python legal-move
   generation inside the simulator and ~1% in inference.
4. **Training was not batched.** It ran one forward pass per sample in a
   Python loop. Replay was a list of dicts of float features (~2 KB per
   candidate), and the whole list was pickled into every checkpoint.
5. **Promotion was unreliable at the configured sizes.** With 4–8 arena
   games, the standard error of the win rate is 0.18–0.25. A candidate that is
   no better passes a 0.55 threshold 25–36% of the time (binomial: 2, 4 and 8
   games), and every rejection discards that iteration's training. `gate` mode
   is kept. The new configs default to `monitor` with 40-game arenas, and
   gating is listed as an experiment.
6. Other gaps: no parallel self-play, no confidence intervals, no learning
   rate schedule or AMP, no root noise or temperature schedule, flat configs
   without overrides, and no run directories or hardware metrics.

## 2. What changed

| Area | Change | Preserved |
|---|---|---|
| Move generation | `game/movegen.py`: the same algorithm with per-board cached anchors and cross-checks, and incremental scoring. Output is **identical** (moves, order, scores) to the reference: checked by a unit test and on WOW24 (113 positions, 65,797 moves). Decisions 2.2× faster in search. | Live games still use the Java-scored reference generator (the authority); `search.movegen: reference` |
| Features | `features.py`: compact integer records (≈1.4 KB/position at K=16), built only from `InformationState`, the observer's belief and candidates. Belief summary = posterior mean + P(holds ≥1) per tile | — |
| Network | `models/networks.py`: `resnet`/`cnn` with board, rack, belief and global encoders, a shared latent, a legal-move scoring policy head (board features pooled over placed squares and the main word, plus move features, plus the latent; optional candidate attention), and scalar/WDL value and margin heads. Every dimension is configurable | Original `PolicyValueNet` unchanged as `legacy_mlp`; its record features equal `encode()` to 1e-6 |
| Search | root Dirichlet noise (self-play only, default off), selectable movegen, rich root belief features, timing counters | All five methods unchanged; with noise off, search is bit-identical to before |
| Self-play | `workers.py`: per-game seeds; spawned worker pool, each process with its own JVM and lexicon; temperature schedule; optional batched GPU inference server | Replay is bit-identical for any worker count (tested) |
| Training | ring replay buffer; batched losses with configurable weights; Adam/AdamW/SGD; constant/cosine/step LR with warmup; gradient clipping; bf16/fp16 AMP; one host sync per iteration | Same targets (visit policy, win sign, margin/100) and loss (`CE + MSE + MSE`, weights 1) |
| Promotion | `arena.mode`: `gate` (original promote/roll back, threshold or Wilson criterion), `monitor`, `off` | `gate` keeps the original model+optimizer rollback |
| Checkpoints | v2: model, best, optimizer, **scheduler**, **AMP scaler**, replay, counters, history, config, fingerprint, all RNGs. Resume classifies config changes (neutral/numeric/trajectory). Run directories | v1 checkpoints still **evaluate** (verified: same games as the original code) |
| Evaluation | Wilson and mean intervals, game length, per-seat results, multiple opponents, previous-checkpoint and alternative-search opponents, parallel workers | Paired seeds, swapped seats, whole-pair discards |
| Logging | `metrics.jsonl`, `metrics.csv`, optional TensorBoard, `environment.json`, per-phase GPU/CPU sampling via `nvidia-smi`/`psutil` | — |
| Configs | `base`, `smoke`, `small`, `medium`, `large`, `cpu`, `gpu`, with `extends:` and `--set` | Original flat `default.yaml` loads unchanged as the legacy baseline |

## 3. Test results (after the final code change)

`python -m unittest discover -s tests/python` was run after the final code
change: **103 tests, all pass** (85 s, CPU). This covers the RackZero
suites plus the existing environment, DQN and Quackle suites. Separately:

* `scripts/verify_movegen.py --dictionary WOW24v3.zip --games 4`: 113
  positions and 65,797 candidate moves are identical across fast, reference
  and the Java authority.
* `train.py --config configs/smoke.yaml`, resume, and `evaluate.py` with
  multiple opponents, parallel workers and an alternative-search opponent all
  complete, on CPU and on CUDA (bf16, inference server, TensorBoard).
* A schema-1 checkpoint from the original trainer evaluates under the new
  code. It reproduces the original code's results exactly: vs greedy 0/4 at
  −68.75, vs random 4/4 at +158.0.
* `scripts/delta/gpu_check.sh` ran end to end on the laptop GPU (CUDA
  PyTorch env on `PATH`). It completed the CUDA/bf16 checks, 55 RackZero
  tests, CUDA smoke training, resume, CUDA evaluation, the benchmarks, and a
  real WOW24 `gpu.yaml` iteration. That run showed its final step skipped
  training (`min_replay` 256 > 182 positions) and evaluation (`every: 2`).
  The step now overrides both, and it was re-run successfully with training,
  arena and baselines. The Slurm/`srun`/`sbatch` wrappers and
  `setup_env.sh` were syntax-checked only.

Fixes found and made during this work, beyond the planned changes:

* the stale belief test;
* YAML reading `off` as false and `3e-4` as a string;
* Windows timer granularity throttling the inference server;
* per-step pinned-memory and host syncs in training (95 → 13k samples/s);
* psutil's first sample reading 0%;
* transient Windows file locks on checkpoint renames;
* pair-granular evaluation tasks idling workers;
* the replay-ratio overfitting described in §4.

## 4. Measured performance

All numbers below are from a Windows laptop: RTX 4070 Laptop GPU (8 GB),
PyTorch 2.11 + CUDA 12.8, 32 logical CPUs, 16 GB RAM, WOW24.

**Search decision (medium config, 1 thread):** 1.55 s per decision, of which
66% is move generation, 22% inference (5.2 ms per call, 65 calls) and 9%
simulator transitions.

| Network | Params | CPU inference / call | GPU training samples/s (bf16, batch 256 / 512 / 1024) | Peak GPU memory @512 |
|---|---|---|---|---|
| small | 0.17–0.20M | 2.7 ms | 13.9k / 30.4k / 36.2k | 0.23 GB |
| medium | 0.80M | 5.5 ms | 12.1k / 14.1k / 13.2k | 0.56 GB |
| large | 4.32M | 19.5 ms | 4.2k / 3.7k / 3.8k | 1.56 GB |
| legacy_mlp | 0.94M | 1.3 ms | — | — |

Batched inference server, large network, 8 workers, 8 WOW24 games:
4.01 s/move with the server vs 3.71 s/move with local CPU inference. The
mean server batch was only 1.56, and GPU utilization during self-play was 22%.
On this machine the server does not pay off. Linux on Delta has to be measured.

### Full-dictionary training runs (pipeline validation, not strength claims)

`configs/gpu.yaml` (medium network) with 32 games per iteration, 8 CPU
self-play workers, bf16 training on the laptop GPU, and 20-game in-training
evaluations.

| Iteration | Positions | Games/h | s/move | Train steps | Value loss | vs greedy (95% CI) | vs heuristic |
|---|---|---|---|---|---|---|---|
| 1 | 1013 | 306 | 2.12 | 32 | 0.736 | 0.40 [0.22, 0.61], −64 | 0.20 [0.08, 0.42], −77 |
| 2 | 1026 | 303 | 2.02 | 33 | 0.355 | 0.33 [0.16, 0.54], −78 | 0.33 [0.16, 0.54], −64 |
| 3 (after `--resume`) | 899 | 354 | 2.09 | 29 | 0.266 | 0.20 [0.08, 0.42], −104 | 0.20 [0.08, 0.42], −101 |

* **The iteration is dominated by self-play and evaluation.** Self-play takes
  about 6 min, the arena and baseline evaluations about 7 min, and training
  1–6 s. Within search time, move generation is 64%, CPU inference 21%, and
  the live game's Java-scored generation another ~17% on top.
* **The GPU is idle during self-play and evaluation** (measured 0%
  utilization). During training it measured 20–66% mean utilization, 0.5 GB
  used, and 0.26 GB peak allocated.
* **Resume worked mid-run.** The process exited after iteration 2; iteration 3
  resumed with `--set iterations=3`, and the change was recorded as
  `config_changes`. There were no truncated games and no belief resets.
* Strength across iterations 1–3 is **flat within noise**. Three iterations
  of about 1,000 positions each are far too few to learn Scrabble, and the
  network is still well below greedy. This run demonstrates the pipeline. It
  does not demonstrate learning.

**Replay-ratio finding (controlled).** The first version of the medium config
trained a fixed 200 × 256 samples per iteration, about 50 samples per new
position. That run and the run above trained on **identical** iteration-1
self-play data (per-game seeds) and were evaluated on identical deals:

| Iteration 1 | Samples/position | Value loss | vs greedy | vs heuristic |
|---|---|---|---|---|
| fixed 200 steps | 50.5 | 0.195 (memorized) | 0.10 [0.03, 0.30], **−159 [−207, −111]** | 0.00, −157 |
| `samples_per_position: 8` | 8.1 | 0.736 | 0.40 [0.22, 0.61], **−64 [−113, −14]** | 0.20, −77 |

The margin intervals do not overlap. Over-training on scarce self-play data
made the network worse. The new configs therefore set
`training.samples_per_position: 8`, which matches the ratio implied by the
original `default.yaml`. `training.samples_per_new_position` is logged every
iteration, and the ratio is an experiment in the scaling plan (row 8b).

## 5. Bottlenecks and their consequences

1. **Self-play move generation (CPU, Python):** 66% of a search decision.
   Games/hour scales with CPU workers, so on Delta request as many cores per
   GPU as allowed (scripts default to 16). A compiled generator (C++ or Rust
   via pybind/PyO3) is the single largest remaining speedup. Its correctness
   bar is set by the existing exact-parity tests.
2. **CPU inference** grows with network size (5.5 ms medium, 19.5 ms large).
   For large networks it approaches half of search time. That is where batched
   GPU inference could help, given enough concurrent workers.
3. **GPU training is not a bottleneck.** A medium iteration at 8 samples per
   new position trains in 1–6 s locally. The A100's job is to make large
   networks and many training steps cheap, not to speed up self-play.
4. Python-level per-decision overheads (live Java scoring 0.17 s/turn and
   belief updates) are small but non-zero. They are reported separately as
   `live_movegen_seconds` and `game_step_seconds`.

## 6. Recommended first experiments

1. On one A100 interactive allocation: `bash scripts/delta/gpu_check.sh`
   (§4 of A100_TRAINING.md). This gives real A100 numbers for training,
   inference and a `gpu.yaml` iteration.
2. **Worker scaling** (throughput only; results are identical by design):
   `selfplay.workers` = 7, 15, 31 with matching `--cpus-per-task`. Choose the
   knee of games/hour.
3. **Medium baseline × 3 seeds** (`configs/gpu.yaml`, `--set seed=1|2|3`),
   ~40 iterations each. This sets the noise floor.
4. **Simulations** 16 / 64 against the baseline, then **network width/depth**.
   Then `large.yaml`, then `selfplay.inference=server` for large with 31
   workers.
5. Cheap, using one trained checkpoint: `evaluate.py --compare-methods ismcts
   belief pimc neural_determinization --games 200 --workers 15`. This decides
   whether `belief` should replace `ismcts` as the default.

Judge every experiment with 200–400 game evaluations and head-to-head matches
(SCALING_PLAN.md §2), never with the in-training 20–40 game evaluations alone.

## 7. Limitations (not done, or deliberately not claimed)

* **No GPU test on Delta itself.** SSH requires Duo 2FA. The CUDA code path was
  validated on a laptop GPU, and the Delta scripts were syntax-checked but not
  run.
* **No strength claim.** The local runs are pipeline validations with
  20-game evaluations. They do not show that the new architecture is stronger;
  that is what the scaling plan is for.
* Search inference is batched only *across* workers (server mode), not
  within a single search: one search is sequential PUCT. Virtual-loss
  batching would add complexity for ~22% of decision time and was not
  implemented.
* `torch.compile`, multi-GPU training and distributed self-play across nodes
  are not implemented. One node with one GPU and many CPU workers fits the
  measured bottleneck.
* Schema-1 checkpoints can be evaluated but not resumed (their replay stored
  pre-encoded features).
* Policy targets cover the shortlisted candidates (top scores plus pass and
  the largest exchange), as before. Low-scoring strategic moves outside the
  shortlist cannot be learned. Raising `search.candidates` is an experiment.
