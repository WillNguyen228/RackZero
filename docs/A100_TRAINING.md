# Training RackZero on NCSA Delta A100 GPUs

This guide covers setup, validation, training, monitoring, resuming,
evaluation and run comparison on Delta.

> **What has and has not been run.** Every `python ...` command below was run
> as written on a Windows development machine, both on CPU and on CUDA (an
> NVIDIA RTX 4070 Laptop GPU, PyTorch 2.11 + CUDA 12.8). Those runs exercised
> bf16 training, GPU evaluation, CUDA RNG checkpoints and the batched
> inference server. The Delta-specific steps (SSH, `sacctmgr`, `srun`,
> `sbatch`, the setup script on Linux) **could not be run**: Delta login
> requires a password plus Duo two-factor authentication. Shell scripts were
> syntax-checked only. Nothing here has been measured on an A100; the numbers
> in §7 are from the laptop GPU. Run `scripts/delta/gpu_check.sh` first, then
> trust its output over this document.

## 1. Connect and find your allocation

```bash
ssh wnguyen1@dt-login01.delta.ncsa.illinois.edu      # password + Duo
sacctmgr show assoc where user=$USER format=Account,Partition
```

Use an account whose name indicates GPU access (Delta GPU allocations are
separate from CPU allocations). Below, `$ACCT` stands for it:

```bash
export ACCT=<your GPU account from sacctmgr>          # do not guess this
```

## 2. Put the code and dictionary on Delta

The changes in this work are not committed yet. Either commit and push them
to `https://github.com/WillNguyen228/RackZero` and `git clone` on Delta, or
copy the working tree from your machine (excluding local environments and
outputs):

```bash
# on your machine, from the repository's parent directory
rsync -av --exclude .venv --exclude runs --exclude target --exclude third_party \
  --exclude node_modules Scrabble/ wnguyen1@dt-login01.delta.ncsa.illinois.edu:<DEST>/Scrabble/
scp Scrabble/WOW24v3.zip wnguyen1@dt-login01.delta.ncsa.illinois.edu:<DEST>/Scrabble/
```

Pick `<DEST>` with enough quota: run `quota` on Delta to see your home,
projects and work directories. Run outputs (`runs/`) grow with
checkpoints. A medium run's `latest.pt` with a 200k replay is a few hundred MB.

## 3. One-time environment setup (login node)

```bash
cd <DEST>/Scrabble
RACKZERO_ENV=<DEST>/rackzero-env bash scripts/delta/setup_env.sh
```

The script installs user-space Miniforge, a conda environment with Python 3.11
and **OpenJDK 17** (JPype needs a JDK), PyTorch CUDA 12.4 wheels
(`TORCH_INDEX` to change), this project in editable mode with `psutil`, and it
builds the Java engine JAR. It needs no Delta modules. Activate the
environment in every shell or job:

```bash
export ACTIVATE="source <DEST>/miniforge3/etc/profile.d/conda.sh && conda activate <DEST>/rackzero-env"
eval "$ACTIVATE"
```

## 4. Interactive validation on one A100 (≤ 1 hour)

Request a GPU. Prefer more CPU cores than the 4 in the basic example:
self-play is CPU-bound (§7), and the check runs parallel workers.

```bash
srun --account=$ACCT --partition=gpuA100x4-interactive --nodes=1 --ntasks=1 \
     --cpus-per-task=16 --mem=64g --gpus-per-node=1 --time=01:00:00 --pty bash
```

Inside the allocation:

```bash
cd <DEST>/Scrabble && eval "$ACTIVATE"
nvidia-smi
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
bash scripts/delta/gpu_check.sh
exit                                                     # release the GPU
```

`gpu_check.sh` prints each command it runs and writes everything under
`runs/delta-check-<jobid>/`. It runs:

1. the RackZero unit tests;
2. the smoke training on CUDA with bf16, a resume, and evaluation on CUDA;
3. GPU training throughput (bf16 and fp32) and inference throughput by
   batch size, plus CPU inference latency;
4. with `WOW24v3.zip` present: move-generation and search profiling, and one
   real `gpu.yaml` iteration using one worker per remaining core.

Send back `runs/delta-check-<jobid>/` (`benchmark.jsonl`, `metrics.jsonl`,
`environment.json`) to calibrate the configurations.

## 5. Batch training

`scripts/delta/train.sbatch` requests 1 GPU, 16 cores and 64 GB by default.
It sets `selfplay.workers` to cores − 1 and logs `nvidia-smi` every 30 s to
`runs/gpu-<jobid>.csv`. Account, partition and time limit are given on the
command line:

```bash
cd <DEST>/Scrabble
mkdir -p runs            # Slurm writes runs/slurm-<jobid>.out; the directory must exist
# small: first full-dictionary run (GPU training, small network)
sbatch --account=$ACCT --partition=gpuA100x4 --time=04:00:00 \
  --export=ALL,RACKZERO_ACTIVATE="$ACTIVATE" \
  scripts/delta/train.sbatch configs/small.yaml --set runtime.device=cuda --set runtime.precision=bf16

# medium baseline on the A100 (configs/gpu.yaml = medium + bf16 + workers)
sbatch --account=$ACCT --partition=gpuA100x4 --time=12:00:00 \
  --export=ALL,RACKZERO_ACTIVATE="$ACTIVATE" \
  scripts/delta/train.sbatch configs/gpu.yaml

# large network (more cores help: e.g. --cpus-per-task=32 --mem=96g)
sbatch --account=$ACCT --partition=gpuA100x4 --time=24:00:00 --cpus-per-task=32 --mem=96g \
  --export=ALL,RACKZERO_ACTIVATE="$ACTIVATE" \
  scripts/delta/train.sbatch configs/large.yaml
```

### Warm-started run (Quackle pre-training)

Quackle is built only on your own computer, so generate the teacher data and
pre-train there (see RACKZERO.md, "Warm start and training improvements").
Then copy the pre-trained run directory to Delta, about 10 MB. Run this on your
computer, from the repository root:

```bash
scp -r runs/<warm-pretrain-run> <user>@dt-login01.delta.ncsa.illinois.edu:<DEST>/Scrabble/runs/
```

On Delta, start self-play from it. The dictionary (WOW24v3.zip) and model
settings must match the pre-training:

```bash
sbatch --account=$ACCT --partition=gpuA100x4 --time=12:00:00 \
  --export=ALL,RACKZERO_ACTIVATE="$ACTIVATE" \
  scripts/delta/train.sbatch configs/warm.yaml \
  --set init_from=runs/<warm-pretrain-run>/checkpoints/pretrained.pt
```

Do not put `quackle` in `evaluation.opponents` for a Delta run unless Quackle
is built there. Evaluate against Quackle on your computer instead, using the
checkpoints copied back with `scripts/delta/pull_runs.sh`.

Any setting can be appended as `--set key=value`; see
[SCALING_PLAN.md](SCALING_PLAN.md) for the one-factor experiments. Each run
creates `runs/<name>-<timestamp>/`. The job log
`runs/slurm-<jobid>.out` prints that path on its first line.
`gpuA100x4-preempt` is cheaper but can be preempted. Pair it with the resume
command in §6. Training loses at most the iteration in progress.

Memory: each self-play worker holds a JVM and the WOW24 trie (~0.5 GB with
PyTorch), so allow about 0.6 GB per worker plus 4 GB for the trainer and
replay.

### Monitoring

```bash
squeue -u $USER                                   # job state, node, elapsed time
tail -f runs/slurm-<jobid>.out                    # one summary line per iteration
column -s, -t < runs/gpu-<jobid>.csv | tail       # GPU utilization/memory over time
python scripts/compare_runs.py runs/<run-dir>     # throughput, losses, win rates with CIs
srun --jobid=<jobid> --overlap nvidia-smi         # live view on the job's node
```

Per-iteration metrics are written to `runs/<run>/metrics.jsonl` and
`metrics.csv`:

* `selfplay.*`: games/hour, seconds per decision, and the move-generation,
  inference and simulator time split;
* `training.*`: losses, learning rate, samples/s, peak GPU memory, samples
  per new position;
* `hardware.{selfplay,training,evaluation}`: mean and max GPU utilization and
  memory, CPU load;
* `arena`, `baselines.<opponent>`: win rate, 95% CI, margin, game length.

For TensorBoard, add `--set logging.tensorboard=true` and
`pip install tensorboard`, then run
`tensorboard --logdir runs --port 6006` on a login node with an SSH tunnel.
Delta login nodes are shared, so keep it short.

Cancel jobs, and confirm nothing is left running:

```bash
scancel <jobid>
squeue -u $USER
```

## 6. Resume, evaluate, compare

```bash
# Resume the last completed iteration (the run directory stores its config).
# Extending iterations or changing worker count is allowed; trajectory-changing
# edits (e.g. learning rate) need --allow-config-change and are logged.
sbatch --account=$ACCT --partition=gpuA100x4 --time=12:00:00 \
  --export=ALL,RACKZERO_ACTIVATE="$ACTIVATE" \
  scripts/delta/train.sbatch --resume runs/<run-dir> --set iterations=80

# Evaluate the best model (inside an allocation; CPU-parallel)
python evaluate.py --checkpoint runs/<run-dir> --opponents greedy heuristic random \
  --games 200 --workers 15 --output runs/<run-dir>/final-eval.jsonl
# Head-to-head against another run's best model
python evaluate.py --checkpoint runs/<run-b> --opponent search \
  --opponent-checkpoint runs/<run-a>/checkpoints/best.pt --games 200 --workers 15
# Compare runs
python scripts/compare_runs.py runs/gpu-* runs/large-* --csv runs/comparison.csv
```

## 7. Where the time goes (measured locally)

These are the measurements to check against Delta. They come from the laptop
(`scripts/benchmark.py`, WOW24, PyTorch 2.11, RTX 4070 Laptop GPU, bf16).

| Measurement | Result |
|---|---|
| One search decision, `medium` (ismcts, 32 sims, depth 2) | 1.55 s: **66% move generation**, 22% inference, 9% simulator steps |
| Move generation, one cold WOW24 position | 31 ms (fast) vs 32 ms (reference); search gains 2.2× from board caching |
| CPU inference per call (1 thread) | small 2.7 ms · medium 5.5 ms · large 19.5 ms |
| GPU inference throughput, batch 256 | small 48k · medium 33k · large 13.6k positions/s (batch 1: 6–7 ms, launch-bound) |
| GPU training throughput (batch 256 / 512 / 1024) | small 13.9k / 30.4k / 36.2k · medium 12.1k / 14.1k / 13.2k · large 4.2k / 3.7k / 3.8k samples/s |
| Peak GPU memory, training (batch 512) | small 0.23 GB · medium 0.56 GB · large 1.6 GB |
| Batched inference server, large net, 8 workers | **slower** than local CPU inference (4.0 vs 3.7 s/move): mean batch 1.6, ~35 ms per round trip on Windows |

What this means for the A100:

* **Self-play is CPU-bound.** In a medium iteration, GPU training takes
  seconds while self-play takes minutes. Throughput (games/hour) scales with
  self-play **workers**, i.e. allocated CPU cores. The GPU sits mostly idle
  during self-play, and its measured utilization will show this.
* **The A100 helps** with large networks, large batches and frequent training.
  An A100 should train substantially faster than the laptop figures above;
  measure it with `gpu_check.sh`.
* **Batched GPU inference** (`--set selfplay.inference=server`) only pays off
  when many workers wait on inference at once: large networks, many workers.
  Locally it lost. Linux IPC is faster, so measure it on Delta before using
  it (SCALING_PLAN row 2).
* The largest remaining speedup would be a compiled move generator. In Python
  it is 66% of search time, and its output is already pinned by exact parity
  tests that a replacement would have to pass.

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA requested but unavailable` | You are on a login node or a non-GPU allocation, or the env has a CPU-only torch: check `python -c "import torch; print(torch.version.cuda)"`. |
| `javac not found` / JPype cannot start the JVM | Activate the env (it provides OpenJDK 17); `java -version` must show 17+. |
| `Engine JAR not found` | `python -m scrabble_gym.build` in the repository root. |
| Workers killed / out of memory / `RuntimeError: bad allocation` | Lower `selfplay.workers` or raise `--mem` (≈0.6 GB per worker). Locally, 12 workers with a CUDA build of PyTorch exhausted 16 GB on Windows, which commits each process's CUDA libraries; 8 workers ran. |
| `already holds a run` | Use `--resume <run-dir>` or a new `--run-dir`. |
| `Resume config differs in trajectory-changing settings` | Intended: pass `--allow-config-change` if the change is deliberate (it is recorded). |
| `No naturally completed self-play games` | Raise `max_turns`; check for truncations in metrics. |
| Low GPU utilization | Expected during self-play (CPU-bound). Training-phase utilization is under `hardware.training`. |
