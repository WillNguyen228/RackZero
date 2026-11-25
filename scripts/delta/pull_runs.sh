#!/usr/bin/env bash
# Copy RackZero run directories from Delta to this computer (run it LOCALLY,
# e.g. in Git Bash on Windows, from the repository root).
#
#   export DELTA_HOST=<user>@dt-login01.delta.ncsa.illinois.edu
#   bash scripts/delta/pull_runs.sh <remote repo dir>              # every run + Slurm logs
#   bash scripts/delta/pull_runs.sh <remote repo dir> <run> [...]  # selected runs + Slurm logs
#
# <remote repo dir> is the RackZero checkout on Delta (e.g. RackZero, relative
# to your Delta home directory). <run> is a directory name under runs/.
#
# Everything travels through ONE ssh connection (a tar stream), so you
# authenticate (password + Duo) once. Files land in ./runs/<run>/ with the
# same layout as on Delta, so evaluate.py / train.py --resume work on them
# locally. Re-running refreshes the copies; nothing on Delta is modified.
# Set LIGHT=1 to skip checkpoints/latest.pt (the large replay-buffer file
# that is only needed to --resume training).
set -euo pipefail

if [ $# -lt 1 ]; then
  sed -n '2,17p' "$0"; exit 2
fi
: "${DELTA_HOST:?set DELTA_HOST=<user>@dt-login01.delta.ncsa.illinois.edu}"
SSH="${SSH:-ssh}"
REMOTE_DIR="$1"; shift
DEST="${DEST:-runs}"
mkdir -p "$DEST"

if [ $# -gt 0 ]; then
  SELECT=$(printf "'%s' " "$@")
else
  SELECT="*/"
fi
EXCLUDE=""
if [ "${LIGHT:-0}" = 1 ]; then EXCLUDE="--exclude=checkpoints/latest.pt"; fi

# Remote side: list what will be sent (to stderr), then stream it.
REMOTE_CMD="cd '$REMOTE_DIR/runs' && shopt -s nullglob && \
items=( $SELECT slurm-*.out gpu-*.csv ) && \
[ \${#items[@]} -gt 0 ] || { echo 'nothing to copy in $REMOTE_DIR/runs' >&2; exit 1; } && \
for i in \"\${items[@]}\"; do [ -e \"\$i\" ] || { echo \"no such run on Delta: \$i\" >&2; exit 1; }; done && \
du -sh \"\${items[@]}\" >&2 && tar -cf - $EXCLUDE \"\${items[@]}\""

echo "pulling from $DELTA_HOST:$REMOTE_DIR/runs into $DEST/" >&2
"$SSH" "$DELTA_HOST" "bash -c $(printf '%q' "$REMOTE_CMD")" | tar -xf - -C "$DEST"

echo "done. Local copies:" >&2
for d in "$DEST"/*/; do
  [ -d "$d/checkpoints" ] || continue
  n=$(ls "$d/checkpoints" | grep -c '^iteration_' || true)
  echo "  $d  ($n iteration checkpoints, $(du -sh "$d" | cut -f1))" >&2
done
