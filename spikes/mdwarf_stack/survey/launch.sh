#!/usr/bin/env bash
# Start the Track C survey as a systemd user unit (survives logout; lingering is enabled).
#   spikes/mdwarf_stack/survey/launch.sh [runner args, e.g. --retry-failed]
# Runs the code and data of the MAIN checkout, never a worktree. Resumable: running it again
# after a stop, crash or reboot continues where the survey left off.
# Overrides for testing only: UNIT, MAIN, PY.
set -euo pipefail
UNIT=${UNIT:-oddball-survey}
MAIN=${MAIN:-$HOME/Projects/oddball-worlds}
PY=${PY:-$MAIN/.venv/bin/python}
RUNNER=$MAIN/spikes/mdwarf_stack/survey/run_survey.py
TAG=survey
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
  if [[ ${args[i]} == --tag ]]; then TAG=${args[i + 1]}; fi
done
OUTDIR=$MAIN/outputs/planets_spike/$TAG
LOG=$OUTDIR/survey.log

case "$MAIN" in */.claude/worktrees/*)
  echo "refusing: MAIN=$MAIN is a worktree (it would be deleted under the run)" >&2; exit 2;;
esac
[[ -f $RUNNER ]] || { echo "missing $RUNNER (is the survey branch merged into main?)" >&2; exit 2; }
if ! "$PY" -I -c "import numba, wotan, astroquery" 2>/dev/null; then
  echo "$PY lacks numba/wotan/astroquery. Install the planets group without removing others:" >&2
  echo "  cd $MAIN && uv sync --inexact --group planets" >&2
  exit 2
fi
if systemctl --user is-active --quiet "$UNIT"; then
  echo "$UNIT is already running"
  systemctl --user status "$UNIT" --no-pager | head -5
  exit 0
fi
systemctl --user reset-failed "$UNIT" 2>/dev/null || true
mkdir -p "$OUTDIR"
echo "== launch $(date -Is) unit=$UNIT args=$*" >> "$LOG"

systemd-run --user --unit="$UNIT" --description="oddball-worlds Track C survey" \
  --working-directory="$MAIN" \
  -p Nice=19 -p IOSchedulingClass=idle -p CPUQuota=300% -p MemoryMax=12G \
  -p Restart=on-failure -p RestartSec=120 -p OOMPolicy=continue \
  -p StartLimitIntervalSec=3600 -p StartLimitBurst=10 -p TimeoutStopSec=60 \
  -p StandardOutput="append:$LOG" -p StandardError="append:$LOG" \
  -E ODDBALL_MAIN="$MAIN" -E CUDA_VISIBLE_DEVICES= -E PYTHONUNBUFFERED=1 \
  -E NUMBA_NUM_THREADS=1 -E OMP_NUM_THREADS=1 -E OPENBLAS_NUM_THREADS=1 -E MKL_NUM_THREADS=1 \
  "$PY" -I "$RUNNER" "$@"
echo "started $UNIT; log: $LOG"
echo "progress: $PY -I $MAIN/spikes/mdwarf_stack/survey/status.py --tag $TAG"
