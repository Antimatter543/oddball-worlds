#!/usr/bin/env bash
# Run vetting job files with ONE niced worker, in order (resumable). Search jobs use the frozen
# gate-2 runner; "vet:<stage>" runs vetting/run_vet.py for that stage.
# Usage: vetting/run_v.sh dev_inj_v test_nulls_v | vetting/run_v.sh vet:dev
set -u
PY=${PY:-/home/antiraedus/Projects/oddball-worlds/.claude/worktrees/agent-a666696a20e0e7529/.venv/bin/python}
HERE=$(cd "$(dirname "$0")" && pwd)
V=/home/antiraedus/Projects/oddball-worlds/outputs/planets_spike/vetting
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for name in "$@"; do
  echo "== $name $(date +%T)"
  case "$name" in
    vet:*) nice -n 19 ionice -c3 "$PY" -I "$HERE/run_vet.py" "${name#vet:}" ;;
    *) nice -n 19 ionice -c3 "$PY" -I "$HERE/../gate2/run2.py" "$V/jobs/$name.jsonl" "$V/$name.out.jsonl" --procs 1 ;;
  esac
done
echo "== done $(date +%T)"
