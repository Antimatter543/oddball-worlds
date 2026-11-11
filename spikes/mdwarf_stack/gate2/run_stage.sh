#!/usr/bin/env bash
# Run gate-2 job files in order, 3 niced CPU workers, resumable (run2.py skips finished ids).
# Usage: gate2/run_stage.sh NAME [NAME ...]   (NAME = job file stem under outputs/.../gate2/jobs)
set -u
PY=${PY:-/home/antiraedus/Projects/oddball-worlds/.claude/worktrees/agent-a666696a20e0e7529/.venv/bin/python}
HERE=$(cd "$(dirname "$0")" && pwd)
G2=/home/antiraedus/Projects/oddball-worlds/outputs/planets_spike/gate2
for name in "$@"; do
  echo "== $name $(date +%T)"
  nice -n 19 ionice -c3 "$PY" -I "$HERE/run2.py" "$G2/jobs/$name.jsonl" "$G2/$name.out.jsonl" --procs 3
done
echo "== done $(date +%T)"
