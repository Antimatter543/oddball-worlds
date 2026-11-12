#!/usr/bin/env bash
# Kaggle shard helper (run from the main checkout). Kernel folders come from build_kernel.py.
#   shards.sh push   N I [I ...]   push shard kernels I.. of N (Kaggle runs at most ~5 at once)
#   shards.sh status N             one status line per shard
#   shards.sh fetch  N [I,J,..]    pull finished outputs into outputs/.../survey/shards + merge
# Env: TAG (default survey), SLUG (default transit-survey), KPY (a python with the kaggle
# package and pandas; default python3).
set -euo pipefail
MAIN=${MAIN:-$HOME/Projects/oddball-worlds}
TAG=${TAG:-survey}
SLUG=${SLUG:-transit-survey}
KPY=${KPY:-python3}
KDIR=$MAIN/outputs/planets_spike/$TAG/kaggle_kernels
cmd=${1:?push|status|fetch}
N=${2:?number of shards}
NNN=$(printf %03d "$N")
user=$(kaggle config view 2>/dev/null | awk -F': ' '/username/{print $2}')
slug() { if [[ $N -eq 1 ]]; then echo "$SLUG"; else echo "$SLUG-$(printf %03d "$1")of$NNN"; fi; }
case $cmd in
  push)
    shift 2
    for i in "$@"; do
      d=$KDIR/$(slug "$i")
      out=$(kaggle kernels push -p "$d" 2>&1 | grep -v "outdated" || true)
      # the API reports a refused push (e.g. too many running sessions) in the text, not the exit code
      if grep -q "successfully pushed" <<<"$out"; then echo "shard $i: pushed"; else echo "shard $i: NOT pushed: $out"; fi
    done ;;
  status)
    for ((i = 0; i < N; i++)); do
      s=$(kaggle kernels status "$user/$(slug "$i")" 2>&1 | grep -o 'status "[^"]*"' || echo "not found")
      echo "shard $i: $s"
    done ;;
  fetch)
    args=(--of "$N" --tag "$TAG" --slug "$SLUG")
    [[ $# -ge 3 ]] && args+=(--shards "$3")
    "$KPY" -I "$MAIN/spikes/mdwarf_stack/survey/kaggle/fetch_outputs.py" "${args[@]}" ;;
  *) echo "unknown command $cmd" >&2; exit 2 ;;
esac
