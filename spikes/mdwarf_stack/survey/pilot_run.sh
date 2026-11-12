#!/usr/bin/env bash
# Survey pilot driver (run from a worktree, 2 niced search workers):
#   1. the 11 gate-2 hold-out hosts (end-to-end reproduction, pilot.py check)
#   2. 30 survey stars (throughput only), killed with SIGKILL to the whole process group once
#      mid-run (while a star is part-downloaded if possible), then restarted to completion.
# Peak RAM = max over 2-s samples of the summed RSS of the runner's process group.
# Usage: PY=/path/to/python pilot_run.sh
set -u
PY=${PY:?set PY}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$HOME/Projects/oddball-worlds/outputs/planets_spike
DATA=$HOME/Projects/oddball-worlds/data/planets_spike
P=$OUT/survey_pilot
mkdir -p "$P"

start() {  # tag targets [extra runner args] -> sets PGID
  local tag=$1 tg=$2
  shift 2
  setsid nice -n 19 ionice -c3 "$PY" -I "$HERE/run_survey.py" --tag "$tag" --targets "$tg" \
    --min-sectors 0 --procs 2 "$@" >> "$P/$tag.log" 2>&1 &
  PGID=$!
}

sample() {  # pgid file : max summed RSS (kB) while the group lives
  local max=0 s
  while kill -0 "$1" 2>/dev/null; do
    s=$(ps -o rss= -g "$1" 2>/dev/null | awk '{t += $1} END {print t + 0}')
    [ "$s" -gt "$max" ] && max=$s && echo "$max" > "$2"
    sleep 2
  done
}

nres() { find "$OUT/$1/results" -name 'tic*.json' 2>/dev/null | wc -l; }

echo "== hosts, MAST route (before) $(date -Is)"
T0=$(date +%s)
start survey_pilot_hosts_mast "$P/hosts.csv" --route mast --download-only
sample "$PGID" "$P/rss_hosts_mast.kb" &
wait "$PGID"; echo "hosts mast exit $? after $(( $(date +%s) - T0 )) s"

echo "== hosts, S3 route (after) $(date -Is)"
T0=$(date +%s)
start survey_pilot_hosts "$P/hosts.csv" --route s3 --keep-npz
sample "$PGID" "$P/rss_hosts.kb" &
wait "$PGID"; echo "hosts s3 exit $? after $(( $(date +%s) - T0 )) s"

echo "== survey30 $(date -Is)"
T0=$(date +%s)
start survey_pilot_s30 "$P/survey30.csv"
sample "$PGID" "$P/rss_s30_a.kb" &
# kill once >= 8 stars are done and some star is part-downloaded (or >= 12 done)
while kill -0 "$PGID" 2>/dev/null; do
  n=$(nres survey_pilot_s30)
  part=$(find "$DATA/survey_pilot_s30/partial" -name 's*.npz' 2>/dev/null | wc -l)
  if { [ "$n" -ge 8 ] && [ "$part" -gt 0 ]; } || [ "$n" -ge 12 ]; then
    {
      echo "KILL at $(date -Is): results=$n partial_sector_files=$part"
      echo "  npz files: $(find "$DATA/survey_pilot_s30/lc" -name 'tic*.npz' | wc -l)"
      echo "  tmp files: $(find "$OUT/survey_pilot_s30" "$DATA/survey_pilot_s30" -name '*.tmp*' | wc -l)"
      echo "  procs in group: $(ps -o pid= -g "$PGID" | wc -l)"
    } | tee "$P/kill_state.txt"
    kill -9 -- -"$PGID"
    break
  fi
  sleep 3
done
sleep 3
echo "  survivors after kill: $(ps -o pid= -g "$PGID" 2>/dev/null | wc -l)" | tee -a "$P/kill_state.txt"
echo "  partial dirs after kill: $(ls "$DATA/survey_pilot_s30/partial" | tr '\n' ' ')" | tee -a "$P/kill_state.txt"
echo "== restart $(date -Is)"
start survey_pilot_s30 "$P/survey30.csv"
sample "$PGID" "$P/rss_s30_b.kb" &
wait "$PGID"; echo "s30 exit $? after $(( $(date +%s) - T0 )) s total"
echo "== done $(date -Is)"
