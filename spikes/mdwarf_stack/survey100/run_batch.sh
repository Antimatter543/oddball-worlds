#!/usr/bin/env bash
# One vetting batch of the 50-100 pc survey (docs/spikes/mdwarf_100pc.md R9), frozen v2 unchanged:
#   1. rebuild the batch's survey light curves + SAP (survey/prep_vet.py, one process)
#   2. v2 measurement (gate3/vet100.py, two niced shards by TIC mod 2)
# The catalogue step (gate3/catalog3.py) runs once for all stars before the first batch.
# The pixel check, cards and the light-curve deletion are run by the caller after each batch
# (survey100/after_batch.sh), because the cards need the light curves.
# Usage: survey100/run_batch.sh BATCH_FILE
set -euo pipefail
B=$1
HERE=$(cd "$(dirname "$0")/.." && pwd)
MAIN=${ODDBALL_MAIN:-$HOME/Projects/oddball-worlds}
PY=$MAIN/.venv/bin/python
O=$MAIN/outputs/planets_spike/survey100
LC=$MAIN/data/planets_spike/lc_cache/lc
cd "$HERE"
echo "prep $(date -u +%FT%TZ) $B"
VET_LC_DIR=$LC nice -n 19 ionice -c3 $PY -I survey/prep_vet.py "$B" --merged $O/merged/results.jsonl.gz \
    --max-mb 1200 --log $O/vet/prep.jsonl > "$B.prep.log" 2>&1
while [ ! -s $O/vet/star_info.json ]; do sleep 30; done   # catalogue step must be finished
echo "vet $(date -u +%FT%TZ)"
# rows of this batch only: a star whose light curve is absent would otherwise get "prepare" error
# rows, which run_vet2 treats as done and would never redo
$PY -I -c "import sys,json; k={int(x) for x in open(sys.argv[1]).read().split()}; \
[sys.stdout.write(l) for l in open(sys.argv[2]) if json.loads(l)['tic'] in k]" \
    "$B" $O/vet/rows_long.jsonl > "$B.rows.jsonl"
for s in 0 1; do
  VET_LC_DIR=$LC nice -n 19 ionice -c3 $PY -I gate3/vet100.py --info $O/vet/star_info.json \
      --rows "$B.rows.jsonl" --jobs $O/vet/jobs_long.jsonl \
      --out $O/vet/vet2_survey100.s${s}of2.jsonl --shard ${s}/2 > "$B.vet$s.log" 2>&1 &
done
wait
echo "done $(date -u +%FT%TZ)"
