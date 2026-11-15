#!/usr/bin/env bash
# After a vetting batch (survey100/run_batch.sh): pixel check on every A/B signal not yet checked,
# cards (frozen vetting/cards2.py, paths patched) for every A+/A/B/K signal of the batch's stars,
# then delete the batch's light curves and SAP files (cards, tables and pixel numbers are kept).
# Usage: survey100/after_batch.sh BATCH_FILE
set -euo pipefail
B=$1
HERE=$(cd "$(dirname "$0")/.." && pwd)
MAIN=${ODDBALL_MAIN:-$HOME/Projects/oddball-worlds}
PY=$MAIN/.venv/bin/python
O=$MAIN/outputs/planets_spike/survey100
LC=$MAIN/data/planets_spike/lc_cache/lc
cd "$HERE"
KEYS=$($PY -I survey100/common100.py --ab-without-pixel)
echo "pixel $(date -u +%FT%TZ): $(echo $KEYS | wc -w) keys"
if [ -n "$KEYS" ]; then
  PIXEL3_INFO=$O/vet/star_info.json PIXEL3_OUT=$O/vet/pixel.jsonl nice -n 19 ionice -c3 \
      $PY -I gate3/pixel3.py $O/vet/vet2_survey100.s0of2.jsonl,$O/vet/vet2_survey100.s1of2.jsonl \
      $KEYS > "$B.pixel.log" 2>&1
fi
echo "cards $(date -u +%FT%TZ)"
VET_LC_DIR=$LC nice -n 19 $PY -I survey100/cards100.py --batch "$B" > "$B.cards.log" 2>&1
echo "sap meta $(date -u +%FT%TZ)"
# keep each star's per-sector CROWDSAP / FLFRCSAP (the dossier needs them after the SAP files go)
$PY -I survey100/sap_meta.py "$B" >> "$B.cards.log" 2>&1
echo "delete $(date -u +%FT%TZ)"
for t in $(cat "$B"); do
  rm -f "$LC/tic$t.npz" "$MAIN/outputs/planets_spike/vetting/sap/tic$t.npz"
done
du -sh "$LC" 2>/dev/null || true
echo "done $(date -u +%FT%TZ)"
