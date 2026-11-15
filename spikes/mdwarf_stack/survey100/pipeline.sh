#!/usr/bin/env bash
# All vetting batches of the 100 pc survey in sequence: run_batch.sh then after_batch.sh per batch.
# Usage: survey100/pipeline.sh BATCH_FILE [BATCH_FILE ...]  (log: <OUT>/survey100/vet/pipeline.log)
set -uo pipefail
D=$(cd "$(dirname "$0")" && pwd)
for b in "$@"; do
  bash "$D/run_batch.sh" "$b" || { echo "FAILED run_batch $b"; exit 1; }
  bash "$D/after_batch.sh" "$b" || { echo "FAILED after_batch $b"; exit 1; }
done
echo "pipeline done $(date -u +%FT%TZ)"
