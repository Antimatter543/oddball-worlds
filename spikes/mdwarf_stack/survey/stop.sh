#!/usr/bin/env bash
# Stop the survey unit. Finished stars are kept; a star in progress is redone on the next launch.
set -euo pipefail
UNIT=${UNIT:-oddball-survey}
if systemctl --user is-active --quiet "$UNIT"; then
  systemctl --user stop "$UNIT"
  echo "stopped $UNIT"
else
  echo "$UNIT is not running"
fi
systemctl --user reset-failed "$UNIT" 2>/dev/null || true
