"""Keep the SAP files' per-sector metadata (CROWDSAP, FLFRCSAP, URI) for a batch's stars in
<OUT>/survey100/vet/sap_meta.json before after_batch.sh deletes the SAP files.
Usage: python -I survey100/sap_meta.py BATCH_FILE"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import OUT

fp = OUT / "survey100" / "vet" / "sap_meta.json"
d = json.loads(fp.read_text()) if fp.exists() else {}
n = 0
for t in Path(sys.argv[1]).read_text().split():
    p = OUT / "vetting" / "sap" / f"tic{t}.npz"
    if p.exists():
        d[t] = json.loads(str(np.load(p, allow_pickle=False)["meta"]))
        n += 1
fp.write_text(json.dumps(d, default=float))
print("sap meta kept for", n, "stars")
