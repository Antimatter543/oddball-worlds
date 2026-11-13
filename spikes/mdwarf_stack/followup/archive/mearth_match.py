"""Match target coordinates to the MEarth DR11/DR2 target tables (index.html files).
Usage: mearth_match.py DIR OUTJSON"""
import json
import re
import sys
from pathlib import Path

import numpy as np

d, outj = Path(sys.argv[1]), Path(sys.argv[2])
STARS = {2102868: (79.33369680453, 32.12492877993, 86.155, -296.162),
         234295610: (357.98397801044, -64.7931962242, 74.585, -59.845),
         146636926: (42.33950035414, -44.26805062609, 69.578, 74.576),
         141816993: (205.44368664769, 58.25523811632, 81.775, -59.357),
         57830249: (174.89049075802, -30.66688311479, -86.422, -25.834)}
rx = re.compile(r"(\d{8}[+-]\d{7})\s+(\d\d) (\d\d) ([\d.]+) ([+-]\d\d) (\d\d) ([\d.]+)")
out = {t: [] for t in STARS}
nearest = {t: (1e9, None) for t in STARS}
nrows = {}
for f in sorted(d.glob("*index.html")):
    n = 0
    for line in f.read_text(errors="replace").splitlines():
        txt = re.sub(r"<[^>]*>", " ", line)
        m = rx.search(txt)
        if not m:
            continue
        n += 1
        ra = 15 * (int(m.group(2)) + int(m.group(3)) / 60 + float(m.group(4)) / 3600)
        sgn = -1 if m.group(5).startswith("-") else 1
        de = sgn * (abs(int(m.group(5))) + int(m.group(6)) / 60 + float(m.group(7)) / 3600)
        for t, (r0, d0, pa, pd) in STARS.items():
            dt = 2000.0 - 2016.0  # MEarth table coordinates are J2000 epoch 2000.0
            r1 = r0 + pa * dt / 3.6e6 / np.cos(np.radians(d0))
            d1 = d0 + pd * dt / 3.6e6
            sep = np.hypot((ra - r1) * np.cos(np.radians(d0)), de - d1) * 3600
            if sep < nearest[t][0]:
                nearest[t] = (float(sep), f.name + ' ' + m.group(1))
            if sep < 30:
                links = re.findall(r'href="([^"]+)"', line)
                out[t].append({"file": f.name, "sep_as": round(float(sep), 2), "row": txt.strip(),
                               "links": links})
    nrows[f.name] = n
res = {"rows_parsed": nrows, "nearest": {str(k): v for k, v in nearest.items()}, "matches": {str(k): v for k, v in out.items()}}
outj.write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1)[:4000])
