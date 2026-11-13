"""50 pc survey: non-C signals (K, A, B) with their period and depth, regraded with frozen v2."""
import json
import sys
from pathlib import Path

W = str(Path(__file__).resolve().parents[2])  # spikes/mdwarf_stack
for p in (W, W + "/vetting", W + "/gate2", W + "/../../src"):
    sys.path.insert(0, p)
from vet2 import grade2, load_frozen2  # noqa: E402

OUT = Path("/home/antiraedus/Projects/oddball-worlds/outputs/planets_spike")
thr = load_frozen2()
pix = {}
for x in open(OUT / "vetting/pixel.jsonl"):
    r = json.loads(x); pix[r["key"]] = r["pixel"]
seen = {}
for f in sorted((OUT / "vetting/v2").glob("vet2_survey.s*of2.jsonl")):
    for x in open(f):
        r = json.loads(x)
        if "stats" not in r:
            continue
        st = dict(r["stats"])
        if r["key"] in pix:
            st["pixel"] = pix[r["key"]]
        g = grade2(st, thr, 8.054233220378304)
        if g["tier"] != "C":
            rp = (st["depth"] ** 0.5) * r["radius"] * 109.1 if r.get("radius") else float("nan")
            seen.setdefault(g["tier"], []).append((r["tic"], round(st["P"], 2), round(st["mes"], 1),
                                                   round(1e3 * st["depth"], 2), round(rp, 2)))
for t, v in sorted(seen.items()):
    inb = [x for x in v if 15 <= x[1] <= 50]
    print(f"tier {t}: {len(v)} signals, {len(inb)} at 15-50 d; unique TICs {len({x[0] for x in v})}")
    for x in sorted(inb, key=lambda z: z[1]):
        print("   tic, P, MES, depth ppt, Rp(rough, R_e)", x)
