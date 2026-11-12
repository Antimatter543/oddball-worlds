"""Plain-English summary for a v2-graded signal (wraps report.summary; v1 wording rules kept).

Adds: tier A+ wording, the v2 tests (gap edge, local red noise, few-transit margin),
contamination as information, and the measured false-alarm odds for the signal's own tier and
transit count from the fresh validation set (v2/analyze_val.json).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import report
from common import OUT

Q2 = {k: v for k, v in report.QUESTION.items() if k not in ("sector_consistency", "contamination")}
Q2.update({
    "gap_edge": "dips that sit at the edge of a gap in the data",
    "red_noise": "a dip no stronger than the star's own slow wiggles around the dip times",
    "low_margin": "a few-transit noise pile-up just above the detection threshold",
})
VERDICT_APLUS = ("Tier A+, the strongest grade: it passed every check, has 3 or more dips, and "
                 "the pixel check places the dip on this star in 3 or more strong dips.")


def load_odds(path=None):
    p = Path(path) if path else OUT / "vetting" / "v2" / "analyze_val.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def odds_sentence(tier: str, n_tr: int, js: dict | None, version: str = "v2") -> str:
    if js is None or tier not in ("A+", "A", "B"):
        return ""
    b = "2" if n_tr == 2 else "3" if n_tr == 3 else "4" if n_tr == 4 else "5+"
    o = js["odds"].get(f"{version}|{tier}|{b}")
    if not o:
        return ""
    nb = "5 or more" if b == "5+" else b
    m = o["fa_in_tier"] + o["inj_in_tier"]
    s = (f"Measured false-alarm odds for signals like this one (tier {tier}, {nb} transits): in "
         f"fresh testing ({js['n_stars']} survey stars, never searched for real), "
         f"{o['fa_in_tier']} of the {o['fa_total']} null false alarms at MES >= 7 (inverted or time-"
         f"scrambled copies of real stars) ended up in this tier with this transit count, and so did {o['inj_in_tier']} of the "
         f"{o['inj_bin']} detected injected planets with {nb} transits. Signals like this one were "
         f"false {o['fa_in_tier']} times out of {m} in testing, but the test planted about three "
         f"planets per star, far more than nature does, so that ratio flatters the candidate. "
         f"Scaled to the survey, about {o['survey_fa_mean']:.1f} "
         f"[{o['survey_fa_lo']:.1f}, {o['survey_fa_hi']:.1f}] false alarms like this are expected "
         f"per 4,700 stars. If that is comparable to the number of real planets the survey can "
         f"find (a handful is the rough expectation, not a measurement), a signal like this one is "
         f"more likely false than real.")
    if o["fa_in_tier"] == 0:
        s += (" Zero test false alarms reached this grade, but with "
              f"{o['fa_total']} false alarms tested, that bounds the rate, it does not make it zero.")
    return s


def summary2(stats, g, rstar=None, name=None, blind=False, odds=None, version="v2") -> str:
    g1 = dict(g)
    tier = g["tier"]
    if tier == "A+":
        g1["tier"] = "A"
    old = report.QUESTION
    report.QUESTION = Q2
    try:
        text = report.summary(stats, g1, rstar=rstar, name=name, blind=blind)
    finally:
        report.QUESTION = old
    if tier == "A+":
        text = text.replace("Tier A, a planet candidate: it passed every check below and has 3 or "
                            "more dips.", VERDICT_APLUS)
    extra = []
    if g.get("info", {}).get("contamination"):
        extra.append("For information (not a strike): " + g["info"]["contamination"]
                     + "; only the pixel check (tier A+) addresses that.")
    o = odds_sentence(tier, int(stats.get("n_transits", 0)), odds, version)
    if o:
        extra.append(o)
    return text + (" " + " ".join(extra) if extra else "")
