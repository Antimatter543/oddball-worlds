"""Unit tests for the shortlist helpers (pytest spikes/mdwarf_stack/survey/test_shortlist_tools.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from physics import hz_limits, implied, next_transits, timing_sigma
from vet_rows import convert


def test_earth_sun():
    d = implied(365.25, (1 / 109.08) ** 2, 1.0, 1.0, 5772.0)
    assert d["rp_re"] == pytest.approx(1.0, rel=1e-3)
    assert d["a_au"] == pytest.approx(1.0, rel=2e-3)
    assert d["s_earth"] == pytest.approx(1.0, rel=5e-3)
    assert d["teq_a03_k"] == pytest.approx(254.8, abs=1.0)
    assert d["zone"] == "inside the conservative habitable zone"


def test_hz_ordering_cool_star():
    h = hz_limits(3000.0)
    assert h["recent_venus"] > h["runaway_greenhouse"] > h["maximum_greenhouse"] > h["early_mars"]
    # the inner edge for a cool star sits below the Sun's ~1.0 S_earth
    assert h["runaway_greenhouse"] < 1.0


def test_next_transits_and_timing():
    nx = next_transits(100.0, 10.0, 125.0, 3)
    assert nx == [130.0, 140.0, 150.0]
    s1 = timing_sigma(10.0, 0.1, [100.0, 110.0, 120.0], 130.0)
    s2 = timing_sigma(10.0, 0.1, [100.0, 110.0, 120.0], 300.0)
    assert 0 < s1 < s2


def test_convert_keeps_signals():
    rec = {"tic": 5, "n_sectors_found": 7, "sectors_found": [1, 2], "incomplete": False,
           "search": {"long": {"max_sig": 6, "signals": [{"P": 20.0, "mes": 9.0}]},
                      "short": {"max_sig": 6, "signals": []}}}
    r = convert(rec)
    assert r["id"] == "survey-5-long" and r["range"] == "long" and r["mode"] == "real"
    assert r["signals"] == [{"P": 20.0, "mes": 9.0}] and r["n_sectors"] == 7
