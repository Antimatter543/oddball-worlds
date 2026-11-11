"""Survey progress. Counts only: it never opens a result file and never prints a signal.

Reads OUT/<tag>/manifest.json, progress.jsonl (statuses and timings written by the runner,
no signal fields) and errors.jsonl, lists the results directory, and measures disk use.

Usage: python -I spikes/mdwarf_stack/survey/status.py [--tag survey] [--window-h 2]
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, OUT  # noqa: E402


def _du(p: Path) -> int:
    n = 0
    for root, _, files in os.walk(p):
        for f in files:
            try:
                n += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return n


def _jsonl(p: Path):
    if not p.exists():
        return []
    rows = []
    for line in open(p):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # torn last line after a kill
    return rows


def _t(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s)


def _gb(n: int) -> str:
    return f"{n / 1e9:.2f} GB"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="survey")
    ap.add_argument("--window-h", type=float, default=2.0, help="window for the recent rate")
    ap.add_argument("--unit", default="oddball-survey")
    a = ap.parse_args()
    out, data = OUT / a.tag, DATA / a.tag
    mp = out / "manifest.json"
    if not mp.exists():
        print(f"no survey at {out} (manifest.json missing)")
        return
    man = json.loads(mp.read_text())
    n_targets = man["n_targets"]
    done_files = {int(p.stem[3:]) for p in (out / "results").glob("tic*.json")}
    prog = {}
    for r in _jsonl(out / "progress.jsonl"):
        prog[int(r["tic"])] = r  # last line per star wins (a --retry-failed rerun)
    prog = {t: r for t, r in prog.items() if t in done_files}
    st = collections.Counter(r["status"] for r in prog.values())
    n_inc = sum(1 for r in prog.values() if r.get("incomplete"))
    n_lt6 = sum(1 for r in prog.values() if "lt6_real_sectors" in (r.get("tags") or []))
    n_fb = sum(1 for r in prog.values() if "fallback_mass_radius" in (r.get("tags") or []))
    pred = sum(r.get("n_sectors_pred") or 0 for r in prog.values() if r["status"] == "ok")
    found = sum(r.get("n_sectors_found") or 0 for r in prog.values() if r["status"] == "ok")
    done = len(done_files)
    remaining = n_targets - done

    try:
        active = subprocess.run(["systemctl", "--user", "is-active", a.unit], capture_output=True,
                                text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        active = "unknown"
    print(f"survey '{a.tag}'  unit {a.unit}: {active}   first start {man.get('first_start')}  "
          f"starts {len(man.get('starts', []))}  frozen {man['frozen_sha256'][:12]}")
    print(f"targets {n_targets}   done {done} ({100 * done / max(n_targets, 1):.1f}%)   "
          f"remaining {remaining}")
    print(f"  ok {st.get('ok', 0)}   no_data {st.get('no_data', 0)}   failed {st.get('failed', 0)}"
          f"   incomplete (some sectors failed) {n_inc}   <6 real sectors {n_lt6}   "
          f"fallback M/R {n_fb}")
    if pred:
        print(f"  sectors found / predicted (ok stars): {found} / {pred} = {found / pred:.2f}")
    npz_dir = data / "lc"
    n_npz = len(list(npz_dir.glob("tic*.npz"))) if npz_dir.exists() else 0
    n_npz_done = sum(1 for t in done_files if (npz_dir / f"tic{t}.npz").exists())
    print(f"  downloaded, awaiting search: {n_npz - n_npz_done}")

    times = sorted(_t(r["finished_utc"]) for r in prog.values() if r.get("finished_utc"))
    if times:
        now = dt.datetime.now(dt.UTC)
        span = (times[-1] - times[0]).total_seconds() / 3600 if len(times) > 1 else 0.0
        w0 = now - dt.timedelta(hours=a.window_h)
        recent = [t for t in times if t >= w0]
        rate_all = (len(times) - 1) / span if span > 0 else float("nan")
        rate_rec = len(recent) / a.window_h
        rate = rate_rec if len(recent) >= 5 else rate_all
        eta = remaining / rate if rate and rate == rate and rate > 0 else float("nan")
        print(f"rate: {rate_rec:.1f} stars/h over the last {a.window_h:g} h, {rate_all:.1f} "
              f"stars/h overall;  ETA {eta:.1f} h  (last finish {times[-1].isoformat()})")
        ok = [r for r in prog.values() if r["status"] == "ok"]
        dl = [r["dl_wall_s"] for r in ok if r.get("dl_wall_s")]
        cpu = [r["search_cpu_s"] for r in ok if r.get("search_cpu_s")]
        if dl:
            print(f"  per star: download {sum(dl) / len(dl):.0f} s wall (mean of {len(dl)}), "
                  f"search {sum(cpu) / max(len(cpu), 1):.0f} CPU-s (mean of {len(cpu)})")
        hw = [r["worker_vmhwm_mb"] for r in ok if r.get("worker_vmhwm_mb")]
        if hw:
            print(f"  search worker peak RSS: {max(hw):.0f} MB")

    print(f"disk: data {_gb(_du(data))} ({n_npz} NPZ, partial {_gb(_du(data / 'partial'))})   "
          f"outputs {_gb(_du(out))}")

    errs = _jsonl(out / "errors.jsonl")
    if errs:
        allk = collections.Counter(e.get("kind", "?") for e in errs)
        fin = collections.Counter(e.get("kind", "?") for e in errs if e.get("final"))
        print(f"errors logged: {len(errs)} (each failed attempt counts; final = gave up)")
        for k, n in allk.most_common():
            print(f"  {k:24s} {n:6d}   final {fin.get(k, 0)}")
        last = max(e.get("utc", "") for e in errs)
        print(f"  last error at {last}")
    else:
        print("errors logged: 0")


if __name__ == "__main__":
    main()
