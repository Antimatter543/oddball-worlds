"""Track C survey runner: the frozen gate-2 search over every target star, resumable.

Pipeline (one process, two stages that never wait on each other):
  downloader thread  stars in order (most predicted sectors first) -> product list (cached)
                     -> every sector, at most 2 MAST requests at a time (download.py)
                     -> compact NPZ in DATA/<tag>/lc/ -> ready queue
  search pool        <= 3 spawn processes (numba 1 thread each) take ready NPZs and run
                     searchjob.search_one (frozen method, long + short range)
  main thread        writes one result JSON per star (write-then-rename) and a progress line.

State and resumability:
  * done      = OUT/<tag>/results/tic<T>.json exists (written atomically, last step for a star).
  * restart   finished stars are skipped; a star with an NPZ but no result goes straight to
              search; a star killed mid-download resumes from its per-sector partial cache.
              Stale *.tmp files are removed at start.
  * failures  per-request retries + backoff + circuit breaker live in download.py. A star whose
              sectors still fail is re-queued up to STAR_ATTEMPTS times, at least RETRY_GAP_S
              later. After that it is searched on what it has and tagged "incomplete" (failed
              sectors listed); with no sector at all it is a "failed" record. A worker crash is
              retried once, then "failed". Nothing stops the run. --retry-failed redoes failed
              and incomplete stars.
  * guards    the SHA-256 of gate2/frozen.json and of every search-path source file is stored
              in OUT/<tag>/manifest.json at first start; a restart with different hashes refuses
              to run (--allow-code-change overrides and is logged). A lock file prevents two
              runners on one tag.

Blinding: nothing about signals (period, MES, depth, counts) is logged or printed. The
signals are only in the per-star result files, for the pre-registered vetting.

Usage (from the main checkout; launch.sh does this under systemd):
  python -I spikes/mdwarf_stack/survey/run_survey.py [--tag survey] [--procs 3]
         [--targets CSV] [--min-sectors 6] [--max-stars N] [--retry-failed]
"""

from __future__ import annotations

import os

for _v in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only

import argparse  # noqa: E402
import datetime as dt  # noqa: E402
import fcntl  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import queue  # noqa: E402
import signal  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait  # noqa: E402
from concurrent.futures.process import BrokenProcessPool  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
SPIKE = HERE.parent
REPO = HERE.parents[2]
sys.path.insert(0, str(SPIKE))
sys.path.insert(0, str(SPIKE / "gate2"))
sys.path.insert(0, str(HERE))

from common import DATA, OUT  # noqa: E402
from make_jobs import FALLBACK_M, FALLBACK_R  # noqa: E402

MAX_PROCS = 3
STAR_ATTEMPTS = 3
RETRY_GAP_S = 900.0
READY_MAX = 200  # downloaded-but-unsearched stars the downloader may run ahead by
FROZEN = SPIKE / "gate2" / "frozen.json"
FROZEN_OUT_COPY = OUT / "gate2" / "frozen.json"
DEFAULT_TARGETS = OUT / "gate2" / "survey_targets.csv"
# every file whose content decides a search result
SEARCH_PATH_FILES = [SPIKE / "common.py", SPIKE / "fetch.py", SPIKE / "search.py",
                     SPIKE / "gate2" / "search2.py", SPIKE / "gate2" / "run2.py",
                     REPO / "src" / "oddball" / "tess" / "lightcurve.py",
                     HERE / "download.py", HERE / "searchjob.py"]


def utc() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def atomic_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(obj, fh, default=_jdefault)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)


def _jdefault(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def git_head() -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                           text=True, timeout=10, check=False)
        return r.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


class Log:
    """Run log on stdout (systemd appends it to the log file) + JSONL error/progress files."""

    def __init__(self, out: Path):
        self.lock = threading.Lock()
        self.err = open(out / "errors.jsonl", "a")
        self.prog = open(out / "progress.jsonl", "a")

    def say(self, msg: str):
        with self.lock:
            print(f"{utc()} {msg}", flush=True)

    def error(self, rec: dict):
        with self.lock:
            self.err.write(json.dumps({"utc": utc(), **rec}, default=_jdefault) + "\n")
            self.err.flush()

    def progress(self, rec: dict):
        with self.lock:
            self.prog.write(json.dumps(rec, default=_jdefault) + "\n")
            self.prog.flush()


def progress_line(r: dict) -> dict:
    """The only per-star fields status.py ever sees: no signal information."""
    keep = ("tic", "status", "incomplete", "tags", "n_sectors_pred", "n_sectors_found",
            "n_sector_failures", "dl_wall_s", "search_cpu_s", "search_wall_s",
            "worker_vmhwm_mb", "npz_bytes", "finished_utc", "error_kind", "run_id")
    return {k: r.get(k) for k in keep}


# ------------------------------------------------------------------------------------- targets


def load_targets(path: Path, min_sectors: int) -> pd.DataFrame:
    t = pd.read_csv(path)
    t = t.drop_duplicates("tic")
    t = t[t["n_sectors"] >= min_sectors].copy()
    t["_tic"] = t["tic"].astype(np.int64)
    # most predicted sectors first; ties by TIC so the order is reproducible
    return t.sort_values(["n_sectors", "_tic"], ascending=[False, True]).reset_index(drop=True)


def mass_radius(row) -> tuple[float, float, str]:
    m, r = row.get("mass", np.nan), row.get("rad", np.nan)
    ok = np.isfinite(m) and np.isfinite(r)
    return (float(m) if np.isfinite(m) else FALLBACK_M,
            float(r) if np.isfinite(r) else FALLBACK_R, "TIC" if ok else "fallback_0.3")


# ------------------------------------------------------------------------------------- runner


class Runner:
    def __init__(self, a):
        self.a = a
        self.out = OUT / a.tag
        self.data = DATA / a.tag
        self.res_dir = self.out / "results"
        for d in (self.out, self.res_dir, self.data):
            d.mkdir(parents=True, exist_ok=True)
        self.lockfh = open(self.out / "run.lock", "w")
        try:
            fcntl.flock(self.lockfh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"another runner holds {self.out / 'run.lock'}; exiting", flush=True)
            sys.exit(3)
        self.log = Log(self.out)
        self.run_id = utc()
        self.stop = threading.Event()
        self.ready: queue.Queue = queue.Queue(maxsize=READY_MAX)
        self.dl_done = threading.Event()
        self.dl_crashed = False
        self.retry: list[dict] = []  # stars whose search process died, searched again first
        self.dl_info: dict[int, dict] = {}  # tic -> download facts for the result record
        self.lock = threading.Lock()

    # ---------------------------------------------------------------- setup and guards
    def check_frozen(self) -> dict:
        cfg = json.loads(FROZEN.read_text())
        if FROZEN_OUT_COPY.exists() and json.loads(FROZEN_OUT_COPY.read_text()) != cfg:
            raise SystemExit(f"{FROZEN} differs from {FROZEN_OUT_COPY}; refusing to run")
        for k in ("tau_long_status", "tau_short_status"):
            if cfg.get(k) != "final":
                raise SystemExit(f"frozen.json {k} is {cfg.get(k)!r}, not final")
        if cfg["rot"] != "harm":
            raise SystemExit(f"frozen rot is {cfg['rot']!r}; gate 2 froze 'harm'")
        return cfg

    def manifest(self, cfg: dict, targets: pd.DataFrame, tpath: Path) -> dict:
        m = {"frozen_sha256": sha256(FROZEN),
             "code_sha256": {str(p.relative_to(REPO)): sha256(p) for p in SEARCH_PATH_FILES},
             "rot": cfg["rot"], "max_sig": __import__("searchjob").MAX_SIG,
             "targets": str(tpath), "targets_sha256": sha256(tpath),
             "min_sectors": self.a.min_sectors, "n_targets": int(len(targets)),
             "out_dir": str(self.out), "data_dir": str(self.data)}
        mp = self.out / "manifest.json"
        if mp.exists():
            old = json.loads(mp.read_text())
            diff = [k for k in ("frozen_sha256", "code_sha256", "rot", "max_sig",
                                "targets_sha256", "min_sectors") if old.get(k) != m.get(k)]
            if diff and not self.a.allow_code_change:
                raise SystemExit(f"manifest mismatch on {diff}: the search path or targets "
                                 "changed since this survey started; refusing to mix results "
                                 "(--allow-code-change to override)")
            if diff:
                self.log.say(f"WARNING manifest changed on {diff} (--allow-code-change)")
            m["first_start"] = old.get("first_start")
            m["starts"] = old.get("starts", []) + [{"utc": self.run_id, "git": git_head(),
                                                    "changed": diff}]
        else:
            m["first_start"] = self.run_id
            m["starts"] = [{"utc": self.run_id, "git": git_head(), "changed": []}]
        atomic_json(mp, m)
        return m

    def clean_tmp(self):
        n = 0
        for d in (self.res_dir, self.data / "lc", self.data / "products", self.out):
            if d.exists():
                for p in d.glob("*.tmp*"):
                    p.unlink(missing_ok=True)
                    n += 1
        part = self.data / "partial"
        if part.exists():
            for p in part.glob("*/*.tmp*"):
                p.unlink(missing_ok=True)
                n += 1
        return n

    def resync_progress(self):
        """Results whose progress line was lost (kill between rename and append)."""
        seen = set()
        pp = self.out / "progress.jsonl"
        if pp.exists():
            for line in open(pp):
                try:
                    seen.add(int(json.loads(line)["tic"]))
                except (ValueError, KeyError, TypeError):
                    continue  # a torn last line from a kill
        n = 0
        for f in self.res_dir.glob("tic*.json"):
            tic = int(f.stem[3:])
            if tic not in seen:
                self.log.progress(progress_line(json.loads(f.read_text())))
                n += 1
        return n

    def result_path(self, tic: int) -> Path:
        return self.res_dir / f"tic{tic}.json"

    def is_done(self, tic: int) -> bool:
        p = self.result_path(tic)
        if not p.exists():
            return False
        if self.a.retry_failed:
            r = json.loads(p.read_text())
            if r.get("status") == "failed" or r.get("incomplete"):
                return False
        return True

    # ---------------------------------------------------------------- download stage
    def downloader(self, todo: list[dict]):
        from download import Downloader, classify

        dl = Downloader(self.data, lambda r: self.log.error(r))
        deferred: list[tuple[float, dict]] = []  # (not before, star)
        pending = list(todo)
        try:
            while (pending or deferred) and not self.stop.is_set():
                now = time.time()
                if pending:
                    star = pending.pop(0)
                else:
                    deferred.sort(key=lambda x: x[0])
                    if deferred[0][0] > now:
                        time.sleep(min(30.0, deferred[0][0] - now))
                        continue
                    star = deferred.pop(0)[1]
                tic = star["tic"]
                w0 = time.time()
                npz = dl.npz_path(tic)
                if npz.exists() and not star.get("redo_download"):
                    self._push(star, {"npz": npz, "dl_wall_s": None, "cached_npz": True})
                    continue
                star["attempt"] = star.get("attempt", 0) + 1
                try:
                    prods, queried = dl.products(tic)
                except Exception as e:  # noqa: BLE001
                    self.log.error({"tic": tic, "kind": "star_query_failed", "final": True,
                                    "attempt": star["attempt"], "msg": repr(e)[:200]})
                    if star["attempt"] < STAR_ATTEMPTS:
                        deferred.append((time.time() + RETRY_GAP_S, star))
                    else:
                        self._finish_nodl(star, "failed", "query", repr(e)[:300])
                    continue
                if not prods:
                    self._finish_nodl(star, "no_data", None, "no SPOC 2-min or TESS-SPOC product")
                    continue
                recs, metas, fails = dl.fetch_star(tic, prods)
                for f in fails:
                    self.log.error({"tic": tic, "sector": f["sector"], "kind": f["kind"],
                                    "final": star["attempt"] >= STAR_ATTEMPTS,
                                    "attempt": star["attempt"], "msg": f["error"]})
                if fails and star["attempt"] < STAR_ATTEMPTS:
                    # successful sectors stay in the partial cache; retry the rest later
                    deferred.append((time.time() + RETRY_GAP_S, star))
                    self.log.say(f"defer tic={tic} sector_failures={len(fails)} "
                                 f"attempt={star['attempt']}")
                    continue
                if not recs:
                    self._finish_nodl(star, "failed", fails[0]["kind"] if fails else "empty",
                                      "every sector failed", fails)
                    continue
                npz = dl.write_npz(tic, recs, metas, fails)
                self._push(star, {"npz": npz, "dl_wall_s": time.time() - w0,
                                  "cached_npz": False, "products_queried_utc": queried,
                                  "n_listed": len(prods)})
        except Exception as e:  # noqa: BLE001 - the downloader must never take the run down
            self.log.say(f"DOWNLOADER CRASH {classify(e)} {e!r:.300}")
            self.log.error({"kind": "downloader_crash", "final": True, "msg": repr(e)[:300]})
            self.dl_crashed = True
        finally:
            self.dl_done.set()

    def _push(self, star, info):
        with self.lock:
            self.dl_info[star["tic"]] = info
        while not self.stop.is_set():
            try:
                self.ready.put(star, timeout=5)
                return
            except queue.Full:
                continue

    def _finish_nodl(self, star, status, kind, msg, fails=None):
        r = self._base_record(star)
        r.update(status=status, error_kind=kind, error=msg, n_sectors_found=0,
                 sector_failures=fails or [], n_sector_failures=len(fails or []),
                 incomplete=bool(fails), tags=["no_data"] if status == "no_data" else [])
        self._write(r)

    # ---------------------------------------------------------------- records
    def _base_record(self, star) -> dict:
        return {"tic": star["tic"], "run_id": self.run_id, "frozen_sha256": self.frozen_sha,
                "code_sha256": self.code_sha, "git_head": self.git,
                "n_sectors_pred": star["n_pred"], "sectors_pred": star["sectors_pred"],
                "mass": star["mass"], "radius": star["radius"], "mr_source": star["mr_source"],
                "tags": [], "incomplete": False}

    def _write(self, r: dict):
        r["finished_utc"] = utc()
        atomic_json(self.result_path(r["tic"]), r)
        self.log.progress(progress_line(r))
        with self.lock:
            self.n_done += 1
        sec = f"{r.get('n_sectors_found', 0)}/{r.get('n_sectors_pred')}"
        cpu = r.get("search_cpu_s")
        dlw = r.get("dl_wall_s")
        self.log.say(f"done tic={r['tic']} status={r['status']} sec={sec} "
                     f"dl={'-' if dlw is None else f'{dlw:.0f}s'} "
                     f"cpu={'-' if cpu is None else f'{cpu:.0f}s'} tags={','.join(r['tags'])} "
                     f"[{self.n_done}/{self.n_todo}]")

    def _search_record(self, star, info, res: dict | None, err: str | None) -> dict:
        from searchjob import load_npz

        r = self._base_record(star)
        npz = Path(info["npz"])
        try:
            _, meta = load_npz(npz)
        except Exception as e:  # noqa: BLE001 - unreadable NPZ: fail the star, refetch later
            npz.unlink(missing_ok=True)
            r.update(status="failed", error_kind="npz_unreadable", error=repr(e)[:300],
                     n_sectors_found=0)
            return r
        found = [int(s["sector"]) for s in meta["sectors"]]
        fails = meta.get("failures", [])
        r.update(npz=str(npz), npz_bytes=npz.stat().st_size,
                 download_utc=meta.get("downloaded_utc"),
                 products_queried_utc=info.get("products_queried_utc"),
                 dl_wall_s=info.get("dl_wall_s"), cached_npz=info.get("cached_npz"),
                 n_sectors_found=len(found), sectors_found=found,
                 products={str(s["sector"]): s.get("product") for s in meta["sectors"]},
                 provenance=[{k: s.get(k) for k in ("sector", "product", "procver", "data_rel",
                                                     "cadence_s", "uri", "downloaded_utc")}
                             for s in meta["sectors"]],
                 sector_failures=fails, n_sector_failures=len(fails), incomplete=bool(fails))
        tags = []
        if fails:
            tags.append("incomplete")
        if len(found) < 6:
            tags.append("lt6_real_sectors")
        if star["mr_source"] != "TIC":
            tags.append("fallback_mass_radius")
        missing = sorted(set(star["sectors_pred"]) - set(found))
        extra = sorted(set(found) - set(star["sectors_pred"]))
        r.update(sectors_missing_vs_pred=missing, sectors_extra_vs_pred=extra)
        if err is not None:
            r.update(status="failed", error_kind=err.split(":", 1)[0], error=err, tags=tags)
            return r
        r.update(status="ok", tags=tags, rot=res["rot"], search=res["search"],
                 search_cpu_s=res["cpu_s"], search_wall_s=res["wall_s"],
                 pre_cpu_s=res["cpu_pre"], n_bins_raw=res["n_bins_raw"],
                 worker_vmhwm_mb=res["worker_vmhwm_mb"])
        return r

    # ---------------------------------------------------------------- search stage
    def _pool(self):
        return ProcessPoolExecutor(self.a.procs, mp_context=get_context("spawn"),
                                   max_tasks_per_child=25)

    def run(self):
        a = self.a
        cfg = self.check_frozen()
        tpath = Path(a.targets)
        targets = load_targets(tpath, a.min_sectors)
        m = self.manifest(cfg, targets, tpath)
        self.frozen_sha, self.code_sha, self.git = (m["frozen_sha256"], m["code_sha256"],
                                                    m["starts"][-1]["git"])
        nt = self.clean_tmp()
        nr = self.resync_progress()
        todo = []
        for _, row in targets.iterrows():
            tic = int(row["_tic"])
            if self.is_done(tic):
                continue
            mass, rad, src = mass_radius(row)
            sp = [int(x) for x in str(row.get("sectors", "")).split()] if "sectors" in row else []
            # --retry-failed: an earlier failed/incomplete result means its NPZ is re-fetched
            redo = self.result_path(tic).exists()
            todo.append({"tic": tic, "n_pred": int(row["n_sectors"]), "sectors_pred": sp,
                         "mass": mass, "radius": rad, "mr_source": src, "redo_download": redo})
        if a.max_stars is not None:
            todo = todo[: a.max_stars]
        self.n_todo, self.n_done = len(todo), 0
        self.log.say(f"start run_id={self.run_id} tag={a.tag} targets={len(targets)} "
                     f"todo={len(todo)} procs={a.procs} frozen={self.frozen_sha[:12]} "
                     f"git={self.git} removed_tmp={nt} resynced={nr}")
        if not todo:
            self.log.say("nothing to do")
            return 0
        dth = threading.Thread(target=self.downloader, args=(todo,), daemon=True, name="dlstar")
        dth.start()
        inflight = {}  # future -> (star, info)
        crashes: dict[int, int] = {}
        pool = self._pool()
        try:
            while not self.stop.is_set():
                while len(inflight) < a.procs:
                    if self.retry:
                        star = self.retry.pop(0)
                    else:
                        try:
                            star = self.ready.get_nowait()
                        except queue.Empty:
                            break
                    with self.lock:
                        info = self.dl_info[star["tic"]]
                    f = pool.submit(_search, star["tic"], str(info["npz"]), star["mass"],
                                    star["radius"], cfg)
                    inflight[f] = (star, info)
                if not inflight:
                    if self.dl_done.is_set() and self.ready.empty() and not self.retry:
                        break
                    time.sleep(1.0)
                    continue
                done, _ = wait(list(inflight), timeout=10, return_when=FIRST_COMPLETED)
                broken = False
                for f in done:
                    star, info = inflight.pop(f)
                    try:
                        res = f.result()
                        self._write(self._search_record(star, info, res, None))
                    except BrokenProcessPool:
                        broken = True
                        self._crash(star, info, crashes)
                    except Exception as e:  # noqa: BLE001
                        self.log.error({"tic": star["tic"], "kind": "search_error",
                                        "final": True, "msg": repr(e)[:300]})
                        self._write(self._search_record(star, info, None,
                                                        f"search_error: {e!r:.300}"))
                if broken:
                    for f, (star, info) in list(inflight.items()):
                        self._crash(star, info, crashes)
                    inflight.clear()
                    _kill_pool(pool)
                    pool = self._pool()
                    self.log.say("search pool rebuilt after a worker died")
        finally:
            self.stop.set()
            _kill_pool(pool)
        if self.dl_crashed:
            self.log.say("exiting with status 1 after a downloader crash (systemd restarts)")
            return 1
        self.log.say(f"finished run_id={self.run_id} done={self.n_done}/{self.n_todo}")
        return 0

    def _crash(self, star, info, crashes):
        tic = star["tic"]
        crashes[tic] = crashes.get(tic, 0) + 1
        self.log.error({"tic": tic, "kind": "worker_crash", "final": crashes[tic] >= 2,
                        "attempt": crashes[tic], "msg": "search process died (OOM kill?)"})
        if crashes[tic] >= 2:
            self._write(self._search_record(star, info, None, "worker_crash: died twice"))
        else:
            self.retry.append(star)


def _kill_pool(pool):
    """Stop a pool now: running searches are abandoned (their stars have no result yet, so
    they are redone on the next start)."""
    procs = list((getattr(pool, "_processes", None) or {}).values())
    pool.shutdown(wait=False, cancel_futures=True)
    for p in procs:
        if p.is_alive():
            p.terminate()
    for p in procs:
        p.join(timeout=10)


def _search(tic, npz, mass, radius, cfg):
    from searchjob import search_one

    return search_one(tic, npz, mass, radius, cfg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="survey", help="output/data subdirectory name")
    ap.add_argument("--targets", default=str(DEFAULT_TARGETS))
    ap.add_argument("--min-sectors", type=int, default=6, help="on PREDICTED sectors")
    ap.add_argument("--procs", type=int, default=MAX_PROCS)
    ap.add_argument("--max-stars", type=int, default=None)
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--allow-code-change", action="store_true")
    a = ap.parse_args()
    a.procs = max(1, min(a.procs, MAX_PROCS))  # machine rule: at most 3 CPU workers
    r = Runner(a)

    def _term(signum, frame):
        r.log.say(f"signal {signum}: stopping (finished stars are safe; the rest resume)")
        r.stop.set()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _term)
    return r.run()


if __name__ == "__main__":
    sys.exit(main())
