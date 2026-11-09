"""Kaggle kernel: ingest one TESS-SPOC sector with Kaggle's bandwidth.

Do not run this by hand. ``scripts/kaggle_ingest.py`` fills in the CONFIG line below and pushes
it as a private, CPU-only, internet-enabled script kernel. The kernel:

1. ``pip install``s oddball-worlds from GitHub at a pinned commit SHA;
2. runs ``oddball.tess.ingest.ingest_sector`` into ``/kaggle/working/tess-spoc/sNNNN``, with a
   wall-clock deadline below Kaggle's session limit (on the deadline the buffer is flushed and
   the run is marked partial, so the output still saves and a local run can resume it);
3. retries transient failures in up to ``retry_passes`` extra passes;
4. gzips the 40-50 MB URL list and adds a ``kaggle`` block to ``provenance.json`` (git SHA,
   kernel slug, environment, network bytes received, timings).

Everything it keeps is in ``/kaggle/working``, which Kaggle saves as the kernel output.
"""

import _thread
import datetime as dt
import gzip
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import threading
import time

# scripts/kaggle_ingest.py replaces the next line with the run config.
CONFIG = {}  # @@CONFIG@@

REPO = "https://github.com/Antimatter543/oddball-worlds"
WORK = "/kaggle/working"
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:8.0f}s] {msg}", flush=True)


def utcnow():
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def net_rx_bytes():
    """Bytes received on all non-loopback interfaces (to measure MAST throughput)."""
    total = 0
    try:
        with open("/proc/net/dev") as fh:
            for line in fh.readlines()[2:]:
                name, rest = line.split(":", 1)
                if name.strip() != "lo":
                    total += int(rest.split()[0])
    except OSError:
        return None
    return total


def kaggle_env():
    """KAGGLE_* environment (run type, URL base, ...), never anything secret-looking."""
    bad = ("TOKEN", "SECRET", "KEY", "PASSWORD", "CREDENTIAL", "AUTH")
    return {
        k: v
        for k, v in sorted(os.environ.items())
        if k.startswith("KAGGLE_") and not any(b in k for b in bad)
    }


def cpu_seconds():
    """User + system CPU of this process, all threads (CPU-bound vs network-bound check)."""
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_utime + ru.ru_stime


def disk_free_gb(path):
    return round(shutil.disk_usage(path).free / 1e9, 2)


def main():
    cfg = CONFIG
    sector, sha = int(cfg["sector"]), cfg["sha"]
    log(
        f"sector {sector}, oddball-worlds@{sha}, python {platform.python_version()}, "
        f"cpus {os.cpu_count()}, free disk {disk_free_gb(WORK)} GB"
    )

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check",
         f"oddball-worlds @ git+{REPO}@{sha}"],
        check=True,
    )  # fmt: skip
    os.environ["ODDBALL_DATA"] = WORK  # data_root() reads this at call time

    import oddball
    from oddball.tess import ingest

    log(f"installed oddball from {os.path.dirname(oddball.__file__)}")

    # Deadline: interrupt the main thread, which ingest_sector turns into flush + finalize.
    deadline = float(cfg["deadline_hours"]) * 3600 - (time.time() - T0)
    hit_deadline = threading.Event()

    def on_deadline():
        hit_deadline.set()
        log("deadline reached: interrupting ingest")
        _thread.interrupt_main()

    timer = threading.Timer(max(60.0, deadline), on_deadline)
    timer.daemon = True
    timer.start()

    rx0, t_ingest, cpu0 = net_rx_bytes(), time.time(), cpu_seconds()
    passes = []
    status = "complete"
    include = {int(t) for t in cfg.get("include_tics", [])}
    for p in range(1 + int(cfg["retry_passes"])):
        try:
            summary = ingest.ingest_sector(
                sector,
                limit=cfg.get("limit"),
                include_tics=include,
                workers=int(cfg["workers"]),
                shard_size=int(cfg["shard_size"]),
                log=log,
            )
        except (SystemExit, KeyboardInterrupt):  # interrupted: finalize already ran
            status = "partial_deadline" if hit_deadline.is_set() else "interrupted"
            passes.append({"pass": p, "status": status})
            break
        last = summary["runs"][-1]
        passes.append({"pass": p, "converted": last["converted"],
                       "failed_transient": last["failed_transient"],
                       "failed_permanent": last["failed_permanent"]})  # fmt: skip
        if last["failed_transient"] == 0:
            break
        log(f"pass {p}: {last['failed_transient']} transient failures; retrying")
    timer.cancel()
    rx1, t_end, cpu1 = net_rx_bytes(), time.time(), cpu_seconds()

    sdir = os.path.join(WORK, "tess-spoc", f"s{sector:04d}")
    script = os.path.join(sdir, "dl-lc.sh")
    if os.path.exists(script):  # keep the exact URL list, compressed (~10x smaller)
        with open(script, "rb") as src, gzip.open(script + ".gz", "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.remove(script)

    prov_path = os.path.join(sdir, "provenance.json")
    if not os.path.exists(prov_path):  # interrupted before ingest_sector reached finalize
        os.makedirs(sdir, exist_ok=True)
        with open(prov_path, "w") as fh:
            json.dump({"sector": sector, "runs": [], "n_converted": 0, "n_in_script": None,
                       "n_shards": 0, "bytes_shards": 0}, fh)  # fmt: skip
    with open(prov_path) as fh:
        prov = json.load(fh)
    got = None if rx0 is None or rx1 is None else rx1 - rx0
    secs = t_end - t_ingest
    prov.setdefault("kaggle", []).append({
        "status": status,
        "git_sha": sha,
        "repo": REPO,
        "kernel": cfg.get("kernel"),
        "pushed_utc": cfg.get("pushed_utc"),
        "finished_utc": utcnow(),
        "limit": cfg.get("limit"),
        "workers": cfg["workers"],
        "passes": passes,
        "ingest_seconds": round(secs, 1),
        "total_seconds": round(t_end - T0, 1),
        "net_rx_bytes": got,
        "net_rx_mbit_s": None if got is None else round(got * 8 / 1e6 / max(secs, 1e-9), 1),
        "cpu_seconds": round(cpu1 - cpu0, 1),
        "cpu_util": round((cpu1 - cpu0) / max(secs * (os.cpu_count() or 1), 1e-9), 3),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "free_disk_gb_after": disk_free_gb(WORK),
        "env": kaggle_env(),
    })  # fmt: skip
    tmp = prov_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(prov, fh, indent=2)
    os.replace(tmp, prov_path)
    log(
        f"done: status={status} converted={prov['n_converted']}/{prov['n_in_script']} "
        f"shards={prov['n_shards']} bytes={prov['bytes_shards']} rx={got} in {secs:.0f}s"
    )


if __name__ == "__main__":
    main()
