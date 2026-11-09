"""Ingest TESS-SPOC sectors on Kaggle (their bandwidth), then pull only the Parquet output.

A sector is ~67 GB of FITS in but ~2.7 GB of Parquet out, so the heavy download happens in a
private, CPU-only Kaggle kernel (``scripts/kaggle/ingest_kernel.py``) and only the output comes
home. Needs the ``cloud`` dependency group and an authenticated Kaggle CLI (``~/.kaggle``)::

    uv run --group cloud python -I scripts/kaggle_ingest.py run 33 34 35 --max-concurrent 4
    uv run --group cloud python -I scripts/kaggle_ingest.py run 33 --limit 2000   # quick slice
    uv run --group cloud python -I scripts/kaggle_ingest.py status 33 34 35
    uv run --group cloud python -I scripts/kaggle_ingest.py download 33   # re-pull a finished run
    uv run --group cloud python -I scripts/kaggle_ingest.py verify 33

``run`` pushes one kernel per sector (at most ``--max-concurrent`` running at once), polls them,
downloads each finished output into ``<data>/tess-spoc/.kaggle-staging/sNNNN`` (resumable,
size-checked), verifies it (shard/row counts, provenance, every shard decodes) and only then
moves it to ``<data>/tess-spoc/sNNNN``. An existing sector dir is never overwritten.
The kernel pip-installs the repo at a pinned commit, so that commit must be on GitHub.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
# scripts/kaggle/ would shadow the kaggle package if this dir were on sys.path (it is not under -I).
sys.path[:] = [p for p in sys.path if not p or Path(p).resolve() != HERE]
sys.path.insert(0, str(HERE.parent / "src"))

REPO_URL = "https://github.com/Antimatter543/oddball-worlds"
TEMPLATE = HERE / "kaggle" / "ingest_kernel.py"
KNOWN = HERE.parent / "docs" / "known_positives.csv"
TERMINAL = {"COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED"}
# Push errors that mean "too many sessions right now": wait and retry rather than fail.
BUSY_WORDS = ("maximum", "concurrent", "session count", "quota", "limit", "too many")


def log(msg: str) -> None:
    print(f"{dt.datetime.now():%H:%M:%S} {msg}", flush=True)


def utcnow() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------------------------
# Kaggle API (imported lazily so `verify` works without the cloud group)


def kaggle_api():
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def kaggle_user(api) -> str:
    user = api.get_config_value(api.CONFIG_NAME_USER)
    if not user:
        raise SystemExit("could not read the Kaggle username from the CLI config")
    return user


def resolve_sha(sha: str | None) -> str:
    """Default: current GitHub main. Either way, check the commit exists on GitHub (the kernel
    pip-installs from there, so an unpushed local commit would fail inside the kernel)."""
    import requests

    if not sha:
        out = subprocess.run(["git", "ls-remote", REPO_URL, "refs/heads/main"],
                             capture_output=True, text=True, check=True).stdout  # fmt: skip
        sha = out.split()[0]
    api_url = REPO_URL.replace("github.com", "api.github.com/repos") + f"/commits/{sha}"
    try:
        r = requests.get(api_url, timeout=30)
    except requests.RequestException as e:
        log(f"warning: could not check {sha} on GitHub ({e}); continuing")
        return sha
    if r.status_code in (404, 422):
        raise SystemExit(f"commit {sha} is not on GitHub; push it first or pass --sha")
    if r.ok:
        sha = r.json()["sha"]  # expands a short SHA
    else:
        log(f"warning: GitHub API said {r.status_code} for {sha}; continuing unchecked")
    return sha


def known_tics(path: Path = KNOWN) -> set[int]:
    if not path.exists():
        return set()
    with path.open() as fh:
        return {int(r["tic"]) for r in csv.DictReader(fh) if r.get("tic")}


@dataclass
class Job:
    sector: int
    slug: str
    ref: str
    pushed_utc: str
    pushed_at: float
    version: int | None = None
    seen_active: bool = False
    status: str = "PUSHED"
    info: dict = field(default_factory=dict)


def push_kernel(api, user: str, sector: int, a, sha: str, include: set[int]) -> Job | None:
    """Push and start a kernel version. Returns None if Kaggle says we're at the session cap."""
    from oddball.common.paths import data_root
    from oddball.tess import cloud

    slug = cloud.kernel_slug(sector, a.limit)
    folder = data_root() / "kaggle" / "kernels" / slug
    folder.mkdir(parents=True, exist_ok=True)
    pushed = utcnow()
    cfg = cloud.kernel_config(
        sector, sha, f"{user}/{slug}", pushed, limit=a.limit, include_tics=sorted(include),
        workers=a.workers, shard_size=a.shard_size, deadline_hours=a.deadline_hours,
        retry_passes=a.retry_passes,
    )  # fmt: skip
    (folder / cloud.KERNEL_FILE).write_text(cloud.render_kernel(TEMPLATE.read_text(), cfg))
    meta = cloud.kernel_metadata(user, sector, a.limit)
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    resp = api.kernels_push(str(folder))
    err = getattr(resp, "error", None)
    if err:
        if any(w in err.lower() for w in BUSY_WORDS):
            log(f"s{sector:04d}: Kaggle busy ({err}); will retry")
            return None
        raise SystemExit(f"s{sector:04d}: push failed: {err}")
    log(f"s{sector:04d}: pushed {resp.ref} v{resp.version_number} (sha {sha[:10]}) {resp.url}")
    return Job(sector, slug, f"{user}/{slug}", pushed, time.time(), resp.version_number)


def kernel_status(api, ref: str) -> tuple[str, str]:
    r = api.kernels_status(ref)
    st = r.status
    name = getattr(st, "name", str(st)).upper()
    return name, (getattr(r, "failure_message", "") or "")


# --------------------------------------------------------------------------------------------
# Download


def list_outputs(api, ref: str) -> tuple[list, str]:
    """All output files of the latest run (paged), plus the run log."""
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

    owner, slug = ref.split("/")
    files, log_text, token = [], "", ""
    with api.build_kaggle_client() as kc:
        while True:
            req = ApiListKernelSessionOutputRequest()
            req.user_name, req.kernel_slug = owner, slug
            req.page_size = 200
            if token:
                req.page_token = token
            resp = kc.kernels.kernels_api_client.list_kernel_session_output(req)
            files += list(resp.files or [])
            log_text = log_text or (resp.log or "")
            token = resp.next_page_token
            if not token:
                return files, log_text


def fetch(url: str, dst: Path, retries: int = 5) -> int:
    """Stream ``url`` to ``dst`` via ``dst.part`` and rename; returns bytes fetched (0 if a
    complete copy was already there). The byte count is checked against Content-Length when the
    server sends one (the kaggle CLI crashes when it doesn't); parquet decoding in the verifier
    is the backstop for truncation either way."""
    import requests

    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()
                size = int(r.headers.get("Content-Length") or -1)
                if dst.exists() and dst.stat().st_size == size:
                    return 0
                n = 0
                with part.open("wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
                        n += len(chunk)
            if size >= 0 and n != size:
                raise OSError(f"got {n} bytes, Content-Length {size}")
            os.replace(part, dst)
            return n
        except (requests.RequestException, OSError) as e:
            last = e
            log(f"  retry {attempt + 1}/{retries} {dst.name}: {e}")
            time.sleep(min(60, 5 * 2**attempt))
    part.unlink(missing_ok=True)
    raise ConnectionError(f"failed to download {dst.name}: {last}")


def _log_text(raw: str) -> str:
    """Kaggle returns the run log as a JSON list of {stream_name, time, data}; flatten it."""
    try:
        return "".join(e.get("data", "") for e in json.loads(raw))
    except (ValueError, TypeError, AttributeError):
        return raw


def _last_kaggle(prov: dict) -> dict:
    return (prov.get("kaggle") or [{}])[-1]


def download_and_install(
    api, sector: int, ref: str, limit: int | None, nonce: str | None = None,
    version: int | None = None,
) -> dict:  # fmt: skip
    import requests

    from oddball.common.paths import data_root, sector_dir
    from oddball.tess import cloud
    from oddball.tess.verify import verify_sector_dir

    tag = f"s{sector:04d}" + (f"-l{limit}" if limit else "")
    staging = data_root() / "tess-spoc" / ".kaggle-staging" / tag
    files, log_raw = list_outputs(api, ref)
    want = {cloud.output_relpath(f.file_name, sector): f.url for f in files}
    want.pop(None, None)
    if "provenance.json" not in want:
        raise SystemExit(f"{tag}: kernel output has no tess-spoc/s{sector:04d}/provenance.json")

    # Provenance first (small): identifies the run before pulling GBs, and gives the size.
    r = requests.get(want["provenance.json"], timeout=120)
    r.raise_for_status()
    remote = r.json()
    kag = _last_kaggle(remote)
    if nonce and kag.get("pushed_utc") != nonce:
        raise SystemExit(f"{tag}: output is from push {kag.get('pushed_utc')}, not {nonce} (stale)")
    old = staging / "provenance.json"
    if old.exists():
        old_push = _last_kaggle(json.loads(old.read_text())).get("pushed_utc")
        if old_push != kag.get("pushed_utc"):
            log(f"{tag}: staging holds a different run ({old_push}); clearing it")
            shutil.rmtree(staging)
    total = int(remote.get("bytes_shards", 0)) + 2 * int(remote.get("bytes_metadata", 0))
    free = shutil.disk_usage(data_root()).free
    if free < total * 1.2 + 1e9:
        raise SystemExit(f"{tag}: only {free / 1e9:.1f} GB free locally, need ~{total / 1e9:.1f}")
    log(f"{tag}: {len(want)} files, ~{total / 1e9:.2f} GB, kernel status {kag.get('status')} "
        f"-> {staging}")  # fmt: skip

    t0, got = time.time(), 0
    for i, rel in enumerate(sorted(want), 1):
        got += fetch(want[rel], staging / rel)
        if i % 10 == 0 or i == len(want):
            el = time.time() - t0
            log(f"  {i}/{len(want)} files, {got / 1e9:.2f} GB in {el:.0f}s "
                f"({got * 8 / 1e6 / max(el, 1e-9):.0f} Mbit/s)")  # fmt: skip
    secs = time.time() - t0
    (staging / "kaggle.log").write_text(_log_text(log_raw))

    rep = verify_sector_dir(staging, sector)
    prov = json.loads((staging / "provenance.json").read_text())
    if _last_kaggle(prov).get("pushed_utc") != kag.get("pushed_utc"):
        rep["problems"].append("provenance changed between listing and download")
    if rep["problems"]:
        raise SystemExit(f"{tag}: verification FAILED, left in {staging}: {rep['problems']}")
    if kag.get("status") != "complete":
        log(f"{tag}: WARNING kernel status {kag.get('status')!r}: output is partial; a local "
            "`scripts/ingest_sector.py` run on the installed dir resumes it")  # fmt: skip

    prov["kaggle_download"] = {"kernel": ref, "version": version, "downloaded_utc": utcnow(),
                               "bytes": got, "seconds": round(secs, 1)}  # fmt: skip
    (staging / "provenance.json").write_text(json.dumps(prov, indent=2))
    dest = sector_dir(sector)
    if limit:
        dest = dest.with_name(dest.name + f"-l{limit}")
    if dest.exists():
        log(f"{tag}: {dest} already exists; verified output left in {staging} (not overwritten)")
        final = staging
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, dest)
        final = dest
    summary = {
        "sector": sector, "limit": limit, "kernel": ref, "version": version, "dir": str(final),
        "installed": final == dest,
        "status": kag.get("status"), "rows": rep["n_rows"], "n_in_script": rep["n_in_script"],
        "shards": rep["n_shards"], "bytes_shards": rep["bytes_shards"], "bytes_downloaded": got,
        "download_seconds": round(secs, 1), "kaggle_ingest_seconds": kag.get("ingest_seconds"),
        "kaggle_total_seconds": kag.get("total_seconds"),
        "kaggle_net_rx_bytes": kag.get("net_rx_bytes"), "kaggle_mbit_s": kag.get("net_rx_mbit_s"),
        "kaggle_cpu_util": kag.get("cpu_util"), "failures": rep["n_failures"],
        "finished_utc": utcnow(),
    }  # fmt: skip
    runs = data_root() / "kaggle" / "runs.jsonl"
    runs.parent.mkdir(parents=True, exist_ok=True)
    with runs.open("a") as fh:
        fh.write(json.dumps(summary) + "\n")
    log(f"{tag}: OK {json.dumps(summary)}")
    return summary


# --------------------------------------------------------------------------------------------
# Commands


def cmd_run(a) -> None:
    api = kaggle_api()
    user = kaggle_user(api)
    sha = resolve_sha(a.sha)
    include = set(a.tic) | (known_tics() if a.include_known else set())
    queue, active, done = deque(dict.fromkeys(a.sectors)), {}, []
    busy_since = None  # all slots taken by sessions that aren't ours
    log(f"{len(queue)} sector(s), max {a.max_concurrent} concurrent, sha {sha}, user {user}")
    while queue or active:
        finished = []
        for s, job in list(active.items()):
            try:
                st, msg = kernel_status(api, job.ref)
            except Exception as e:  # noqa: BLE001 - transient API error: keep polling
                log(f"s{s:04d}: status error {e}")
                continue
            if st not in TERMINAL:
                job.seen_active = True
            elif not job.seen_active and time.time() - job.pushed_at < 600:
                continue  # may still be the previous version's status
            if st != job.status:
                el = (time.time() - job.pushed_at) / 60
                log(f"s{s:04d}: {st} after {el:.1f} min {msg}".rstrip())
                job.status = st
            if st in TERMINAL:
                finished.append(active.pop(s))
        while queue and len(active) < a.max_concurrent:
            job = push_kernel(api, user, queue[0], a, sha, include)
            if job is None:
                if not active:
                    busy_since = busy_since or time.time()
                    if time.time() - busy_since > 3600:
                        raise SystemExit(
                            "Kaggle CPU sessions held by other kernels for 1 h; "
                            f"not started: {list(queue)}"
                        )
                break
            busy_since = None
            active[queue.popleft()] = job
        for job in finished:
            if job.status != "COMPLETE":
                log(
                    f"s{job.sector:04d}: kernel ended {job.status}; see "
                    f"`kaggle kernels logs {job.ref}`; output (if any) not pulled"
                )
                done.append({"sector": job.sector, "status": job.status})
            elif a.no_download:
                done.append({"sector": job.sector, "status": "COMPLETE (not downloaded)"})
            else:
                try:
                    done.append(download_and_install(api, job.sector, job.ref, a.limit,
                                                     job.pushed_utc, job.version))  # fmt: skip
                except (SystemExit, Exception) as e:  # noqa: BLE001 - keep other sectors going
                    log(f"s{job.sector:04d}: download/verify failed: {e}")
                    done.append({"sector": job.sector, "status": f"download failed: {e}"})
        if (queue or active) and not finished:
            time.sleep(a.poll)
    log("all done:\n" + "\n".join(json.dumps(d) for d in done))


def cmd_status(a) -> None:
    from oddball.tess import cloud

    api = kaggle_api()
    user = kaggle_user(api)
    for s in a.sectors:
        ref = f"{user}/{cloud.kernel_slug(s, a.limit)}"
        try:
            st, msg = kernel_status(api, ref)
        except Exception as e:  # noqa: BLE001
            st, msg = "UNKNOWN", str(e)
        print(f"s{s:04d} {ref}: {st} {msg}".rstrip())


def cmd_download(a) -> None:
    from oddball.tess import cloud

    api = kaggle_api()
    user = kaggle_user(api)
    for s in a.sectors:
        ref = f"{user}/{cloud.kernel_slug(s, a.limit)}"
        st, msg = kernel_status(api, ref)
        if st != "COMPLETE":
            raise SystemExit(f"{ref} is {st} {msg}")
        download_and_install(api, s, ref, a.limit)


def cmd_verify(a) -> None:
    from oddball.common.paths import sector_dir
    from oddball.tess.verify import verify_sector_dir

    bad = 0
    for s in a.sectors:
        sdir = Path(a.dir) if a.dir else sector_dir(s)
        rep = verify_sector_dir(sdir, s)
        print(json.dumps(rep, indent=2))
        bad += not rep["ok"]
    sys.exit(1 if bad else 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data-root", help="override $ODDBALL_DATA / <repo>/data for this run")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="push, poll, download and verify one kernel per sector")
    r.add_argument("sectors", type=int, nargs="+")
    r.add_argument("--limit", type=int, default=None, help="only the first N LCs (test slice)")
    r.add_argument("--include-known", action="store_true", help="add known positives (w/ --limit)")
    r.add_argument("--tic", type=int, action="append", default=[], help="extra TIC(s) to include")
    r.add_argument(
        "--max-concurrent",
        type=int,
        default=4,
        help="kernels running at once (Kaggle caps batch CPU sessions at 5 per account; "
        "default 4 leaves one for interactive use)",
    )
    r.add_argument("--sha", help="commit to pip-install (default: GitHub main)")
    r.add_argument("--workers", type=int, default=32)
    r.add_argument("--shard-size", type=int, default=10_000)
    r.add_argument(
        "--deadline-hours",
        type=float,
        default=11.5,
        help="stop ingesting and save output after this long (Kaggle cap is 12 h)",
    )
    r.add_argument("--retry-passes", type=int, default=2, help="extra passes over transient fails")
    r.add_argument("--poll", type=float, default=60.0, help="seconds between status polls")
    r.add_argument("--no-download", action="store_true", help="leave the output on Kaggle")
    r.set_defaults(fn=cmd_run)

    for name, fn, help_ in (
        ("status", cmd_status, "status of each sector's kernel"),
        ("download", cmd_download, "pull + verify a finished kernel"),
    ):
        p = sub.add_parser(name, help=help_)
        p.add_argument("sectors", type=int, nargs="+")
        p.add_argument("--limit", type=int, default=None)
        p.set_defaults(fn=fn)

    v = sub.add_parser("verify", help="verify local sector dir(s)")
    v.add_argument("sectors", type=int, nargs="+")
    v.add_argument("--dir", help="explicit directory (default: the sector dir)")
    v.set_defaults(fn=cmd_verify)

    a = ap.parse_args()
    if a.data_root:
        os.environ["ODDBALL_DATA"] = str(Path(a.data_root).expanduser().resolve())
    a.fn(a)


if __name__ == "__main__":
    main()
