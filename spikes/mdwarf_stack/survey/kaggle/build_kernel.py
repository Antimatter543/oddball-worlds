"""Build the Kaggle script-kernel folders for survey shards (nothing is pushed here).

For each shard: <out>/<slug>/survey_shard.py (kernel_template.py with the payload filled in)
and kernel-metadata.json (from kernel-metadata.template.json: private, internet on, no GPU).
The folders are written outside git (default OUT/<tag>/kaggle_kernels/) because the metadata
holds the Kaggle account name, read from the local Kaggle CLI config at build time.

Code: --mode git (default) clones the public repo at --commit, which must already be pushed;
--mode embed ships `git archive <commit>` of the spike code inside the script instead (used for
the smoke test, before the survey branch is merged and pushed).
Targets: the rows of --targets with >= --min-sectors predicted sectors, embedded gzip+base64.
Resume: --skip-from-local embeds the TICs that already have a local result for that shard.

Usage:
  python -I survey/kaggle/build_kernel.py --of 6 [--shards 0,1] [--mode git --commit SHA]
  python -I survey/kaggle/build_kernel.py --of 1 --targets HOSTS.csv --min-sectors 0 \
      --mode embed --tag survey_kaggle_smoke --budget-hours 1.5 --slug transit-survey-smoke
Then: kaggle kernels push -p <folder>
"""

from __future__ import annotations

import argparse
import base64
import gzip
import io
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SURVEY = HERE.parent
REPO = HERE.parents[3]
sys.path.insert(0, str(SURVEY.parent))
sys.path.insert(0, str(SURVEY.parent / "gate2"))
sys.path.insert(0, str(SURVEY))

from common import OUT  # noqa: E402

# the public repo is read from `git remote get-url origin` at build time (git mode only)
PYTHON = "3.13.2"  # the gate-2 interpreter
UV = "0.11.8"
# the gate-2 environment (worktree venv used for every gate-2 search), search-path packages only
PINS = ["numpy==2.5.4", "numba==0.68.1", "llvmlite==0.50.0", "scipy==1.18.1", "astropy==8.0.2",
        "astropy-iers-data==0.2026.11.9.0.35.9", "pyerfa==2.0.1.5", "astroquery==0.4.11",
        "wotan==2.1", "pandas==2.3.3", "requests==2.34.2", "urllib3==2.8.0",
        "certifi==2026.10.23", "pyvo==1.10.0"]
EMBED_PATHS = ["spikes/mdwarf_stack", "src/oddball"]


def git(*a) -> bytes:
    return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, check=True).stdout


def kaggle_user() -> str:
    out = subprocess.run(["kaggle", "config", "view"], capture_output=True, text=True,
                         check=True).stdout
    for line in out.splitlines():
        if "username" in line:
            return line.split(":", 1)[1].strip()
    raise SystemExit("no Kaggle username in `kaggle config view`")


def b64gz(data: bytes) -> str:
    return base64.b64encode(gzip.compress(data, 9)).decode()


def main() -> None:
    import pandas as pd
    from run_survey import DEFAULT_TARGETS, shard_dir

    ap = argparse.ArgumentParser()
    ap.add_argument("--of", type=int, required=True)
    ap.add_argument("--shards", default=None, help="comma list (default: all)")
    ap.add_argument("--tag", default="survey")
    ap.add_argument("--targets", default=str(DEFAULT_TARGETS))
    ap.add_argument("--min-sectors", type=int, default=6)
    ap.add_argument("--mode", choices=("git", "embed"), default="git")
    ap.add_argument("--commit", default=None, help="default: HEAD of this checkout")
    ap.add_argument("--budget-hours", type=float, default=11.0,
                    help="wall clock for setup + run; the runner's guard gets the rest")
    ap.add_argument("--grace-min", type=float, default=20.0)
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--route", default="s3")
    ap.add_argument("--slug", default="transit-survey", help="kernel slug prefix")
    ap.add_argument("--skip-from-local", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    commit = a.commit or git("rev-parse", "HEAD").decode().strip()
    repo_url = None
    if a.mode == "git":
        repo_url = git("remote", "get-url", "origin").decode().strip()
        remote = git("ls-remote", "origin").decode()
        if commit not in remote and git("branch", "-r", "--contains", commit).strip() == b"":
            print(f"WARNING: {commit[:12]} is not on origin yet; push it before the kernels run")
    t = pd.read_csv(a.targets)
    t = t[t["n_sectors"] >= a.min_sectors]
    buf = io.StringIO()
    t.to_csv(buf, index=False)
    targets = b64gz(buf.getvalue().encode())
    code = ""
    if a.mode == "embed":
        code = base64.b64encode(git("archive", "--format=tar.gz", commit, *EMBED_PATHS)).decode()
    tmpl = (HERE / "kernel_template.py").read_text()
    meta_t = json.loads((HERE / "kernel-metadata.template.json").read_text())
    user = kaggle_user()
    out = Path(a.out) if a.out else OUT / a.tag / "kaggle_kernels"
    shards = [int(x) for x in a.shards.split(",")] if a.shards else range(a.of)
    for i in shards:
        cfg = {"shard": i, "of": a.of, "tag": a.tag, "min_sectors": a.min_sectors,
               "procs": a.procs, "route": a.route, "budget_hours": a.budget_hours,
               "grace_min": a.grace_min, "commit": commit, "repo": repo_url, "pins": PINS,
               "python": PYTHON, "uv": UV, "mode": a.mode}
        skip = ""
        if a.skip_from_local:
            rd = shard_dir(OUT / a.tag, i, a.of) / "results"
            done = sorted(p.stem[3:] for p in rd.glob("tic*.json")) if rd.exists() else []
            skip = b64gz("\n".join(done).encode()) if done else ""
            print(f"shard {i}: skipping {len(done)} stars finished locally")
        slug = f"{a.slug}-{i:03d}of{a.of:03d}" if a.of > 1 else a.slug
        d = out / slug
        d.mkdir(parents=True, exist_ok=True)
        src = (tmpl.replace("@@CFG@@", base64.b64encode(json.dumps(cfg).encode()).decode())
               .replace("@@TARGETS@@", targets).replace("@@SKIP@@", skip)
               .replace("@@CODE@@", code))
        (d / "survey_shard.py").write_text(src)
        meta = dict(meta_t, id=f"{user}/{slug}", title=slug.replace("-", " "),
                    code_file="survey_shard.py")
        (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
        print(f"{d}  ({len(src) / 1e3:.0f} kB, commit {commit[:12]}, mode {a.mode})")


if __name__ == "__main__":
    main()
