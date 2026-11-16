"""Run cool/pixev.py for each line of a jobs file ("TIC t1 t2 ... width_h", '#' comments), one at a time.
Usage: python -I cool/pixjobs.py JOBS.txt"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for line in Path(sys.argv[1]).read_text().splitlines():
    p = line.split("#")[0].split()
    if not p:
        continue
    cmd = [sys.executable, "-I", str(HERE / "pixev.py"), p[0], "--times", *p[1:-1], "--width-h", p[-1]]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE.parent)
    out = [x for x in (r.stdout + r.stderr).splitlines() if "arn" not in x]
    print("\n".join(out[-8:]), flush=True)
