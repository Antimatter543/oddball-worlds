"""Block until no python process is running any of the named scripts (argv[0] must be a python binary, so
shells whose command text merely mentions the script do not count). Usage: python -I cool/waitfor.py deep.py ..."""
import os
import sys
import time

names = sys.argv[1:]
me = os.getpid()


def busy():
    for p in os.listdir("/proc"):
        if not p.isdigit() or int(p) == me:
            continue
        try:
            argv = open(f"/proc/{p}/cmdline", "rb").read().split(b"\0")
        except OSError:
            continue
        if not argv or b"python" not in os.path.basename(argv[0]):
            continue
        if any(a.decode(errors="ignore").endswith(n) for a in argv[1:4] for n in names):
            return True
    return False


while busy():
    time.sleep(10)
