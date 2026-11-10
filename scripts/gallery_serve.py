"""Serve pre-rendered galleries for labelling, on 127.0.0.1 only (stdlib HTTP server).

    uv run python -I scripts/gallery_serve.py                      # every gallery
    uv run python -I scripts/gallery_serve.py --gallery ssl_v2_iforest_s0032_n60_seed0

Then open http://127.0.0.1:8765/ . Labels are appended to ``labels/labels_v1.csv`` (commit it).
Build galleries with ``scripts/gallery_build.py``. Ctrl-C stops the server; nothing is lost,
every label is written to disk (fsync) when you press the key.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from oddball.common.paths import outputs_root
from oddball.gallery.labels import DEFAULT_LABELS_CSV, validate_name
from oddball.gallery.server import HOST, make_server


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gallery", default=None, help="serve only this gallery (default: all)")
    ap.add_argument("--root", type=Path, default=outputs_root() / "gallery")
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS_CSV)
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    if a.gallery:
        validate_name(a.gallery, "gallery")
        if not (a.root / a.gallery / "manifest.json").exists():
            raise SystemExit(f"no gallery {a.root / a.gallery}; build it with gallery_build.py")
    srv = make_server(a.root, a.labels, a.port, a.gallery)
    port = srv.server_address[1]
    url = f"http://{HOST}:{port}/" + (f"g/{a.gallery}/" if a.gallery else "")
    print(f"serving {a.root} on {url}  (labels -> {a.labels}); Ctrl-C to stop", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
