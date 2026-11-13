"""Download small image cutouts (DSS1/DSS2, 2MASS, PS1) and a Gaia DR3 field list around a
target. Usage: s6_download.py RA DEC OUTDIR [size_arcmin]"""
import json
import sys
import time
from pathlib import Path

import requests

UA = {"User-Agent": "astro-research-client/1.0"}
ra, dec, out = float(sys.argv[1]), float(sys.argv[2]), Path(sys.argv[3])
size = float(sys.argv[4]) if len(sys.argv) > 4 else 3.0
out.mkdir(parents=True, exist_ok=True)
log = {}


def get(url, params, fn, timeout=120):
    try:
        r = requests.get(url, params=params, headers=UA, timeout=timeout)
        r.raise_for_status()
        c = r.content
        if c[:2] == b"\x1f\x8b":
            import gzip
            c = gzip.decompress(c)
        if len(c) < 3000 or not c[:6] == b"SIMPLE":
            log[fn] = f"bad content ({len(c)} B) {r.url}: {c[:200]!r}"
            return False
        (out / fn).write_bytes(c)
        log[fn] = f"ok {len(c)} B {r.url}"
        return True
    except Exception as ex:  # noqa: BLE001
        log[fn] = f"failed {url} {ex!r}"[:300]
        return False
    finally:
        time.sleep(1.0)


for sv in ["poss1_red", "poss1_blue", "poss2ukstu_red", "poss2ukstu_blue", "poss2ukstu_ir"]:
    get("https://archive.stsci.edu/cgi-bin/dss_search",
        {"v": sv, "r": f"{ra:.6f}", "d": f"{dec:.6f}", "e": "J2000", "h": size, "w": size,
         "f": "fits", "c": "none"}, f"dss_{sv}.fits")

try:
    r = requests.get("https://ps1images.stsci.edu/cgi-bin/ps1filenames.py",
                     params={"ra": ra, "dec": dec, "filters": "grizy"}, headers=UA, timeout=60)
    r.raise_for_status()
    lines = [ln.split() for ln in r.text.strip().splitlines()]
    hdr = lines[0]
    for row in lines[1:]:
        d = dict(zip(hdr, row))
        px = int(size * 60 / 0.25)
        get("https://ps1images.stsci.edu/cgi-bin/fitscut.cgi",
            {"ra": ra, "dec": dec, "size": px, "format": "fits", "red": d["filename"]},
            f"ps1_{d['filter']}.fits")
except Exception as ex:  # noqa: BLE001
    log["ps1_list"] = f"failed {ex!r}"

try:
    r = requests.get("https://irsa.ipac.caltech.edu/cgi-bin/2MASS/IM/nph-im_sia",
                     params={"POS": f"{ra},{dec}", "SIZE": 0.01, "FORMAT": "image/fits"},
                     headers=UA, timeout=120)
    r.raise_for_status()
    (out / "2mass_sia.xml").write_bytes(r.content)
    from astropy.io.votable import parse
    t = parse(str(out / "2mass_sia.xml")).get_first_table().to_table()
    log["2mass_sia_cols"] = list(t.colnames)
    log["2mass_sia_n"] = len(t)
    done = set()
    for row in t:
        band = str(row["band"]).strip() if "band" in t.colnames else "?"
        url = str(row["download"]) if "download" in t.colnames else None
        if band in done or url is None:
            continue
        try:
            rr = requests.get(url, headers=UA, timeout=180)
            rr.raise_for_status()
            c = rr.content
            if c[:2] == b"\x1f\x8b":
                import gzip
                c = gzip.decompress(c)
            fn = f"2mass_{band}.fits"
            (out / fn).write_bytes(c)
            log[fn] = f"ok {len(c)} B {url}"
            done.add(band)
        except Exception as ex:  # noqa: BLE001
            log[f"2mass_{band}"] = f"failed {url} {ex!r}"[:300]
        time.sleep(1.0)
except Exception as ex:  # noqa: BLE001
    log["2mass_sia"] = f"failed {ex!r}"[:300]

from astroquery.vizier import Vizier  # noqa: E402
import astropy.units as u  # noqa: E402
from astropy.coordinates import SkyCoord  # noqa: E402

for server in ("vizier.idia.ac.za", "vizier.cfa.harvard.edu"):
    try:
        v = Vizier(columns=["Source", "RA_ICRS", "DE_ICRS", "pmRA", "pmDE", "Gmag", "BP-RP",
                            "Plx"], row_limit=-1, vizier_server=server, timeout=120)
        res = v.query_region(SkyCoord(ra * u.deg, dec * u.deg), radius=size * 0.75 * u.arcmin,
                             catalog="I/355/gaiadr3")
        tab = res[0]
        tab = tab[(tab["Gmag"] < 30)]
        tab.write(out / "gaia_field.csv", format="csv", overwrite=True)
        log["gaia_field"] = f"ok {server} {len(tab)} rows"
        break
    except Exception as ex:  # noqa: BLE001
        log[f"gaia_field_{server}"] = f"failed {ex!r}"[:300]
    time.sleep(1.0)

(out / "download_log.json").write_text(json.dumps(log, indent=1))
print(json.dumps(log, indent=1))
