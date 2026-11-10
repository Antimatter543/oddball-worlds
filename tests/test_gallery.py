"""Gallery labelling: label validation, the append-only CSV, selection, and the HTTP server."""

from __future__ import annotations

import csv
import http.client
import json
import threading

import numpy as np
import pandas as pd
import pytest

from oddball.gallery.labels import (
    COLUMNS,
    NOTE_MAX,
    LabelError,
    LabelStore,
    clean_note,
    make_row,
    validate_label,
    validate_tic,
)
from oddball.gallery.select import select
from oddball.gallery.xmatch import verdict


def _row(**kw):
    base = {"tic": 12345, "sector": 32, "scorer": "ssl_v2_iforest", "rank": 3, "split": "train",
            "reason": "top: rank 3", "label": "interesting", "note": "", "gallery": "g1"}  # fmt: skip
    base.update(kw)
    return make_row(**base)


# ---------------------------------------------------------------- validation


@pytest.mark.parametrize("lab", ["interesting", "systematic", "known-boring-variable", "unsure"])
def test_valid_labels(lab):
    assert validate_label(lab) == lab


@pytest.mark.parametrize("lab", ["", "Interesting", "boring", None, 1, ["unsure"], "unsure "])
def test_invalid_labels(lab):
    with pytest.raises(LabelError):
        validate_label(lab)


@pytest.mark.parametrize("tic,ok", [(12345, 12345), ("12345", 12345), (0, None), (-5, None),
                                    (True, None), (1.0, None), ("12a", None), ("1e5", None),
                                    ("", None), (None, None), ("٣٣", None), (10**12, None)])  # fmt: skip
def test_tic_validation(tic, ok):
    if ok is None:
        with pytest.raises(LabelError):
            validate_tic(tic)
    else:
        assert validate_tic(tic) == ok


def test_note_cleaning():
    assert clean_note(None) == ""
    assert clean_note("  two\nlines\r\n\tand tab ") == "two lines and tab"
    assert clean_note("a\x00b c") == "a b c"
    assert clean_note("=HYPERLINK(1)") == "'=HYPERLINK(1)"
    assert clean_note("-3 sigma dip") == "'-3 sigma dip"
    assert clean_note('quote " and, comma') == 'quote " and, comma'
    assert len(clean_note("x" * NOTE_MAX)) == NOTE_MAX
    with pytest.raises(LabelError):
        clean_note("x" * (NOTE_MAX + 1))
    with pytest.raises(LabelError):
        clean_note(123)


def test_make_row_rejects_bad_fields():
    with pytest.raises(LabelError):
        _row(label="weird")
    with pytest.raises(LabelError):
        _row(scorer="../etc")
    with pytest.raises(LabelError):
        _row(gallery="a b")
    with pytest.raises(LabelError):
        _row(rank=-1)
    with pytest.raises(LabelError):
        _row(sector=0)


# ---------------------------------------------------------------- the CSV


def test_append_and_latest(tmp_path):
    st = LabelStore(tmp_path / "labels" / "l.csv")
    assert st.rows() == [] and st.latest() == {}
    st.append(_row(label="unsure", note='has "quotes", commas\nand newline'))
    st.append(_row(tic=999, label="systematic"))
    st.append(_row(label="interesting", note="changed my mind"))
    with (tmp_path / "labels" / "l.csv").open(newline="") as fh:
        recs = list(csv.reader(fh))
    assert tuple(recs[0]) == COLUMNS and len(recs) == 4  # header written once, append-only
    assert recs[1][COLUMNS.index("note")] == 'has "quotes", commas and newline'
    lat = st.latest()
    assert lat[(12345, 32)].label == "interesting" and lat[(12345, 32)].note == "changed my mind"
    assert lat[(999, 32)].label == "systematic"
    assert len(st.rows()) == 3


def test_survives_truncated_last_line(tmp_path):
    p = tmp_path / "l.csv"
    st = LabelStore(p)
    st.append(_row())
    with p.open("a") as fh:  # simulate a crash mid-write
        fh.write("777,32,ssl_v2,1,train,top,interes")
    assert [r.tic for r in st.rows()] == [12345]  # the partial row is skipped
    st.append(_row(tic=888, label="unsure"))  # and the next write starts a fresh line
    assert [r.tic for r in st.rows()] == [12345, 888]


def test_garbage_rows_skipped(tmp_path):
    p = tmp_path / "l.csv"
    p.write_text(",".join(COLUMNS) + "\n" + "abc,32,s,1,train,r,interesting,,t,g\n"
                 + "5,32,s,1,train,r,nonsense,,t,g\n" + "6,32,s,1,train,r,unsure,,t,g\n")
    assert [r.tic for r in LabelStore(p).rows()] == [6]


def test_concurrent_appends(tmp_path):
    st = LabelStore(tmp_path / "l.csv")

    def work(k):
        for j in range(25):
            st.append(_row(tic=1000 * k + j + 1, note="x" * 300))

    ts = [threading.Thread(target=work, args=(k,)) for k in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(st.rows()) == 100


# ---------------------------------------------------------------- selection


def _fake_scores(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"tic": np.arange(1, n + 1) * 7919 + 13, "sector": 32,
                         "score": rng.random(n), "score_a": rng.random(n),
                         "score_b": rng.random(n)})  # fmt: skip


def test_select_mix_and_reasons():
    d = _fake_scores()
    s = select(d, 60, 0.3, seed=1)
    assert len(s) == 60 and s.tic.is_unique
    c = s.pick.value_counts().to_dict()
    assert c == {"top": 42, "explore-top5": 9, "explore-mid": 9}
    assert (s.split != "known").all() and s.reason.str.len().gt(10).all()
    top = s[s.pick == "top"]
    assert list(top["rank"]) == list(range(1, 43))
    assert (s[s.pick == "explore-top5"].rank_pct <= 5).all()
    mid = s[s.pick == "explore-mid"]
    assert ((mid.rank_pct > 5) & (mid.rank_pct <= 25)).all()
    # deterministic for a seed
    assert list(select(d, 60, 0.3, seed=1).tic) == list(s.tic)


def test_select_excludes_labelled_and_known():
    from oddball.common.splits import known_positive_tics

    d = _fake_scores()
    kp = sorted(known_positive_tics())[:3]
    d.loc[:2, "tic"] = kp
    d.loc[:2, "score"] = 2.0  # known positives at the very top: must never be shown
    s0 = select(d, 10, 0.0)
    assert not set(kp) & set(s0.tic)
    excl = {(int(t), 32) for t in s0.tic[:5]}
    s1 = select(d, 10, 0.0, exclude=excl)
    assert not {t for t, _ in excl} & set(s1.tic)


# ---------------------------------------------------------------- catalogue verdict


def test_verdict_kinds():
    common = {"vsx": {"ok": True, "rows": [{"Name": "V1", "Type": "EA", "Period": 2.0,
                                            "sep_arcsec": 0.3}]},
              "simbad": {"ok": True, "by_tic_id": None, "nearby": []}}  # fmt: skip
    assert verdict(common)["kind"] == "common"
    young = {"vsx": {"ok": True, "rows": [{"Name": "V2", "Type": "YSO", "Period": None,
                                           "sep_arcsec": 12.0}]}, "simbad": {}}  # fmt: skip
    assert verdict(young)["kind"] == "young"
    toi = {"toi": {"toi": [{"TOI": "1234.01", "TFOPWG Disposition": "PC"}], "ctoi": []}}
    v = verdict(toi)
    assert v["kind"] == "toi" and "1234.01" in v["text"]
    assert verdict({})["kind"] == "none"


# ---------------------------------------------------------------- the server


@pytest.fixture()
def server(tmp_path):
    from oddball.gallery.server import make_server

    g = tmp_path / "gallery" / "g1"
    (g / "cards").mkdir(parents=True)
    (g / "cards" / "12345_s0032.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
    (tmp_path / "secret.txt").write_text("secret")
    card = {"tic": 12345, "sector": 32, "png": "cards/12345_s0032.png", "rank": 3,
            "rank_pct": 0.1, "score": 0.99, "pick": "top", "reason": "top: rank 3",
            "split": "train", "tmag": 10.0, "camera": 1, "ccd": 2, "ra": 10.0, "dec": -20.0,
            "rms_ppt": 1.0, "subscores": {}, "neighbours": [], "other_sectors": [],
            "window": {"t0": 1, "t1": 2, "source": "<script>alert(1)</script>"},
            "xmatch": {"verdict": {"kind": "none", "text": "<b>x</b>"},
                       "simbad": {"ok": True, "by_tic_id": {"main_id": "<img src=x>",
                                                            "otype": "*"}, "nearby": []}}}  # fmt: skip
    m = {"gallery": "g1", "scorer": "ssl_v2_iforest", "sector": 32, "cards": [card]}
    (g / "manifest.json").write_text(json.dumps(m))
    labels = tmp_path / "labels.csv"
    srv = make_server(tmp_path / "gallery", labels, port=0, token="tok")
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield srv, labels
    srv.shutdown()
    srv.server_close()


def _req(srv, method, path, body=None, headers=None):
    port = srv.server_address[1]
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    h = {"Host": f"127.0.0.1:{port}"}
    h.update(headers or {})
    data = body if isinstance(body, bytes | type(None)) else json.dumps(body).encode()
    c.request(method, path, body=data, headers=h)
    r = c.getresponse()
    out = r.status, r.read()
    c.close()
    return out


def _post(srv, body, **hdr):
    h = {"Content-Type": "application/json", "X-Label-Token": "tok"}
    h.update(hdr)
    return _req(srv, "POST", "/api/label", body, h)


def test_server_binds_localhost(server):
    srv, _ = server
    assert srv.server_address[0] == "127.0.0.1"


def test_server_pages_escape_catalogue_text(server):
    srv, _ = server
    st, body = _req(srv, "GET", "/g/g1/")
    assert st == 200
    assert b"<script>alert(1)" not in body and b"&lt;script&gt;alert(1)" in body
    assert b"<img src=x>" not in body
    st, body = _req(srv, "GET", "/g/g1/cards/12345_s0032.png")
    assert st == 200 and body.startswith(b"\x89PNG")


@pytest.mark.parametrize("path", ["/g/g1/../secret.txt", "/g/../secret.txt",
                                  "/g/g1/cards/../manifest.json", "/g/g1/manifest.json",
                                  "/g/g1/cards/99_s0032.png", "/g/nope/", "/etc/passwd"])  # fmt: skip
def test_server_no_traversal(server, path):
    srv, _ = server
    st, body = _req(srv, "GET", path)
    assert st == 404 and b"secret" not in body


def test_server_label_roundtrip(server):
    srv, labels = server
    st, body = _post(srv, {"gallery": "g1", "tic": 12345, "label": "systematic",
                           "note": "=1+1, shared dip"})  # fmt: skip
    assert st == 200, body
    rows = LabelStore(labels).rows()
    assert len(rows) == 1
    r = rows[0]
    assert (r.tic, r.sector, r.scorer, r.rank, r.split, r.label) == (
        12345, 32, "ssl_v2_iforest", 3, "train", "systematic")
    assert r.note == "'=1+1, shared dip" and r.reason == "top: rank 3"
    st, body = _req(srv, "GET", "/g/g1/state")
    assert json.loads(body)["12345"]["label"] == "systematic"


@pytest.mark.parametrize("body,hdr,code", [
    ({"gallery": "g1", "tic": 12345, "label": "boring"}, {}, 400),
    ({"gallery": "g1", "tic": "12345; rm", "label": "unsure"}, {}, 400),
    ({"gallery": "g1", "tic": 54321, "label": "unsure"}, {}, 400),  # not a card of g1
    ({"gallery": "../g1", "tic": 12345, "label": "unsure"}, {}, 400),
    ({"gallery": "g1", "tic": 12345, "label": "unsure", "note": "x" * 600}, {}, 400),
    ({"gallery": "g1", "tic": 12345, "label": "unsure"}, {"X-Label-Token": "wrong"}, 403),
    ({"gallery": "g1", "tic": 12345, "label": "unsure"}, {"Content-Type": "text/plain"}, 415),
    ({"gallery": "g1", "tic": 12345, "label": "unsure"}, {"Origin": "https://evil.example"}, 403),
    ({"gallery": "g1", "tic": 12345, "label": "unsure"}, {"Host": "evil.example:80"}, 403),
    (b"not json", {}, 400),
    (b"[1, 2]", {}, 400),
    (b"{" + b" " * 5000 + b"}", {}, 413),
])  # fmt: skip
def test_server_rejects_bad_posts(server, body, hdr, code):
    srv, labels = server
    st, _ = _post(srv, body, **hdr)
    assert st == code
    assert LabelStore(labels).rows() == []
