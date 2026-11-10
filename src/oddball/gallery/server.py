"""Tiny local labelling server for pre-rendered galleries (stdlib only, 127.0.0.1 only).

Routes:

* ``GET /``                         list of galleries under the root
* ``GET /g/<gallery>/``             the labelling page (server-rendered, inline CSS/JS)
* ``GET /g/<gallery>/cards/<png>``  a card image (only files listed in the manifest)
* ``GET /g/<gallery>/state``        current labels for the gallery's stars (JSON)
* ``POST /api/label``               ``{"gallery", "tic", "label", "note"}`` -> appends a row

Hardening (it is a local tool, but a browser will happily let any web page POST to
localhost): the server binds 127.0.0.1 only; the Host header must be ``127.0.0.1:<port>`` or
``localhost:<port>`` (DNS rebinding); a POST needs ``Content-Type: application/json`` (no
CORS headers are ever sent, so cross-origin JSON is blocked at preflight), the per-process
random token in ``X-Label-Token``, and a matching ``Origin`` if one is sent; bodies are capped
at ``MAX_BODY`` bytes; every field is validated (``oddball.gallery.labels``) and the star must
be a card of that gallery, whose rank/split/reason/scorer are taken from the manifest, never
from the request. All catalogue text is HTML-escaped (it comes from external services).
"""

from __future__ import annotations

import hmac
import html
import json
import re
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote

from oddball.gallery.labels import (
    LABELS,
    LabelError,
    LabelStore,
    make_row,
    validate_name,
    validate_tic,
)

HOST = "127.0.0.1"
MAX_BODY = 4096
_PNG_RE = re.compile(r"^cards/[0-9]+_s[0-9]{4}\.png$")
KEYS = {"interesting": "i", "systematic": "s", "known-boring-variable": "b", "unsure": "u"}


class Galleries:
    """Manifests under ``root``, re-read when a gallery is rebuilt (mtime change)."""

    def __init__(self, root: Path, only: str | None = None):
        self.root = Path(root)
        self.only = only
        self._cache: dict[str, tuple[float, dict]] = {}
        self._lock = threading.Lock()

    def names(self) -> list[str]:
        if self.only:
            return [self.only] if (self.root / self.only / "manifest.json").exists() else []
        return sorted(p.parent.name for p in self.root.glob("*/manifest.json"))

    def get(self, name: str) -> dict | None:
        try:
            validate_name(name, "gallery")
        except LabelError:
            return None
        if name not in self.names():
            return None
        f = self.root / name / "manifest.json"
        mt = f.stat().st_mtime
        with self._lock:
            hit = self._cache.get(name)
            if hit is None or hit[0] != mt:
                m = json.loads(f.read_text())
                m["_by_tic"] = {int(c["tic"]): c for c in m["cards"]}
                m["_pngs"] = {c["png"] for c in m["cards"]}
                self._cache[name] = (mt, m)
                hit = self._cache[name]
        return hit[1]


def _e(v) -> str:
    return html.escape("" if v is None else str(v), quote=True)


def _fmt(v, nd=3) -> str:
    if v is None:
        return "–"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


GUIDE = """
<details open id="guide"><summary><b>How to label (read once)</b></summary>
<p>Each card is one star watched by NASA's TESS telescope for about 27 days (one
<i>sector</i>). The computer ranked these stars as unusual; your job is to say <i>what kind</i>
of unusual. <b>Black dots</b> are the star's brightness after NASA's pipeline removed known
spacecraft trends (PDCSAP). <b>Orange dots</b> are the less-processed brightness (SAP), drawn
below. A dip in both is more trustworthy (but PDCSAP sometimes flattens a real, long dip, so
also look at orange). The red fill shows <i>where</i> the computer found the oddity, and the
middle row zooms in there.</p>
<ul>
<li><b>interesting</b> (<kbd>i</kbd>/<kbd>1</kbd>): the star itself seems to do something odd
that the checks below don't explain: irregular or lopsided dips, a single deep dip, dips that
change depth or shape, a sudden brightening.</li>
<li><b>systematic</b> (<kbd>s</kbd>/<kbd>2</kbd>): the telescope, not the star. Tell-tales:
the same dip or ramp shows up in the <b>neighbour stars</b> (right-hand panel, other stars on
the same camera chip); it sits at the <b>start or end of an orbit</b> (purple dotted lines,
next to a data gap) or inside an <b>orange band</b> (scattered sunlight / earthshine); it lines
up with a <b>red line</b> (a momentum dump: the spacecraft fires thrusters, which jolts the
pointing); it is a sudden step or jump.</li>
<li><b>known-boring-variable</b> (<kbd>b</kbd>/<kbd>3</kbd>): a normal, well-understood
variable star: dips that repeat like clockwork (an eclipsing binary), or a smooth regular wave
(starspots rotating, or the star pulsating). The <i>catalogue</i> line says <i>known common
variable</i> when a catalogue already lists it as one, but trust the shape too.</li>
<li><b>unsure</b> (<kbd>u</kbd>/<kbd>4</kbd>): can't tell. That is useful information too.</li>
</ul>
<p>Watch for: <b>TOI/CTOI</b> = already a TESS planet candidate. <b>young star / irregular
dimmer</b> = a catalogued member of exactly the class we hunt: look closely, it may be a
real dipper. A neighbour under 2 pixels away shares the same pixels, so a dip shared with it
may be a <i>blend</i> (light from that neighbour), not the instrument: note it.</p>
<p>Keys: <kbd>i</kbd> <kbd>s</kbd> <kbd>b</kbd> <kbd>u</kbd> (or <kbd>1</kbd>-<kbd>4</kbd>)
label the highlighted card and move on; <kbd>j</kbd>/<kbd>k</kbd> (or arrows) next/previous;
<kbd>n</kbd> write a note (<kbd>Enter</kbd> saves it with the card's label, <kbd>Esc</kbd>
leaves the box); <kbd>h</kbd> hide labelled cards; <kbd>g</kbd> show/hide this guide. A new
label replaces the old one (both are kept in the file; the latest wins).</p>
</details>
"""

CSS = """
:root{--bg:#f6f6f3;--fg:#1d1d1b;--mut:#666;--card:#fff;--hi:#1f6feb;--line:#ddd}
body{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:0 auto;
max-width:1220px;padding:12px 16px 80px}
h1{font-size:20px;margin:6px 0} kbd{border:1px solid #bbb;border-radius:3px;padding:0 4px;
font-size:12px;background:#fff} details#guide{background:#fff;border:1px solid var(--line);
padding:8px 14px;border-radius:6px} .bar{position:sticky;top:0;background:var(--bg);
padding:6px 0;z-index:5;border-bottom:1px solid var(--line);display:flex;gap:16px;
align-items:center;flex-wrap:wrap}
.card{background:var(--card);border:2px solid var(--line);border-radius:8px;margin:18px 0;
padding:10px 12px;scroll-margin-top:50px} .card.cur{border-color:var(--hi);
box-shadow:0 0 0 3px #1f6feb33} .card img{width:100%;height:auto;display:block}
.head{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}
.badge{font-size:12px;padding:1px 7px;border-radius:10px;background:#eee}
.lab-interesting{background:#2da44e;color:#fff} .lab-systematic{background:#cf222e;color:#fff}
.lab-known-boring-variable{background:#8c959f;color:#fff} .lab-unsure{background:#bf8700;
color:#fff} .v-toi{background:#8250df;color:#fff} .v-young{background:#fb8500;color:#fff}
.v-common{background:#8c959f;color:#fff} .v-variable{background:#d0d7de}
.ctx{display:grid;grid-template-columns:1fr 1fr;gap:4px 24px;font-size:13px;margin-top:6px}
.ctx div{overflow-wrap:anywhere} .mut{color:var(--mut)} .ctl{display:flex;gap:8px;
margin-top:8px;flex-wrap:wrap;align-items:center} .ctl button{padding:4px 10px;
cursor:pointer} .ctl input{flex:1;min-width:240px;padding:4px} .hide-done .card.done{display:none}
@media (max-width:760px){.ctx{grid-template-columns:1fr}}
"""

JS = """
const TOKEN=document.querySelector('meta[name=label-token]').content;
const GAL=document.body.dataset.gallery;
const cards=[...document.querySelectorAll('.card')];
const KEYMAP={i:'interesting','1':'interesting',s:'systematic','2':'systematic',
 b:'known-boring-variable','3':'known-boring-variable',u:'unsure','4':'unsure'};
let cur=0;
function visible(){return cards.filter(c=>c.offsetParent!==null);}
function setCur(i,scroll=true){cards.forEach(c=>c.classList.remove('cur'));
 cur=Math.max(0,Math.min(i,cards.length-1));cards[cur].classList.add('cur');
 if(scroll)cards[cur].scrollIntoView({behavior:'smooth',block:'start'});}
function step(d){const v=visible();if(!v.length)return;let j=v.indexOf(cards[cur]);
 if(j<0){j=d>0?-1:v.length;} j=Math.max(0,Math.min(j+d,v.length-1));setCur(cards.indexOf(v[j]));}
function nextUnlabelled(){for(let k=1;k<=cards.length;k++){const c=cards[(cur+k)%cards.length];
 if(!c.classList.contains('done')){setCur(cards.indexOf(c));return;}} step(1);}
function progress(){const n=cards.filter(c=>c.classList.contains('done')).length;
 document.getElementById('prog').textContent=n+' / '+cards.length+' labelled';}
function show(c,label,note){const b=c.querySelector('.cur-label');b.className='badge cur-label lab-'+label;
 b.textContent=label;c.classList.add('done');if(note!==undefined)c.querySelector('input').value=note;progress();}
async function send(c,label,advance=true){const st=c.querySelector('.status');
 const note=c.querySelector('input').value;st.textContent='saving…';
 try{const r=await fetch('/api/label',{method:'POST',headers:{'Content-Type':'application/json',
  'X-Label-Token':TOKEN},body:JSON.stringify({gallery:GAL,tic:Number(c.dataset.tic),label:label,note:note})});
  const j=await r.json();if(!r.ok){st.textContent='error: '+(j.error||r.status);return;}
  st.textContent='saved '+j.utc;show(c,label,j.note);if(advance)nextUnlabelled();}
 catch(e){st.textContent='error: '+e;}}
async function load(){const r=await fetch('/g/'+encodeURIComponent(GAL)+'/state');
 const j=await r.json();for(const c of cards){const s=j[c.dataset.tic];if(s){show(c,s.label,s.note);
 c.querySelector('.status').textContent='labelled '+s.utc+(s.gallery!==GAL?' (in '+s.gallery+')':'');}}
 progress();const f=cards.findIndex(c=>!c.classList.contains('done'));setCur(f<0?0:f,false);}
document.addEventListener('keydown',e=>{const t=e.target;
 if(t.tagName==='INPUT'){if(e.key==='Escape'){t.blur();}
  else if(e.key==='Enter'){const c=t.closest('.card');const l=c.querySelector('.cur-label').textContent;
   if(KEYMAP[l]||Object.values(KEYMAP).includes(l)){send(c,l,false);} else {c.querySelector('.status').textContent='pick a label key (i/s/b/u) to save with this note';}
   t.blur();}
  return;}
 if(e.ctrlKey||e.metaKey||e.altKey)return;
 const k=e.key;
 if(KEYMAP[k]){e.preventDefault();send(cards[cur],KEYMAP[k]);}
 else if(k==='j'||k==='ArrowDown'||k==='ArrowRight'){e.preventDefault();step(1);}
 else if(k==='k'||k==='ArrowUp'||k==='ArrowLeft'){e.preventDefault();step(-1);}
 else if(k==='n'){e.preventDefault();cards[cur].querySelector('input').focus();}
 else if(k==='h'){document.body.classList.toggle('hide-done');}
 else if(k==='g'){const g=document.getElementById('guide');g.open=!g.open;}});
cards.forEach((c,i)=>{c.addEventListener('click',()=>setCur(i,false));
 c.querySelectorAll('button[data-label]').forEach(b=>b.addEventListener('click',ev=>{
  ev.stopPropagation();setCur(i,false);send(c,b.dataset.label);}));});
load();
"""


def _xmatch_html(c: dict, otypes: dict) -> str:
    xm = c.get("xmatch")
    if not xm:
        return "<div><b>catalogues:</b> <span class=mut>not queried (offline build)</span></div>"
    v = xm.get("verdict") or {}
    kind = _e(v.get("kind"))
    head = (f"<div style='grid-column:1/-1'><b>catalogues:</b> <span class='badge v-{kind}'>"
            f"{kind}</span> {_e(v.get('text'))}</div>")  # fmt: skip
    parts = [head]
    sb = xm.get("simbad") or {}
    if sb.get("ok"):
        me = sb.get("by_tic_id")
        near = sb.get("nearby", [])[:4]
        txt = (f"{_e(me['main_id'])} [{_e(me['otype'])}: {_e(otypes.get(me['otype'], ''))}]"
               if me else "no SIMBAD object carries this TIC id")  # fmt: skip
        nt = "; ".join(f"{_e(r['main_id'])} [{_e(r['otype'])}] {r['sep_arcsec']:.1f}\""
                       for r in near)  # fmt: skip
        parts.append(f"<div><b>SIMBAD:</b> {txt}<br><span class=mut>within 21\": "
                     f"{nt or 'nothing'}</span></div>")  # fmt: skip
    else:
        parts.append(f"<div><b>SIMBAD:</b> <span class=mut>{_e(sb.get('error', 'n/a'))}"
                     "</span></div>")  # fmt: skip
    vs = xm.get("vsx") or {}
    rows = vs.get("rows", [])[:4]
    vt = "; ".join(f"{_e(r['Name'])} [{_e(r['Type'])}"
                   + (f", P={r['Period']:.4g} d" if r.get("Period") else "")
                   + f"] {r['sep_arcsec']:.1f}\"" for r in rows)  # fmt: skip
    vt = vt or ("none within 21&quot;" if vs.get("ok") else "n/a")
    parts.append(f"<div><b>VSX:</b> {vt}</div>")
    g = (xm.get("gaia") or {}).get("rows", [])
    gt = "; ".join(f"{_e(r['class'])} (score {r['score']:.2f}, {r['sep_arcsec']:.1f}\")"
                   for r in g[:2])  # fmt: skip
    gt = gt or "none within 3&quot;"
    parts.append(f"<div><b>Gaia DR3 variability class:</b> {gt}</div>")
    t = xm.get("toi")
    if t is None:
        tt = "<span class=mut>TOI list unavailable</span>"
    else:
        tl = [f"TOI {_e(r.get('TOI'))} ({_e(r.get('TFOPWG Disposition') or r.get('TESS Disposition'))})"
              for r in t.get("toi", [])]  # fmt: skip
        tl += [f"CTOI {_e(r.get('CTOI'))}" for r in t.get("ctoi", [])]
        tt = "; ".join(tl) or "not a TOI or CTOI"
    parts.append(f"<div><b>TOI/CTOI:</b> {tt}</div>")
    return "".join(parts)


def _candidate_html(cd: dict | None) -> str:
    """Per-sector ranks of a cross-sector candidate card (``gallery_build.py --candidates``)."""
    if not cd:
        return ""
    rows = []
    for q in cd.get("per_sector", []):
        tag = "<b>candidate</b>" if q.get("candidate") else "not a candidate"
        flag = ", cmv3-flagged" if q.get("cmv3_flag") else ""
        rows.append(f"S{int(q['sector'])}: {tag}, classical #{int(q['cls_rank'])} / ens_v3 "
                    f"#{int(q['ens_rank'])} of {int(q['n_stars'])}, eb_like "
                    f"{_fmt(q.get('eb_like'), 2)}{flag}, strongest residual event "
                    f"{_e(q.get('res_event'))} at BTJD {_fmt(q.get('res_t'), 2)}")  # fmt: skip
    extra = []
    if cd.get("cat_class"):
        extra.append(f"catalogue class {_e(cd['cat_class'])}")
    if cd.get("kostov_nn"):
        extra.append("in Kostov+2025's unvetted NN-flagged list (soft; also fires on dippers)")
    if cd.get("triage_s32"):
        extra.append(f"S32 triage label {_e(cd['triage_s32'])}")
    extra.append("SkyBoT checked: no known asteroid" if cd.get("skybot_checked")
                 else "SkyBoT not checked")  # fmt: skip
    multi = "<b>multi-sector</b>" if cd.get("multi") else "single sector"
    return (f"<div><b>candidate ({_e(cd.get('list'))}, {multi}, from {_e(cd.get('source'))}):"
            "</b><br>" + "<br>".join(rows) + f"<br><span class=mut>{'; '.join(extra)}</span>"
            "</div>")


def _card_html(i: int, n: int, c: dict, m: dict) -> str:
    tic = int(c["tic"])
    subs = sorted(c.get("subscores", {}).items(), key=lambda kv: -(kv[1].get("pct") or 0))
    st = ", ".join(f"{_e(k[6:])} {100 * (v.get('pct') or 0):.1f}%" for k, v in subs[:8])
    sysd = c.get("sys") or {}
    sys_txt = (f"flag {_fmt(sysd.get('sys_flag'))}, shared-with-CCD share "
               f"{_fmt(sysd.get('cm_frac'), 2)}, at-dump share {_fmt(sysd.get('dump_frac'), 2)}"
               if sysd else "–")  # fmt: skip
    kc = c.get("knownclass")
    kc_txt = "–"
    if kc:
        top = ", ".join(f"{_e(k)} {p:.2f}" for k, p in kc.get("top", []))
        kc_txt = (f"looks like <b>{_e(kc['pred'])}</b> (p {kc['pmax']:.2f}; {top}); explained "
                  f"share {kc['explained']:.2f}"
                  + (f"; catalogue label {_e(kc['cat_label'])}" if kc.get("cat_label") else ""))
    cl = c.get("classical")
    cl_txt = f"; classical rank {cl['rank']} (score {cl['score']:.4f})" if cl else ""
    nbs = ", ".join(f"TIC {n_['tic']} ({n_['sep_arcsec'] / 21:.1f} px, T={n_['tmag']:.1f})"
                    for n_ in c.get("neighbours", []))  # fmt: skip
    simbad = (f"https://simbad.cfa.harvard.edu/simbad/sim-coo?Coord={c['ra']:.6f}%20"
              f"{c['dec']:+.6f}&Radius=21&Radius.unit=arcsec")  # fmt: skip
    exofop = f"https://exofop.ipac.caltech.edu/tess/target.php?id={tic}"
    buttons = "".join(f"<button data-label='{lab}'>{lab} <kbd>{KEYS[lab]}</kbd></button>"
                      for lab in LABELS)  # fmt: skip
    png = f"/g/{quote(m['gallery'])}/{quote(c['png'])}"
    cand_html = _candidate_html(c.get("candidate"))
    return f"""
<section class=card id=c{i} data-tic={tic}>
 <div class=head><b>{i + 1}/{n}</b><b>TIC {tic}</b>
  <span class=badge>{_e(c.get('pick'))}</span><span class=mut>{_e(c.get('reason'))}</span>
  <span class='badge cur-label'>unlabelled</span></div>
 <img loading=lazy src="{png}" alt="light curves of TIC {tic}">
 <div class=ctx>
  <div><b>star:</b> Tmag {c['tmag']:.2f}, cam {c['camera']}/CCD {c['ccd']}, Teff
   {_fmt(c.get('teff'), 0)} K, R {_fmt(c.get('radius'), 2)} Rsun, CROWDSAP
   {_fmt(c.get('crowdsap'), 2)} (1 = no contaminating light), scatter {c['rms_ppt']:.2f} ppt;
   other sectors: {', '.join('S' + str(s) for s in c.get('other_sectors', [])) or 'none'}</div>
  <div><b>score:</b> {_e(m['scorer'])} rank {c['rank']} (top {c['rank_pct']:.2f}%), score
   {c['score']:.5f}{cl_txt}<br><span class=mut>sub-score percentiles: {st or '–'}</span></div>
  {cand_html}
  <div><b>systematics check:</b> {sys_txt}</div>
  <div><b>known-class model:</b> {kc_txt}</div>
  <div><b>neighbours shown:</b> <span class=mut>{nbs or 'none'}</span></div>
  <div><b>window:</b> <span class=mut>{_e(c['window']['source'])}</span></div>
  {_xmatch_html(c, m.get('otypes', {}))}
  <div><a href="{_e(simbad)}" target=_blank rel=noopener>SIMBAD</a> ·
   <a href="{_e(exofop)}" target=_blank rel=noopener>ExoFOP</a></div>
 </div>
 <div class=ctl>{buttons}<input maxlength=500 placeholder="note (n to type, Enter saves)">
  <span class='status mut'></span></div>
</section>"""


def page_html(m: dict, token: str) -> str:
    cards = "".join(_card_html(i, len(m["cards"]), c, m) for i, c in enumerate(m["cards"]))
    sel = m.get("selection", {})
    return f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<meta name=label-token content="{_e(token)}"><title>Gallery {_e(m['gallery'])}</title>
<style>{CSS}</style></head><body data-gallery="{_e(m['gallery'])}">
<h1>Oddball gallery: {_e(m['scorer'])}, sector {m['sector']} ({len(m['cards'])} stars)</h1>
<div class=mut>built {_e(m.get('built_utc'))} (git {_e(m.get('git'))}); selection:
{_e(sel.get('counts'))}, {_e(sel.get('rule'))}; card order: {_e(sel.get('order'))}.
<a href="/">all galleries</a></div>
{GUIDE}
<div class=bar><b id=prog></b><span class=mut>i/s/b/u label · j/k move · n note · h hide
labelled · g guide</span></div>
{cards}
<script>{JS}</script></body></html>"""


def index_html(g: Galleries) -> str:
    items = []
    for name in g.names():
        m = g.get(name)
        if m:
            items.append(f"<li><a href='/g/{quote(name)}/'>{_e(name)}</a> <span class=mut>"
                         f"{_e(m['scorer'])}, S{m['sector']}, {len(m['cards'])} cards, built "
                         f"{_e(m.get('built_utc'))}</span></li>")  # fmt: skip
    body = "".join(items) or "<li>no galleries: run scripts/gallery_build.py</li>"
    return (f"<!doctype html><html lang=en><head><meta charset=utf-8><title>Galleries</title>"
            f"<style>{CSS}</style></head><body><h1>Galleries</h1><ul>{body}</ul></body></html>")


def make_handler(galleries: Galleries, store: LabelStore, token: str, port_ref: list[int]):
    write_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        server_version = "oddball-gallery/1"
        sys_version = ""

        def log_message(self, fmt, *args):  # quieter: one line per request, no bodies
            pass

        # ---- helpers
        def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy",
                             "default-src 'self'; img-src 'self'; style-src 'unsafe-inline'; "
                             "script-src 'unsafe-inline'; connect-src 'self'; "
                             "frame-ancestors 'none'")  # fmt: skip
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, code: int, obj) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def _host_ok(self) -> bool:
            host = self.headers.get("Host", "")
            return host in {f"127.0.0.1:{port_ref[0]}", f"localhost:{port_ref[0]}"}

        # ---- GET
        def do_GET(self):
            if not self._host_ok():
                return self._json(403, {"error": "bad host"})
            path = self.path.split("?", 1)[0]
            if path == "/":
                only = galleries.names()
                if galleries.only and only:
                    return self._send(302, b"", "text/plain",
                                      {"Location": f"/g/{quote(only[0])}/"})  # fmt: skip
                return self._send(200, index_html(galleries).encode(), "text/html; charset=utf-8")
            parts = path.split("/")
            if len(parts) >= 3 and parts[1] == "g":
                m = galleries.get(parts[2])
                if m is None:
                    return self._json(404, {"error": "no such gallery"})
                rest = "/".join(parts[3:])
                if rest == "":
                    body = page_html(m, token).encode()
                    return self._send(200, body, "text/html; charset=utf-8")
                if rest == "state":
                    latest = store.latest()
                    out = {}
                    for c in m["cards"]:
                        r = latest.get((int(c["tic"]), int(c["sector"])))
                        if r:
                            out[str(r.tic)] = {"label": r.label, "note": r.note, "utc": r.utc,
                                               "gallery": r.gallery}  # fmt: skip
                    return self._json(200, out)
                if _PNG_RE.match(rest) and rest in m["_pngs"]:
                    f = galleries.root / m["gallery"] / rest
                    if f.is_file():
                        return self._send(200, f.read_bytes(), "image/png")
                return self._json(404, {"error": "not found"})
            return self._json(404, {"error": "not found"})

        do_HEAD = do_GET

        # ---- POST
        def do_POST(self):
            if not self._host_ok():
                return self._json(403, {"error": "bad host"})
            if self.path != "/api/label":
                return self._json(404, {"error": "not found"})
            origin = self.headers.get("Origin")
            if origin is not None and origin not in {f"http://127.0.0.1:{port_ref[0]}",
                                                     f"http://localhost:{port_ref[0]}"}:  # fmt: skip
                return self._json(403, {"error": "bad origin"})
            if not hmac.compare_digest(self.headers.get("X-Label-Token", ""), token):
                return self._json(403, {"error": "bad token (reload the page)"})
            ctype = self.headers.get("Content-Type", "").split(";")[0].strip().lower()
            if ctype != "application/json":
                return self._json(415, {"error": "content type must be application/json"})
            try:
                n = int(self.headers.get("Content-Length", ""))
            except ValueError:
                return self._json(411, {"error": "length required"})
            if not 0 < n <= MAX_BODY:
                return self._json(413, {"error": "body too large"})
            try:
                body = json.loads(self.rfile.read(n).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return self._json(400, {"error": "invalid JSON"})
            if not isinstance(body, dict):
                return self._json(400, {"error": "invalid JSON"})
            try:
                gname = validate_name(body.get("gallery"), "gallery")
                m = galleries.get(gname)
                if m is None:
                    raise LabelError("no such gallery")
                tic = validate_tic(body.get("tic"))
                card = m["_by_tic"].get(tic)
                if card is None:
                    raise LabelError("tic is not in this gallery")
                row = make_row(tic=tic, sector=int(card["sector"]), scorer=m["scorer"],
                               rank=int(card["rank"]), split=card.get("split", ""),
                               reason=card.get("reason", ""), label=body.get("label"),
                               note=body.get("note", ""), gallery=gname)  # fmt: skip
            except LabelError as e:
                return self._json(400, {"error": str(e)})
            with write_lock:
                store.append(row)
            return self._json(200, {"ok": True, "tic": row.tic, "label": row.label,
                                    "note": row.note, "utc": row.utc})  # fmt: skip

        def do_OPTIONS(self):
            return self._json(405, {"error": "no CORS"})

    return Handler


def make_server(root: Path, labels_csv: Path, port: int = 8765, only: str | None = None,
                token: str | None = None) -> ThreadingHTTPServer:  # fmt: skip
    """Build (not start) the server. ``port=0`` picks a free port (tests)."""
    galleries = Galleries(root, only)
    store = LabelStore(labels_csv)
    store.rows()  # fail fast on a corrupt header
    token = token or secrets.token_urlsafe(24)
    port_ref = [port]
    srv = ThreadingHTTPServer((HOST, port), make_handler(galleries, store, token, port_ref))
    port_ref[0] = srv.server_address[1]
    srv.daemon_threads = True
    srv.label_token = token  # type: ignore[attr-defined]
    return srv
