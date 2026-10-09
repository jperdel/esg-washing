"""Issue #2 groundwork: pilot on company-compiled analyst consensus archived in the Wayback Machine.

For a few pilot firms (anonymous firm_id), reads the raw CDX listings saved by gw_wayback_cdx.py,
dates each consensus PDF/XLS (quarter tag or date in its file name, else the archive capture date;
see doc_date) and for each report year t picks the best document for the t+1 EPS forecast after
publication of report t:
    strict window : collected Mar-Jul of t+1 (pre-Q1 / pre-Q2 consensus)
    lenient window: collected Jan-Jul of t+1 (adds the pre-Q4 consensus, just before publication)
The chosen document is downloaded from the archive (raw `id_` capture, one request every few
seconds), stored outside the repo and parsed as data only (run this script with `python -I`).
It records whether the document contains EPS with min/max (or high/low) and the number of
estimates; for spreadsheets it also extracts n, high, low and mean of EPS for FY t+1.

Output (firm_id only): results/issue2_groundwork/piloto_consenso_ir.csv
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import time
import urllib.request
from pathlib import Path

UA = {"User-Agent": "esg-research-groundwork/0.1 (academic; low rate)"}
# quarter tags in file names: q1-24, q1-fy-2020, q423fy23 | 1q21, pre-1q22, 4q2025 | 22q1, 2022-q1 |
# q1h1fy2024, q3andfy2024 | firstquarter2022, fullyear2022
TAGS = [
    (re.compile(r"(?:^|[^a-z0-9])(?:pre[-_ ]?)?q([1-4])[-_ ]?(?:fy[-_ ]?)?(20\d{2}|\d{2})(?!\d)"), "qy"),
    (re.compile(r"(?:^|[^0-9])([1-4])q[-_ ]?(20\d{2}|\d{2})(?!\d)"), "qy"),
    (re.compile(r"(?<![0-9])(20\d{2}|\d{2})[-_ ]?q([1-4])(?![0-9])"), "yq"),
    (re.compile(r"q([1-4])[a-z0-9]{0,6}?fy[-_ ]?(20\d{2}|\d{2})(?!\d)"), "qy"),
]
WORDQ = re.compile(r"(first|second|third|fourth)quarter[-_ ]?(20\d{2})|(fullyear|full-year)[-_ ]?(20\d{2})")
DATE_ISO = re.compile(r"(?<!\d)(20[12]\d)[-_.](\d{2})[-_.](\d{2})(?!\d)")
DATE_DOT = re.compile(r"(?<!\d)(\d{1,2})\.(\d{1,2})\.(20[12]\d|[12]\d)(?!\d)")
OFF_TOPIC = re.compile(r"management-unplugged|press|news|blog|career|job|scientific|consensus-statement|"
                       r"gu%c3%adas|clinic", re.I)
Q_MONTH = {1: 4, 2: 7, 3: 10}  # consensus before Qq results is collected ~1 month after quarter end


def _ym(y: int, q: int, post: bool) -> tuple[int, int]:
    y = y + 2000 if y < 100 else y
    y2, m = (y + 1, 1) if q == 4 else (y, Q_MONTH[q])
    return (y2, m + 2) if post else (y2, m)


def doc_date(url: str, ts: str | None = None):
    """Approximate collection date (year, month, origin) of a consensus document.
    origin: 'trimestre' (quarter tag in the name), 'fecha' (date in the name), 'captura' (archive date)."""
    from urllib.parse import unquote
    base = unquote(url.lower().split("?")[0]).rstrip("/").rsplit("/", 1)[-1]
    post = "post" in base
    for rx, order in TAGS:
        m = rx.search(base)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            q, y = (a, b) if order == "qy" else (b, a)
            y2, mo = _ym(y, q, post)
            return (y2, mo, "trimestre") if 2016 <= y2 <= 2027 else None
    m = WORDQ.search(base)
    if m:
        if m.group(1):
            q = ["first", "second", "third", "fourth"].index(m.group(1)) + 1
            return (*_ym(int(m.group(2)), q, post), "trimestre")
        return (*_ym(int(m.group(4)), 4, post), "trimestre")
    m = DATE_ISO.search(base)
    if m:
        return int(m.group(1)), int(m.group(2)), "fecha"
    m = DATE_DOT.search(base)
    if m:
        a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        mo = b if a > 12 else a  # DD.MM if the first field cannot be a month, else US-style MM.DD
        y = y + 2000 if y < 100 else y
        if 1 <= mo <= 12:
            return y, mo, "fecha"
    if ts:
        return int(ts[:4]), int(ts[4:6]), "captura"
    return None


def in_window(d, t: int, strict: bool) -> bool:
    """report t: forecasts of FY t+1 collected after publication (strict: Mar-Jul of t+1;
    lenient: Jan-Jul of t+1, i.e. also the pre-Q4 consensus collected just before publication)."""
    return bool(d) and d[0] == t + 1 and (3 if strict else 1) <= d[1] <= 7


def load_listing(raw: Path, fid: str):
    rows = []
    for f in sorted(raw.glob(f"ir_{fid}_*.txt")):
        for line in f.read_text(encoding="utf-8").splitlines():
            p = line.split(" ")
            if len(p) >= 4 and re.fullmatch(r"\d{14}", p[1]) and p[3] in ("200", "-"):
                rows.append({"url": p[0], "ts": p[1], "mime": p[2]})
    return rows


def fetch(url: str, ts: str, dest: Path, pause: float) -> bytes | None:
    try:
        req = urllib.request.Request(f"http://web.archive.org/web/{ts}id_/{url}", headers=UA)
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
    except Exception as e:  # noqa: BLE001
        print("  fetch error", type(e).__name__)
        data = None
    time.sleep(pause)
    if data:
        dest.write_bytes(data)
    return data


def text_of(data: bytes, mime: str, url: str) -> tuple[str, object]:
    u = url.lower()
    if data[:4] == b"%PDF":
        import fitz
        doc = fitz.open(stream=data, filetype="pdf")
        return "\n".join(p.get_text() for p in doc), None
    if data[:2] == b"PK" or u.endswith((".xlsx", ".xls")):
        import openpyxl
        wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        rows = [list(r) for ws in wb.worksheets for r in ws.iter_rows(values_only=True)]
        txt = "\n".join(" ".join(str(c) for c in r if c is not None) for r in rows)
        return txt, rows
    t = data.decode("utf-8", errors="replace")
    t = re.sub(r"<script.*?</script>|<style.*?</style>", " ", t, flags=re.S | re.I)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", t)), None


def xlsx_eps(rows, fy: int):
    """Spreadsheet layout with blocks '<period>', '# of estimate', 'High', 'Low', 'Average'."""
    hdr = None
    for r in rows:
        cells = [str(c).strip().lower() if c is not None else "" for c in r]
        if any("of estimate" in c for c in cells) and "high" in cells:
            hdr = cells
        label = next((c for c in cells if c and not re.fullmatch(r"[-\d.,e+ ]+", c)), "")
        if hdr and re.search(r"\beps\b|earnings per share", label):
            for i, c in enumerate(hdr):
                if re.fullmatch(rf"fy\s*({fy}|{fy % 100})", c):
                    try:
                        j = i + 1  # block: value at i, then '# of estimate', 'High', 'Low', 'Average'
                        assert "of estimate" in hdr[j] and hdr[j + 1] == "high" and hdr[j + 2] == "low"
                        return {"n_est": r[j], "alto": r[j + 1], "bajo": r[j + 2], "media": r[j + 3]}
                    except (AssertionError, IndexError):
                        return None
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--dl-dir", required=True)
    ap.add_argument("--firms", required=True, help="comma-separated firm_ids")
    ap.add_argument("--out", default="results/issue2_groundwork")
    ap.add_argument("--pause", type=float, default=5.0)
    a = ap.parse_args()
    raw, dl = Path(a.raw_dir), Path(a.dl_dir)
    dl.mkdir(parents=True, exist_ok=True)

    res = []
    for fid in a.firms.split(","):
        rows = [r for r in load_listing(raw, fid) if not OFF_TOPIC.search(r["url"])]
        files = []
        for r in rows:
            if re.search(r"\.(pdf|xlsx?)($|\?)", r["url"], re.I):
                d = doc_date(r["url"], r["ts"])
                if d:
                    files.append((d, r))
        for t in range(2018, 2025):
            choice, kind = None, ""
            for strict in (True, False):
                cand = [(d, r) for d, r in files if in_window(d, t, strict)]
                if cand:
                    # dated by name first, then by capture; nearest to April; spreadsheets first
                    cand.sort(key=lambda x: (x[0][2] == "captura", abs(x[0][1] - 4), ".xls" not in x[1]["url"].lower()))
                    (d, choice), kind = cand[0], ("estricta" if strict else "laxa") + "_" + cand[0][0][2]
                    break
            rec = {"firm_id": fid, "year": t, "fuente": kind or "ninguna", "captura": choice["ts"][:8] if choice else "",
                   "tiene_bpa": "", "tiene_rango": "", "tiene_n": "", "n_est": "", "alto": "", "bajo": "", "media": ""}
            if choice:
                ext = re.search(r"\.(pdf|xlsx?)", choice["url"], re.I)
                dest = dl / f"{fid.replace(' ', '_')}_{t}{ext.group(0).lower() if ext else '.html'}"
                data = dest.read_bytes() if dest.exists() else fetch(choice["url"], choice["ts"], dest, a.pause)
                if data:
                    txt, xrows = text_of(data, choice["mime"], choice["url"])
                    low = re.sub(r"\s+", " ", txt.lower())
                    rec["tiene_bpa"] = int(bool(re.search(r"\beps\b|earnings per share|ergebnis je aktie|bpa\b", low)))
                    rec["tiene_rango"] = int(bool(re.search(r"\bmin\b|\bminimum\b|\blow\b", low) and re.search(r"\bmax\b|\bmaximum\b|\bhigh\b", low)))
                    rec["tiene_n"] = int(bool(re.search(r"# ?(of )?estimates?|number of (estimates|analysts|contributors)|no\. of analysts|# ?contributors|n\.? ?of brokers|contributors with|\d+ institutions|\d+ analysts|analysts having", low)))
                    if xrows:
                        v = xlsx_eps(xrows, t + 1)
                        if v:
                            rec.update({k: (round(float(x), 4) if isinstance(x, (int, float)) else x) for k, x in v.items()})
            res.append(rec)
            print(fid, t, rec["fuente"], rec["tiene_bpa"], rec["tiene_rango"], rec["tiene_n"], rec["n_est"])

    out = Path(a.out)
    with open(out / "piloto_consenso_ir.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res[0].keys()))
        w.writeheader()
        w.writerows(res)


if __name__ == "__main__":
    main()
