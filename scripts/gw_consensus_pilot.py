"""Issue #2 groundwork: pilot on company-compiled analyst consensus archived in the Wayback Machine.

For a few pilot firms (anonymous firm_id), reads the raw CDX listings saved by gw_wayback_cdx.py,
tags each consensus document with the (quarter, year) in its URL, and for each report year t picks
the best document for the t+1 EPS forecast after publication of report t:
    strict window : consensus before Q1 or Q2 results of t+1 (collected ~Apr-Jul of t+1)
    lenient window: also the Q4-t consensus (collected ~Jan-Feb of t+1, before publication)
HTML consensus pages without a quarter tag are used when captured in Feb-Jun of t+1.
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
TAG1 = re.compile(r"(?:^|[^a-z0-9])(?:pre[-_]?)?q([1-4])[-_ ]?(?:fy[-_ ]?)?(20\d{2}|\d{2})(?!\d)")
TAG2 = re.compile(r"(?<![0-9])(20\d{2}|\d{2})[-_ ]?q([1-4])(?![0-9])")
YEAR_PAGE = re.compile(r"/(20\d{2})/?$")


def tag_of(url: str):
    b = url.lower().split("?")[0]
    base = b.rstrip("/").rsplit("/", 1)[-1]
    m = TAG1.search(base)
    if m:
        q, y = int(m.group(1)), int(m.group(2))
    else:
        m = TAG2.search(base)
        if not m:
            return None
        y, q = int(m.group(1)), int(m.group(2))
    y = y + 2000 if y < 100 else y
    return (y, q) if 2015 <= y <= 2027 else None


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
        if hdr and cells and re.search(r"\beps\b|earnings per share", cells[0] or ""):
            for i, c in enumerate(hdr):
                if re.fullmatch(rf"fy ?({fy}|{fy % 100})", c):
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
        rows = load_listing(raw, fid)
        docs = {}
        pages = []
        for r in rows:
            tg = tag_of(r["url"])
            is_file = bool(re.search(r"\.(pdf|xlsx?)($|\?)", r["url"], re.I))
            if tg and is_file:
                prev = docs.get(tg)
                # prefer spreadsheets (parseable) over PDFs for the same quarter
                if prev is None or (".xls" in r["url"].lower() and ".xls" not in prev["url"].lower()):
                    docs[tg] = r
            elif not is_file and "text/html" in r["mime"]:
                pages.append(r)
        for t in range(2018, 2025):
            choice, kind = None, ""
            for tg, k in (((t + 1, 1), "estricta"), ((t + 1, 2), "estricta"), ((t, 4), "laxa")):
                if tg in docs:
                    choice, kind = docs[tg], k
                    break
            if choice is None:
                win = [p for p in pages if p["ts"][:4] == str(t + 1) and 2 <= int(p["ts"][4:6]) <= 6]
                if win:
                    choice, kind = sorted(win, key=lambda p: p["ts"])[0], "pagina_html"
            rec = {"firm_id": fid, "year": t, "fuente": kind or "ninguna", "captura": choice["ts"][:8] if choice else "",
                   "tiene_bpa": "", "tiene_rango": "", "tiene_n": "", "n_est": "", "alto": "", "bajo": "", "media": ""}
            if choice:
                ext = re.search(r"\.(pdf|xlsx?)", choice["url"], re.I)
                dest = dl / f"{fid.replace(' ', '_')}_{t}{ext.group(0).lower() if ext else '.html'}"
                data = dest.read_bytes() if dest.exists() else fetch(choice["url"], choice["ts"], dest, a.pause)
                if data:
                    txt, xrows = text_of(data, choice["mime"], choice["url"])
                    low = txt.lower()
                    rec["tiene_bpa"] = int(bool(re.search(r"\beps\b|earnings per share|ergebnis je aktie|bpa\b", low)))
                    rec["tiene_rango"] = int(bool(re.search(r"\bmin\b|\bminimum\b|\blow\b", low) and re.search(r"\bmax\b|\bmaximum\b|\bhigh\b", low)))
                    rec["tiene_n"] = int(bool(re.search(r"# ?(of )?estimates?|number of (estimates|analysts|contributors)|no\. of analysts|# ?contributors|n\.? ?of brokers", low)))
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
