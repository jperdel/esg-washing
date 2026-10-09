"""
ctrv_oecdwatch.py
-----------------
Issue #2 (H2): quejas ante los Puntos Nacionales de Contacto (PNC) de la OCDE
por incumplimiento de las Lineas Directrices para Empresas Multinacionales,
segun la base publica de OECD Watch (oecdwatch.org, gratuita, robots.txt sin
restricciones). La base oficial de la OCDE (mneguidelines.oecd.org/database)
esta tras un reto de Cloudflare y no se usa.

    python -I scripts/ctrv_oecdwatch.py [--pause 2]

1. Lee el sitemap de quejas (~500 paginas) y descarga cada ficha con cache.
2. Extrae: titulo, fecha de presentacion, palabras clave, demandado, empresa
   infractora, estado y fecha de cierre.
3. Empareja demandado y titulo con las 49 empresas (regex del fichero interno)
   y clasifica E/S/G por palabras clave. Quejas paralelas presentadas el mismo
   dia ante varios PNC cuentan como un evento.

Fecha del evento = fecha de presentacion de la queja (dia exacto).
Severidad: 1 = queja presentada; 2 = aceptada (paso a buenos oficios /
mediacion o declaracion final con determinacion). Se guarda solo en el
fichero interno events_oecdwatch_internal.csv.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ctrv_common import DATA, Fetcher, html_text, load_terms, match_firms, save_events  # noqa: E402

SITEMAP = "https://www.oecdwatch.org/complaint-sitemap.xml"
E_KW = re.compile(r"environment|climate|pollut|deforest|biodiversity|water|land", re.I)
S_KW = re.compile(r"human rights|labou?r|indigenous|employment|health|safety|community|"
                  r"displacement|consumer|privacy|child", re.I)
G_KW = re.compile(r"disclosure|bribery|corruption|taxation|competition|general policies|"
                  r"due diligence|science", re.I)
FIELDS = ["Date filed", "Keywords", "Countries of harm", "Current status", "Sector", "NCP",
          "Allegations", "Relevant OECD Guidelines", "Outcome", "More details", "Defendant",
          "Company in violation", "Complainants", "Affected people",
          "Date rejected / concluded", "Documents"]


def parse(html: str) -> dict:
    x = html_text(html)
    i = x.find("\nComplaint\n")
    x = x[i:] if i >= 0 else x
    title = x.split("\n")[2].strip() if i >= 0 else ""
    out = {"title": title}
    lines = [ln.strip() for ln in x.split("\n")]
    cur = None
    for ln in lines:
        if ln in FIELDS:
            cur = ln
            out.setdefault(cur, [])
            continue
        if cur and ln:
            out[cur].append(ln)
    for k in list(out):
        if isinstance(out[k], list):
            out[k] = " | ".join(v.rstrip(",") for v in out[k])
    return out


def to_date(s: str):
    for fmt in ("%d %B %Y", "%B %Y", "%Y"):
        try:
            return datetime.strptime(s.strip(), fmt).date()
        except (ValueError, AttributeError):
            continue
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pause", type=float, default=2.0)
    args = ap.parse_args()
    f = Fetcher(DATA / "oecdwatch_raw", pause=args.pause)
    sm = f.get(SITEMAP)
    urls = re.findall(r"<loc>(https://www\.oecdwatch\.org/complaint/[^<]+)</loc>", sm)
    print(f"{len(urls)} fichas de queja")
    terms = load_terms()
    rows, allrows = [], []
    for k, u in enumerate(urls, 1):
        try:
            d = parse(f.get(u))
        except Exception as e:  # noqa: BLE001
            print("ERROR", u, e)
            continue
        d["url"] = u
        allrows.append(d)
        # solo titulo y demandado: "Company in violation" incluye a veces socios o
        # clientes que no son parte de la queja (falsos positivos)
        who = " ; ".join([d.get("Defendant", ""), d.get("title", "")])
        firms = match_firms(who, terms)
        if not firms:
            continue
        date = to_date(d.get("Date filed", "").split("|")[0])
        kw = d.get("Keywords", "")
        cats = [c for c, rx in (("E", E_KW), ("S", S_KW), ("G", G_KW)) if rx.search(kw)] or ["G"]
        st = (d.get("Current status", "") + " " + d.get("Outcome", "")).lower()
        sev = 2 if re.search(r"agreement|mediation|good offices|accepted|determination|"
                             r"breach|in violation|did not observe", st) else 1
        for fid in firms:
            rows.append({"firm_id": fid, "date": date, "year": date.year if date else None,
                         "source": "oecdwatch", "category": "+".join(cats), "severity": sev,
                         "title": d.get("title", ""), "url": u})
        if k % 50 == 0:
            print(k, flush=True)
    pd.DataFrame(allrows).to_csv(DATA / "oecdwatch_all_internal.csv", sep=";", index=False)
    ev = pd.DataFrame(rows)
    # quejas paralelas ante varios PNC el mismo dia = un solo evento
    ev = ev.sort_values("severity", ascending=False).drop_duplicates(["firm_id", "date"])
    ev = ev.sort_values(["firm_id", "date"])
    p = save_events(ev, "oecdwatch")
    print(f"{len(ev)} eventos de {ev['firm_id'].nunique() if len(ev) else 0} empresas -> {p}")
    if len(ev):
        print(ev.groupby("year").size().to_string())


if __name__ == "__main__":
    main()
