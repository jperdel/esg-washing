"""
ctrv_bhrrc.py
-------------
Issue #2 (H2): noticias, alegaciones y demandas sobre las 49 empresas
recopiladas por el Business & Human Rights Resource Centre (BHRRC,
business-humanrights.org). Acceso gratuito y sin registro; robots.txt permite
todo con Crawl-Delay 5, que se respeta (pausa de 5 s).

    python -I scripts/ctrv_bhrrc.py [--pause 5]

Usa los mismos puntos de acceso JSON que la propia web publica sin token:
  /en/api/internal/company-name-search/?q=<nombre>   -> id(s) de empresa
  /en/api/internal/explore/?companies=<id>&limit=..&offset=..  -> items
Para cada empresa se toman el id de la matriz y los de sus filiales
etiquetadas como "(part of <matriz>)"; los items se deduplican por pk.

Cada item tiene fecha (`backdate`: fecha de publicacion / del incidente),
titulo, resumen, tipo de contenido (Article, Story, Lawsuit, Briefing,
company response...) y la marca `contains_allegations`.

Eventos (internos): items de tipo noticia/demanda que contienen alegaciones o
cuyo titulo/resumen describe un incidente (regex), excluidas respuestas de la
empresa, benchmarks y encuestas. Categoria E/S/G/GW por palabras clave del
titulo y el resumen. Severidad: 1 = alegacion en prensa; 2 = accion legal o
regulatoria (content type Lawsuit, o demanda/denuncia/multa/condena/sancion/
tribunal en el titulo). focal = 1 si el nombre de la empresa o de una de sus
marcas figura en el titulo (el item va dirigido a ella; si no, suele ser un
informe o noticia sobre muchas empresas).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ctrv_common import DATA, Fetcher, load_terms, save_events  # noqa: E402

BASE = "https://www.business-humanrights.org/en/api/internal/"
LIMIT = 50

GW = re.compile(r"greenwash|misleading|false (environmental|climate|green)|carbon[- ]neutral claim|"
                r"climate claim|green claim|deceptive", re.I)
E = re.compile(r"environment|pollut|climate|emission|deforest|oil spill|spill|toxic|pesticide|"
               r"glyphosate|plastic|water|biodiversity|fossil|coal|gas flaring|methane|carbon|"
               r"mining|land grab|ecocide|chemical|contaminat", re.I)
G = re.compile(r"corrupt|brib|tax (avoid|evas|haven)|money laundering|fraud|cartel|antitrust|"
               r"lobby|sanction|insider|price[- ]fixing|governance|disclosure", re.I)
S = re.compile(r"labou?r|worker|wage|union|strike|forced|child|human rights|indigenous|"
               r"community|communities|displace|health|safety|death|killed|injur|discriminat|"
               r"harass|privacy|surveillance|data|recall|consumer|migrant|abuse|conflict|"
               r"war|military|weapon|arms|israel|gaza|occupied|myanmar|uyghur|xinjiang|russia", re.I)
INCIDENT = re.compile(r"alleg|accus|lawsuit|sue[sd]?\b|court|fine[sd]?\b|penalt|complaint|"
                      r"investigat|probe|protest|report (finds|reveals|alleges)|exposes|"
                      r"criticis|condemn|violat|abuse|scandal|raid|charged|convicted|ruling|"
                      r"misleading|greenwash|linked to|risk of", re.I)
SEVERE = re.compile(r"\bfine[sd]?\b|penalt|convicted|court|ruled|ruling|guilty|settle|"
                    r"sanction|ordered to pay|damages|lawsuit|\bsues?\b|\bsued\b|charged|"
                    r"indict|prosecut|legal action|complaint filed|files? (a )?complaint", re.I)
SKIP_TYPES = {"company response", "company non-response", "briefing", "benchmark", "survey",
              "event", "report", "guidance", "blog", "opinion", "resource", "disclosure"}


def api(f: Fetcher, path: str, **params) -> dict:
    return json.loads(f.get(BASE + path + "?" + urllib.parse.urlencode(params)))


def company_ids(f: Fetcher, name: str, rx: re.Pattern) -> list[tuple[int, str]]:
    res = api(f, "company-name-search/", q=name).get("results", [])
    keep = []
    for r in res:
        dn = r["display_name"]
        parent_ok = rx.search(dn) or dn.lower().startswith(name.lower())
        sub_ok = re.search(r"part of", dn, re.I) and (rx.search(dn) or name.lower() in dn.lower())
        if parent_ok or sub_ok:
            keep.append((r["value"], dn))
    return keep


def items_for(f: Fetcher, cid: int) -> list[dict]:
    out, off = [], 0
    while True:
        d = api(f, "explore/", companies=cid, language="", limit=LIMIT, offset=off)
        out.extend(d.get("results", []))
        if not d.get("next"):
            break
        off += LIMIT
        if off > 20000:
            break
    return out


def classify(text: str) -> str:
    cats = []
    if GW.search(text):
        cats.append("GW")
    if E.search(text):
        cats.append("E")
    if S.search(text):
        cats.append("S")
    if G.search(text):
        cats.append("G")
    return "+".join(cats) if cats else "S"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pause", type=float, default=5.0)
    args = ap.parse_args()
    f = Fetcher(DATA / "bhrrc_raw", pause=args.pause)
    terms = load_terms()
    terms["brands"] = terms["brands"].fillna("")
    allrows, ids = [], []
    for r in terms.itertuples():
        # ids revisados a mano (columna bhrrc_ids del fichero interno) a partir de
        # company_ids(): matriz y filiales consolidadas durante 2018-2024; se
        # excluyen homonimos, concesionarios y escisiones (p. ej. negocios
        # escindidos antes de 2018 o fuera del perimetro del emisor).
        if isinstance(r.bhrrc_ids, str) and r.bhrrc_ids.strip():
            cids = [(int(x), "") for x in r.bhrrc_ids.split(",")]
        elif isinstance(r.bhrrc_ids, str):
            cids = []
        else:
            cids = company_ids(f, r.name, r.rx)
        ids.append({"firm_id": r.firm_id, "n_ids": len(cids), "ids": json.dumps(cids, ensure_ascii=False)})
        seen = set()
        for cid, dn in cids:
            for it in items_for(f, cid):
                if it["pk"] in seen:
                    continue
                seen.add(it["pk"])
                allrows.append({"firm_id": r.firm_id, "pk": it["pk"], "date": it.get("backdate"),
                                "type": it.get("content_type_display"),
                                "allegations": it.get("contains_allegations"),
                                "title": it.get("translated_title") or "",
                                "abstract": re.sub(r"<[^>]+>", " ", it.get("translated_abstract") or ""),
                                "url": "https://www.business-humanrights.org" + (it.get("translated_url") or ""),
                                "via": dn})
        print(f"{r.firm_id}: {len(cids)} ids, {len(seen)} items", flush=True)
    pd.DataFrame(ids).to_csv(DATA / "bhrrc_ids_internal.csv", sep=";", index=False)
    a = pd.DataFrame(allrows)
    a.to_csv(DATA / "bhrrc_items_internal.csv", sep=";", index=False)

    a["date"] = pd.to_datetime(a["date"], errors="coerce")
    txt = a["title"] + " " + a["abstract"]
    is_inc = (a["allegations"] == True) | txt.str.contains(INCIDENT) | (a["type"] == "Lawsuit")  # noqa: E712
    ev = a[is_inc & ~a["type"].fillna("").str.lower().isin(SKIP_TYPES) & a["date"].notna()].copy()
    ev["year"] = ev["date"].dt.year
    ev["source"] = "bhrrc"
    ev["category"] = [classify(t) for t in (ev["title"] + " " + ev["abstract"])]
    ev["severity"] = [2 if (ty == "Lawsuit" or SEVERE.search(t)) else 1
                      for ty, t in zip(ev["type"], ev["title"])]
    rx = {r.firm_id: re.compile(r.regex + (f"|{r.brands}" if r.brands else ""), re.I)
          for r in terms.itertuples()}
    ev["focal"] = [int(bool(rx[f].search(t))) for f, t in zip(ev["firm_id"], ev["title"])]
    ev["date"] = ev["date"].dt.date
    p = save_events(ev, "bhrrc")
    print(f"{len(a)} items, {len(ev)} eventos -> {p}")
    print(ev.groupby("year").size().to_string())


if __name__ == "__main__":
    main()
