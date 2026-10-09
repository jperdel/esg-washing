"""
ctrv_adrulings.py
-----------------
Issue #2 (H2): resoluciones de autorregulacion publicitaria sobre las 49
empresas (y sus marcas), con atencion especial a las alegaciones
ambientales (greenwashing). Dos organismos con archivo publico, gratuito y
rastreable sin registro (robots.txt lo permite):

  JDP  Jury de Deontologie Publicitaire (Francia, ARPP): listado /avis/ con
       fecha, anunciante, medio y fallo ("Plainte fondee", "rejetee"...).
  RCC  Reclame Code Commissie (Paises Bajos): buscador de resoluciones con
       fecha, motivo y fallo; se buscan terminos ambientales desde 2017.

    python -I scripts/ctrv_adrulings.py [--pause 3]

Emparejamiento: regex de empresa y de marcas (columnas `regex` y `brands` del
fichero interno) sobre el anunciante (JDP) o el primer parrafo de la
publicidad impugnada (RCC). Categoria GW si el texto de la resolucion contiene
terminos ambientales; en otro caso S (etica publicitaria: imagen de la mujer,
seguridad, alcohol...). Severidad: 2 = queja estimada (JDP "fondee", RCC
"Aanbeveling"); 1 = queja desestimada u otra resolucion. Fecha = fecha de la
resolucion.

No se han automatizado (documentado en el README): Autocontrol (Espana), IAP
(Italia), Deutscher Werberat, JEP (Belgica) y el consejo finlandes: sus
archivos son PDF o buscadores sin campo de anunciante y el volumen esperado
para estas empresas es muy bajo.
"""

from __future__ import annotations

import argparse
import html as htmlmod
import re
import sys
import urllib.parse
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ctrv_common import DATA, Fetcher, html_text, load_terms, save_events  # noqa: E402

ENV = re.compile(r"environnement|(é|e)cologi|d(é|e)veloppement durable|climat|carbone|"
                 r"neutr(e|alit)|recycl|biodiversit|vert(e|s)?\b|plan(è|e)te|durable|"
                 r"duurza|milieu|klimaat|co2|groen|circulair|klimaatneutraal|compens", re.I)
FR_MONTHS = {m: i for i, m in enumerate(
    ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre",
     "octobre", "novembre", "décembre"], 1)}
RCC_TERMS = ["duurzaam", "milieu", "klimaat", "CO2", "groen", "circulair", "recyclebaar",
             "klimaatneutraal", "compensatie"]


def firm_rx(terms: pd.DataFrame) -> list[tuple[str, re.Pattern]]:
    out = []
    for r in terms.itertuples():
        pat = r.regex if not r.brands else f"(?:{r.regex})|(?:{r.brands})"
        out.append((r.firm_id, re.compile(pat, re.I)))
    return out


def matches(text: str, rxs) -> list[str]:
    return [fid for fid, rx in rxs if rx.search(text or "")]


# ---------------------------------------------------------------- JDP ------
def jdp(f: Fetcher, rxs) -> list[dict]:
    rows, page = [], 1
    while True:
        url = "https://www.jdp-pub.org/avis/" + ("" if page == 1 else f"page/{page}/")
        try:
            h = f.get(url)
        except Exception as e:  # noqa: BLE001  (404 al pasar de la ultima pagina)
            print("JDP fin", page, e)
            break
        cards = re.findall(r'(?s)<article class="jdp-avis-auto__card".*?</article>', h)
        if not cards:
            break
        for c in cards:
            link = re.search(r'href="(https://www\.jdp-pub\.org/avis/[^"]+)"', c)
            date = re.search(r'datetime="(\d{4}-\d{2}-\d{2})', c)
            badge = re.search(r'jdp-avis-auto__badge">\s*([^<]+?)\s*<', c)
            lines = [htmlmod.unescape(x).strip() for x in
                     re.findall(r'jdp-avis-auto__title-line">\s*([^<]+?)\s*<', c)]
            adv = lines[0] if lines else ""
            slug = link.group(1).rstrip("/").split("/")[-1] if link else ""
            rows.append({"url": link.group(1) if link else "", "date": date.group(1) if date else None,
                         "outcome": badge.group(1) if badge else "", "advertiser": adv,
                         "medium": " ".join(lines[1:]), "slug": slug})
        page += 1
        if page > 200:
            break
    allj = pd.DataFrame(rows)
    allj.to_csv(DATA / "jdp_all_internal.csv", sep=";", index=False)
    print(f"JDP: {len(allj)} avis listados")
    ev = []
    for r in allj.itertuples():
        firms = matches(r.advertiser + " " + r.slug.replace("-", " "), rxs)
        if not firms:
            continue
        body = html_text(f.get(r.url))
        i = body.find("Les arguments")
        txt = body[: i + 4000] if i > 0 else body[:6000]
        cat = "GW" if ENV.search(txt) else "S"
        sev = 2 if re.search(r"fond(é|e)e", r.outcome, re.I) and not re.search(r"non fond", r.outcome, re.I) else 1
        for fid in firms:
            ev.append({"firm_id": fid, "date": r.date, "source": "jdp", "category": cat,
                       "severity": sev, "title": f"{r.advertiser} - {r.medium} - {r.outcome}",
                       "url": r.url})
    return ev


# ---------------------------------------------------------------- RCC ------
def rcc(f: Fetcher, rxs) -> list[dict]:
    base = "https://www.reclamecode.nl/uitspraken/uitspraken-zoeken/?"
    items = {}
    for term in RCC_TERMS:
        page = 1
        while True:
            q = {"Trefwoord": term, "Dossiernummer": "", "Startdatum": "01-01-2017",
                 "Totdatum": "", "UitspraakSoortId": "0"}
            if page > 1:
                q["pagina"] = str(page)
            h = f.get(base + urllib.parse.urlencode(q))
            blocks = re.findall(r'(?s)<div class="rl_layout242_item">(.*?)Bekijk uitspraak', h)
            if not blocks:
                break
            for b in blocks:
                uid = re.search(r"uitspraakId=(\d+)", b)
                if not uid or uid.group(1) in items:
                    continue
                vals = [htmlmod.unescape(x).strip() for x in
                        re.findall(r'rl_blog8_read-time-text">\s*([^<]*?)\s*<', b)]
                snippet = htmlmod.unescape(re.sub(r"<[^>]+>", " ",
                                                  (re.findall(r"(?s)<p>(.*?)</p>", b) or [""])[0]))
                d = re.search(r"(\d{2}-\d{2}-\d{4})", " ".join(vals))
                items[uid.group(1)] = {"uid": uid.group(1), "fields": " | ".join(vals),
                                       "date": datetime.strptime(d.group(1), "%d-%m-%Y").date() if d else None,
                                       "snippet": " ".join(snippet.split())}
            page += 1
            if page > 100:
                break
        print(f"RCC '{term}': {len(items)} resoluciones acumuladas", flush=True)
    allr = pd.DataFrame(items.values())
    allr.to_csv(DATA / "rcc_all_internal.csv", sep=";", index=False)
    ev = []
    for r in allr.itertuples():
        # anunciante: primer parrafo de la publicidad impugnada (snippet del listado)
        firms = matches(r.snippet[:400], rxs)
        if not firms:
            continue
        sev = 2 if re.search(r"Aanbeveling", r.fields) and not re.search(r"Afwijzing", r.fields) else 1
        cat = "GW" if ENV.search(r.fields + " " + r.snippet) else "S"
        for fid in firms:
            ev.append({"firm_id": fid, "date": r.date, "source": "rcc", "category": cat,
                       "severity": sev, "title": r.snippet[:200],
                       "url": f"https://www.reclamecode.nl/uitspraak/?uitspraakId={r.uid}"})
    return ev


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pause", type=float, default=3.0)
    args = ap.parse_args()
    terms = load_terms()
    terms["brands"] = terms["brands"].fillna("")
    rxs = firm_rx(terms)
    ev = jdp(Fetcher(DATA / "jdp_raw", args.pause), rxs) + rcc(Fetcher(DATA / "rcc_raw", args.pause), rxs)
    ev = pd.DataFrame(ev)
    ev["date"] = pd.to_datetime(ev["date"], errors="coerce")
    ev["year"] = ev["date"].dt.year
    ev["date"] = ev["date"].dt.date
    p = save_events(ev, "adrulings")
    print(f"{len(ev)} eventos -> {p}")
    print(ev.groupby(["source", "year"]).size().unstack(0).to_string())


if __name__ == "__main__":
    main()
