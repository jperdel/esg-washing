"""
ets_fetch.py
------------
Issue #2, criterio externo gratuito de "sustancia": emisiones verificadas del
Regimen de Comercio de Derechos de Emision de la UE (EU ETS).

Descarga, sin registro ni credenciales, las fuentes publicas y deja un
manifiesto con URL, fecha de descarga, tamano y SHA-256:

  1. Comision Europea (DG CLIMA), Union Registry:
     - verified_emissions_2025_en.xlsx  emisiones verificadas por instalacion,
       2008-2025 (extraccion 01-04-2026)
     - compliance_2025_code_en.xlsx     estado de cumplimiento por instalacion
     - policy_ets_registry_operators_ets_en.xlsx  operador (titular de la cuenta),
       n.o de registro mercantil y LEI por instalacion
  2. EUETS.INFO (J. Abrell), version octubre 2024 (EUTL 15.1.3), CSV
     procesados: cuentas (con el BvD ID del emparejamiento JRC-Orbis),
     titulares, instalaciones, cumplimiento. Mas su documentacion.
  3. GLEIF API (Level 1 y Level 2, CC0): LEI de cada matriz del panel y sus
     filiales consolidadas (ultimate-children).
  4. Wikidata (CC0): ingresos totales por ejercicio (P2139) de cada matriz, por
     ISIN; y BCE: tipo de cambio medio anual USD/EUR. Solo para la intensidad.

Copia del manifiesto (sin nombres): results/issue2_ets/fuentes_ets.json.
Los ficheros descargados son datos no fiables: van a data/issue2_ets/raw y
data/issue2_ets/gleif (data/ no se versiona) y se leen con `python -I`.

    python -I scripts/ets_fetch.py [--skip-raw] [--skip-gleif] [--skip-revenue]

Entrada interna (no versionada, contiene nombres de empresas):
    data/issue2_ets/ets_firms_internal.csv   firm;legal_name;isin;patterns;exclude
        firm        = nombre de la empresa tal como aparece en
                      results/analysis_v3/firm_ids_internal.csv
        legal_name  = denominacion legal (respaldo para buscar el LEI)
        isin        = ISIN de la accion cotizada (via principal al LEI)
        patterns    = expresiones regulares (minusculas) para el cruce por
                      nombre con operadores e instalaciones (ets_match.py)
        exclude     = expresiones que anulan un acierto (escisiones, homonimos)
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "issue2_ets"
RAW = DATA / "raw"
GLEIF_DIR = DATA / "gleif"

EC = "https://climate.ec.europa.eu/document/download/"
SOURCES = {
    "verified_emissions_2025_en.xlsx":
        EC + "53018483-62b3-499e-9ab9-b4a831cc44f4_en?filename=verified_emissions_2025_en.xlsx",
    "compliance_2025_code_en.xlsx":
        EC + "f7de3320-ca3d-43d1-afd4-bf0b5d3d355f_en?filename=compliance_2025_code_en.xlsx",
    "policy_ets_registry_operators_ets_en.xlsx":
        EC + "ab2c1214-decb-40bc-bb0d-d37f080bdebd_en?filename=policy_ets_registry_operators_ets_en.xlsx",
    "eutl_2024_202410.zip":
        "https://euets-info-public.s3.eu-central-1.amazonaws.com/eutl_2024_202410.zip",
    "Description_EUTL_database.pdf":
        "https://euets-info-public.s3.eu-central-1.amazonaws.com/Description_EUTL_database.pdf",
}
GLEIF = "https://api.gleif.org/api/v1/"
UA = {"User-Agent": "Mozilla/5.0 (research; issue2-ets)"}


def get(url: str, timeout: int = 300, headers: dict | None = None) -> bytes:
    """GET con reintentos. Si el almacen de certificados de Python esta
    caducado (SSL), recurre a curl del sistema."""
    h = {**UA, **(headers or {})}
    req = urllib.request.Request(url, headers=h)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.URLError as e:
            if "CERTIFICATE" in str(e):
                cmd = ["curl", "-sSL", "-m", str(timeout)]
                for k, v in h.items():
                    cmd += ["-H", f"{k}: {v}"]
                return subprocess.run(cmd + [url], check=True, capture_output=True).stdout
            if attempt == 3:
                raise
        except Exception:  # noqa: BLE001
            if attempt == 3:
                raise
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(url)


def fetch_raw(manifest: list[dict]) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        p = RAW / name
        if not p.exists():
            p.write_bytes(get(url, timeout=900))
        b = p.read_bytes()
        manifest.append({"file": f"raw/{name}", "url": url, "bytes": len(b),
                         "sha256": hashlib.sha256(b).hexdigest(),
                         "downloaded": dt.datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")})


def gleif_json(path: str, params: dict) -> dict:
    url = GLEIF + path + "?" + urllib.parse.urlencode(params)
    return json.loads(get(url, timeout=120))


def find_lei(legal_name: str, isin: str | None = None) -> dict | None:
    """LEI de la matriz. Primero por el ISIN de la accion (mapa ISIN-LEI de
    GLEIF, inequivoco); si no hay, por nombre legal exacto y luego texto libre,
    descartando sucursales y prefiriendo la entidad ACTIVA sin matriz."""
    cands = []
    queries = ([("filter[isin]", isin)] if isinstance(isin, str) and isin else []) + [
        ("filter[entity.legalName]", legal_name), ("filter[fulltext]", legal_name)]
    for key, val in queries:
        js = gleif_json("lei-records", {key: val, "page[size]": 20})
        cands = [d for d in js.get("data", [])
                 if d["attributes"]["entity"]["status"] == "ACTIVE"
                 and d["attributes"]["entity"].get("category") != "BRANCH"]
        if cands:
            break
    if not cands:
        return None

    def score(d):
        a = d["attributes"]
        nm = a["entity"]["legalName"]["name"].lower()
        exact = nm == legal_name.lower()
        rel = d.get("relationships", {})
        has_parent = "ultimate-parent" in rel and bool(
            rel["ultimate-parent"].get("links", {}).get("lei-record"))
        return (exact, not has_parent, -len(nm))

    best = max(cands, key=score)
    a = best["attributes"]
    return {"lei": a["lei"], "gleif_name": a["entity"]["legalName"]["name"],
            "gleif_country": a["entity"]["legalAddress"]["country"]}


def children(lei: str) -> list[dict]:
    out, page = [], 1
    while True:
        try:
            js = gleif_json(f"lei-records/{lei}/ultimate-children",
                            {"page[size]": 200, "page[number]": page})
        except Exception:  # noqa: BLE001  (404 = sin filiales reportadas)
            break
        for d in js.get("data", []):
            a = d["attributes"]
            out.append({"parent_lei": lei, "child_lei": a["lei"],
                        "child_name": a["entity"]["legalName"]["name"],
                        "child_country": a["entity"]["legalAddress"]["country"],
                        "child_status": a["entity"]["status"]})
        last = js.get("meta", {}).get("pagination", {}).get("lastPage", 1)
        if page >= last:
            break
        page += 1
        time.sleep(0.3)
    return out


def fetch_gleif(manifest: list[dict]) -> None:
    GLEIF_DIR.mkdir(parents=True, exist_ok=True)
    firms = pd.read_csv(DATA / "ets_firms_internal.csv", sep=";", encoding="utf-8")
    rows, kids = [], []
    for f in firms.itertuples():
        hit = find_lei(f.legal_name, f.isin) or {}
        rows.append({"firm": f.firm, "legal_name": f.legal_name, **hit})
        if hit.get("lei"):
            kids.extend(children(hit["lei"]))
        time.sleep(0.3)
    pd.DataFrame(rows).to_csv(GLEIF_DIR / "parents_internal.csv", sep=";", index=False)
    pd.DataFrame(kids).to_csv(GLEIF_DIR / "children_internal.csv", sep=";", index=False)
    meta = gleif_json("lei-records", {"filter[lei]": rows[0].get("lei", ""), "page[size]": 1})
    manifest.append({"file": "gleif/parents_internal.csv + children_internal.csv",
                     "url": GLEIF + "lei-records/{lei}/ultimate-children",
                     "golden_copy": meta.get("meta", {}).get("goldenCopy", {}).get("publishDate"),
                     "downloaded": dt.datetime.now().isoformat(timespec="seconds"),
                     "n_parents_found": int(sum(bool(r.get("lei")) for r in rows)),
                     "n_children": len(kids)})


WD_SPARQL = "https://query.wikidata.org/sparql"
ECB_FX = "https://data-api.ecb.europa.eu/service/data/EXR/A.USD.EUR.SP00.A?format=csvdata"


def fetch_revenue(manifest: list[dict]) -> None:
    """Denominador gratuito para la intensidad: ingresos totales (Wikidata
    P2139, con fecha P585 y unidad; CC0) por ISIN (P946), y tipo de cambio
    medio anual USD/EUR del BCE para convertir los valores en dolares."""
    out = DATA / "revenue"
    out.mkdir(parents=True, exist_ok=True)
    firms = pd.read_csv(DATA / "ets_firms_internal.csv", sep=";", encoding="utf-8", keep_default_na=False)
    vals = " ".join(f'"{i}"' for i in firms["isin"])
    q = ("SELECT ?isin ?rev ?date ?unit WHERE { VALUES ?isin { %s } ?item wdt:P946 ?isin . "
         "?item p:P2139 ?st . ?st psv:P2139 ?v . ?v wikibase:quantityAmount ?rev ; "
         "wikibase:quantityUnit ?unit . OPTIONAL { ?st pq:P585 ?date } }" % vals)
    b = get(WD_SPARQL + "?" + urllib.parse.urlencode({"query": q}), timeout=180,
            headers={"Accept": "text/csv", "User-Agent": "research-issue2-ets/0.1 (academic, no-contact)"})
    (out / "wikidata_revenue_internal.csv").write_bytes(b)
    fx = get(ECB_FX, timeout=120)
    (out / "ecb_usd_eur_annual.csv").write_bytes(fx)
    now = dt.datetime.now().isoformat(timespec="seconds")
    manifest.append({"file": "revenue/wikidata_revenue_internal.csv", "url": WD_SPARQL + " (P2139 por ISIN P946)",
                     "licence": "CC0", "downloaded": now, "bytes": len(b)})
    manifest.append({"file": "revenue/ecb_usd_eur_annual.csv", "url": ECB_FX, "downloaded": now, "bytes": len(fx)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-raw", action="store_true")
    ap.add_argument("--skip-gleif", action="store_true")
    ap.add_argument("--skip-revenue", action="store_true")
    a = ap.parse_args()
    manifest: list[dict] = []
    mpath = DATA / "manifest.json"
    if mpath.exists():
        old = json.loads(mpath.read_text(encoding="utf-8"))
        manifest = [m for m in old if not ((not a.skip_raw and m["file"].startswith("raw/"))
                                           or (not a.skip_gleif and m["file"].startswith("gleif/"))
                                           or (not a.skip_revenue and m["file"].startswith("revenue/")))]
    if not a.skip_raw:
        fetch_raw(manifest)
    if not a.skip_gleif:
        fetch_gleif(manifest)
    if not a.skip_revenue:
        fetch_revenue(manifest)
    mpath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    # Copia citable del manifiesto (sin nombres de empresas: solo fuentes y huellas)
    out = ROOT / "results" / "issue2_ets"
    out.mkdir(parents=True, exist_ok=True)
    (out / "fuentes_ets.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
