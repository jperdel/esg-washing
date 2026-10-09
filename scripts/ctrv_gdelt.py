"""
ctrv_gdelt.py
-------------
Issue #2 (H2): series diarias de cobertura mediatica negativa ESG por empresa
desde la API DOC 2.0 de GDELT (gratuita, sin registro), 2017-01-01 a la fecha
de corte. Es el sustituto gratuito mas cercano a RepRisk, que tambien es un
cribado de noticias.

    python scripts/ctrv_gdelt.py [--end 20260930] [--pause 8] [--only Energy-01]

Para cada empresa se lanzan 6 consultas en modo `timelinevolraw` (n.o diario
de articulos que casan con la consulta y total de articulos monitorizados):

    ALL  empresa sola (atencion mediatica: denominador / control)
    E    empresa AND terminos de incidente ambiental
    S    empresa AND terminos de incidente social
    G    empresa AND terminos de incidente de gobierno corporativo
    GW   empresa AND terminos de greenwashing / alegaciones enganosas
    NEG  empresa AND (E OR S OR G): cualquier incidente ESG (resultado principal)

Los terminos de busqueda de cada empresa contienen su nombre, asi que viven en
un fichero interno no versionado:
    data/issue2_controversies/firm_terms_internal.csv  (firm_id;gdelt;regex)
La salida en bruto (que repite la consulta) tambien es interna:
    data/issue2_controversies/gdelt_raw/<firm_id>__<tema>.json

Respeta el limite publicado por GDELT (una consulta cada 5 s): pausa fija
entre consultas y espera creciente ante HTTP 429. Reanudable: no repite las
consultas ya guardadas.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "issue2_controversies"
RAW = DATA / "gdelt_raw"
API = "https://api.gdeltproject.org/api/v2/doc/doc"
UA = "esg-research-academic/1.0 (non-commercial; low-rate)"

TOPICS = {
    "ALL": None,
    "E": ('(greenwashing OR pollution OR "oil spill" OR contamination OR deforestation '
          'OR "toxic waste" OR "emissions scandal" OR "environmental damage" '
          'OR "environmental fine" OR "climate lawsuit")'),
    "S": ('("human rights" OR "child labor" OR "child labour" OR "forced labor" '
          'OR "forced labour" OR "working conditions" OR discrimination OR "data breach" '
          'OR "product recall" OR "workplace accident")'),
    "G": ('(bribery OR corruption OR fraud OR "money laundering" OR cartel '
          'OR "tax evasion" OR "insider trading" OR "antitrust fine")'),
    "GW": ('(greenwashing OR "misleading claims" OR "misleading advertising" '
           'OR "false claims" OR "misleading environmental")'),
}
# NEG: incidente ESG de cualquier tipo. GDELT limita la longitud de la consulta,
# asi que se usa un subconjunto de los terminos mas especificos de E, S y G.
TOPICS["NEG"] = ('(greenwashing OR pollution OR "oil spill" OR deforestation '
                 'OR "human rights" OR "child labor" OR "forced labor" OR bribery '
                 'OR corruption OR fraud OR "money laundering" OR cartel)')
# orden de prioridad: primero atencion y NEG para todas las empresas, luego el detalle
ORDER = ["ALL", "NEG", "E", "GW", "S", "G"]


def query_for(firm_q: str, topic: str) -> str:
    t = TOPICS[topic]
    return firm_q if t is None else f"{firm_q} {t}"


def fetch(q: str, start: str, end: str, pause: float, max_retry: int = 60,
          args_wait: float = 90.0) -> dict:
    params = {"query": q, "mode": "timelinevolraw", "format": "json",
              "startdatetime": start + "000000", "enddatetime": end + "235959"}
    url = API + "?" + urllib.parse.urlencode(params)
    wait = args_wait  # espera fija tras un 429 (los reintentos rapidos prolongan el bloqueo)
    for attempt in range(max_retry):
        time.sleep(pause)
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                body = r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(wait)
                continue
            raise
        except Exception:  # timeouts, resets
            time.sleep(wait)
            continue
        if body.lstrip().startswith("{"):
            return json.loads(body)
        # GDELT devuelve errores de sintaxis como texto plano con HTTP 200
        if "limit requests" in body:
            time.sleep(wait)
            continue
        raise ValueError(f"respuesta no JSON: {body[:200]}")
    raise RuntimeError("demasiados reintentos")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20170101")
    ap.add_argument("--end", default="20260930")
    ap.add_argument("--pause", type=float, default=8.0)
    ap.add_argument("--only", default=None)
    ap.add_argument("--wait", type=float, default=90.0)
    ap.add_argument("--topics", default=",".join(ORDER))
    args = ap.parse_args()

    terms = pd.read_csv(DATA / "firm_terms_internal.csv", sep=";", dtype=str)
    RAW.mkdir(parents=True, exist_ok=True)
    jobs = [(r.firm_id, r.gdelt, t) for t in args.topics.split(",") for r in terms.itertuples()
            if args.only is None or r.firm_id == args.only]
    for i, (fid, fq, topic) in enumerate(jobs, 1):
        out = RAW / f"{fid}__{topic}.json"
        if out.exists():
            continue
        q = query_for(fq, topic)
        try:
            d = fetch(q, args.start, args.end, args.pause, args_wait=args.wait)
        except Exception as e:  # noqa: BLE001
            print(f"[{i}/{len(jobs)}] {fid} {topic}: ERROR {e}", flush=True)
            continue
        out.write_text(json.dumps(d), encoding="utf-8")
        n = sum(x["value"] for s in d.get("timeline", []) for x in s["data"]) if d.get("timeline") else 0
        print(f"[{i}/{len(jobs)}] {fid} {topic}: {n} articulos", flush=True)


if __name__ == "__main__":
    main()
