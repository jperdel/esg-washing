"""
extraction_stats_v3.py
----------------------
Densidad de palabras clave en la etapa de extraccion (paper, seccion 4) para
la especificacion adoptada (vocabulario v3 con sectoriales).

Cuenta, sobre el texto crudo extraido (`relevant_text` de cada JSON en
data/chunks_lexical_v3_sec/), las coincidencias de la MISMA expresion regular
compilada que usa el extractor (`_ESG_KW_RE` de src/lexical_document_filter.py;
no se reimplementa) por cada 100 palabras (division por espacios, como
`_kw_density`). Da la cifra agregada (suma de coincidencias / suma de palabras)
y la media de los valores por informe, en total y por ano.

Solo lee: data/chunks_lexical_v3_sec/*.json y
results/analysis_v3/extraction_by_doc_internal.csv (ano y marca de duplicado, escritos
por scripts/analysis_v3.py). Escribe results/analysis_v3/extraction_stats.json.

    PYTHONIOENCODING=utf-8 python scripts/extraction_stats_v3.py
"""

from __future__ import annotations

import json
import os
import sys
import unicodedata
from pathlib import Path

# El vocabulario adoptado: debe fijarse ANTES de importar el extractor, que
# compila la regex al importarse.
os.environ["ESG_RUN_TAG"] = "v3"
os.environ["ESG_INCLUDE_SECTORAL"] = "1"

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

import lexical_document_filter as ldf  # noqa: E402

CHUNKS = BASE_DIR / "data" / "chunks_lexical_v3_sec"
BY_DOC = BASE_DIR / "results" / "analysis_v3" / "extraction_by_doc_internal.csv"
OUT = BASE_DIR / "results" / "analysis_v3" / "extraction_stats.json"


def nfc(x: str) -> str:
    return unicodedata.normalize("NFC", str(x))


def main():
    rows = []
    for p in sorted(CHUNKS.rglob("*.json")):
        with open(p, encoding="utf-8") as fh:
            js = json.load(fh)
        text = js.get("relevant_text", "")
        zones = js.get("zones", [])
        rows.append({
            "rel_json": nfc(p.relative_to(CHUNKS).as_posix()),
            "hits": len(ldf._ESG_KW_RE.findall(text)),
            "words": len(text.split()),
            "zones": len(zones),
            "zone_kw_count": int(sum(z.get("kw_count", 0) for z in zones)),
            "zone_words": int(sum(z.get("word_count", 0) for z in zones)),
        })
    ex = pd.DataFrame(rows)
    meta = pd.read_csv(BY_DOC, sep=";")
    meta["rel_json"] = meta["rel_json"].map(nfc)
    ex = ex.merge(meta[["rel_json", "year", "duplicate"]], on="rel_json", how="left", validate="1:1")
    if ex["duplicate"].isna().any():
        raise SystemExit("JSON sin cruce con extraction_by_doc_internal.csv")
    ex["density"] = 100 * ex["hits"] / ex["words"]

    u = ex[~ex["duplicate"].astype(bool)].copy()
    u["year"] = u["year"].astype(int)

    def block(df: pd.DataFrame) -> dict:
        return {
            "n": int(len(df)),
            "zonas": int(df["zones"].sum()),
            "coincidencias": int(df["hits"].sum()),
            "palabras": int(df["words"].sum()),
            "densidad_agregada": float(100 * df["hits"].sum() / df["words"].sum()),
            "densidad_media_por_informe": float(df["density"].mean()),
            "densidad_mediana_por_informe": float(df["density"].median()),
            "densidad_sd_por_informe": float(df["density"].std(ddof=0)),
            "zonas_media": float(df["zones"].mean()),
            "zonas_sd_poblacional": float(df["zones"].std(ddof=0)),
            "zonas_sd_muestral": float(df["zones"].std(ddof=1)),
        }

    S = {
        "nota": "coincidencias de ldf._ESG_KW_RE sobre relevant_text; palabras = split() por espacios",
        "terminos_en_regex": len(ldf._ESG_TERMS),
        "patrones_en_regex": len(ldf._EXTRA_PATTERNS),
        "todos_344": block(ex),
        "sin_duplicado_343": block(u),
        "comprobacion_suma_kw_count_zonas_343": int(u["zone_kw_count"].sum()),
        "comprobacion_suma_word_count_zonas_343": int(u["zone_words"].sum()),
        "por_ano_343": {int(y): block(g) for y, g in u.groupby("year")},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(S, fh, ensure_ascii=False, indent=1)
    print(json.dumps(S, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
