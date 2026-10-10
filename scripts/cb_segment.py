"""
cb_segment.py
-------------
Paso 1 de la comprobacion con ClimateBERT (issues #2 y #4): parte el texto ESG
extraido de cada informe en pasajes del tamano de un parrafo y comprueba el
idioma.

Unidad: Bingler et al. (2022, 2024) clasifican parrafos de los informes; los
modelos publicados avisan de que se entrenaron con parrafos y pueden fallar con
frases. Aqui un pasaje es un parrafo de las zonas ESG ya extraidas (bloques
separados por linea en blanco, reuniendo las frases cortadas por la maquetacion),
con al menos MIN_WORDS palabras; los de mas de MAX_WORDS se parten por frases
(cb_common). Ningun pasaje supera el limite de 512 tokens de DistilRoBERTa salvo
casos raros, que el tokenizador trunca.

Idioma: (i) proporcion de palabras funcionales inglesas en cada pasaje (todos);
(ii) langdetect (semilla fija) sobre una muestra de hasta LANG_SAMPLE pasajes
por informe.

Salidas (cache, no versionada):
    data/climatebert/passages.parquet  doc_key, pid, h, text, n_words, en_share
    (h = SHA-1 del texto: clave de la cache de predicciones)
    data/climatebert/lang_sample.parquet  doc_key, pid, lang
y resumen versionable sin nombres:
    results/climatebert/segmentation.json
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
from langdetect import DetectorFactory, detect
from langdetect.lang_detect_exception import LangDetectException

from cb_common import (CACHE_DIR, MAX_WORDS, MIN_WORDS, OUT_DIR, iter_docs, load_scores,
                       paragraphs, split_long, text_hash)

LANG_SAMPLE = 60
SEED = 20261009

# Palabras funcionales inglesas frecuentes (no aparecen en frances, aleman, etc.
# salvo casualmente); su cuota en un texto ingles ronda el 30-40 %.
EN_FUNC = set("""the of and to in is are was were be been that this these those with for on
by as at from it its which or an have has had not but their they we our will would can
should also than such more other into over under""".split())
WORD = re.compile(r"[A-Za-zÀ-ɏ]+")


def en_share(text: str) -> float:
    w = WORD.findall(text.lower())
    return float(np.mean([x in EN_FUNC for x in w])) if w else 0.0


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scores = load_scores()
    rows, stats = [], []
    for k, zones in iter_docs(scores):
        n_par = w_all = w_kept = 0
        pid = 0
        for zt in zones:
            for par in paragraphs(zt):
                nw = len(par.split())
                n_par += 1
                w_all += nw
                if nw < MIN_WORDS:
                    continue
                w_kept += nw
                for piece in split_long(par):
                    rows.append({"doc_key": k, "pid": pid, "text": piece,
                                 "n_words": len(piece.split())})
                    pid += 1
        stats.append({"doc_key": k, "paragraphs": n_par, "words": w_all, "words_kept": w_kept,
                      "passages": pid})
    P = pd.DataFrame(rows)
    P["en_share"] = P["text"].map(en_share)
    P.insert(2, "h", P["text"].map(text_hash))
    S = pd.DataFrame(stats)
    assert len(S) == len(scores) == 343, len(S)

    DetectorFactory.seed = 0
    rng = np.random.default_rng(SEED)
    lang_rows = []
    for k, g in P.groupby("doc_key"):
        take = g.sample(n=min(LANG_SAMPLE, len(g)), random_state=int(rng.integers(1 << 31)))
        for r in take.itertuples():
            try:
                lg = detect(r.text)
            except LangDetectException:
                lg = "unk"
            lang_rows.append({"doc_key": k, "pid": r.pid, "lang": lg})
    L = pd.DataFrame(lang_rows)

    P.to_parquet(CACHE_DIR / "passages.parquet", index=False)
    L.to_parquet(CACHE_DIR / "lang_sample.parquet", index=False)

    by_doc = L.assign(en=L["lang"].eq("en")).groupby("doc_key")["en"].mean()
    words_by_doc = P.groupby("doc_key").apply(
        lambda g: np.average(g["en_share"], weights=g["n_words"]), include_groups=False)
    summ = {
        "informes": int(len(S)),
        "parrafos": int(S["paragraphs"].sum()),
        "palabras_extraidas": int(S["words"].sum()),
        "palabras_en_pasajes": int(S["words_kept"].sum()),
        "cuota_palabras_en_pasajes": float(S["words_kept"].sum() / S["words"].sum()),
        "pasajes": int(len(P)),
        "pasajes_por_informe": {"mediana": float(S["passages"].median()),
                                "min": int(S["passages"].min()), "max": int(S["passages"].max())},
        "palabras_por_pasaje": {q: float(P["n_words"].quantile(v)) for q, v in
                                [("p05", .05), ("p50", .5), ("p95", .95)]},
        "min_palabras": MIN_WORDS, "max_palabras": MAX_WORDS,
        "idioma": {
            "muestra_langdetect": int(len(L)),
            "cuota_en_muestra": float(L["lang"].eq("en").mean()),
            "otros_idiomas_muestra": L.loc[~L["lang"].eq("en"), "lang"].value_counts().to_dict(),
            "cuota_en_por_informe": {"min": float(by_doc.min()), "p05": float(by_doc.quantile(.05)),
                                     "mediana": float(by_doc.median())},
            "informes_con_menos_de_90pct_en": int((by_doc < 0.9).sum()),
            "cuota_palabras_funcionales_en_por_informe": {
                "min": float(words_by_doc.min()), "mediana": float(words_by_doc.median())},
            "pasajes_con_cuota_funcional_lt_0.10": float((P["en_share"] < 0.10).mean()),
        },
    }
    with open(OUT_DIR / "segmentation.json", "w", encoding="utf-8") as fh:
        json.dump(summ, fh, indent=2, ensure_ascii=False)
    print(json.dumps(summ, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
