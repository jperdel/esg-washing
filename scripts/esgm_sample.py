"""
esgm_sample.py
--------------
Paso 1 del issue #5: muestra comun de pasajes por informe y sus frases.

Coste: los modelos BERT-base (FinBERT) clasifican unos 25 pasajes/s en la CPU
(i7-9700K, 8 hilos) y los DistilRoBERTa (ESGBERT, NetZero) unos 50/s; los
318.258 pasajes del #4 llevarian 3,5 h por modelo FinBERT y 1,8 h por modelo
ESGBERT. Se usa una muestra aleatoria estratificada por informe: PER_REPORT
pasajes sin reemplazo por informe (todos si tiene menos), semilla fija. Dentro
de cada informe la muestra es autoponderada: la cuota de pasajes de una clase
en la muestra estima sin sesgo la del informe, con error tipico
sqrt(p(1-p)/n * (N-n)/(N-1)) <= 0,5/sqrt(200) = 0,035.

Salidas (cache, no versionada):
    data/esg_models/sample.parquet      doc_key, pid, h, text, n_words, N_report
    data/esg_models/sentences.parquet   h (pasaje), sid, hs (hash de la frase), text
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from esgm_common import CACHE, CB_CACHE, OUT, PER_REPORT, SEED, sentences, text_hash


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    P = pd.read_parquet(CB_CACHE / "passages.parquet", columns=["doc_key", "pid", "h", "text", "n_words"])
    rng = np.random.default_rng(SEED)
    parts = []
    for k, g in P.groupby("doc_key", sort=True):
        n = min(PER_REPORT, len(g))
        idx = rng.choice(len(g), size=n, replace=False)
        parts.append(g.iloc[np.sort(idx)].assign(N_report=len(g)))
    S = pd.concat(parts, ignore_index=True)
    rows = []
    for h, t in S[["h", "text"]].drop_duplicates("h").itertuples(index=False):
        for i, s in enumerate(sentences(t)):
            rows.append({"h": h, "sid": i, "hs": text_hash(s), "text": s})
    F = pd.DataFrame(rows)
    S.to_parquet(CACHE / "sample.parquet", index=False)
    F.to_parquet(CACHE / "sentences.parquet", index=False)
    nr = S.groupby("doc_key").size()
    info = {"pasajes_totales": int(len(P)), "informes": int(P["doc_key"].nunique()),
            "pasajes_por_informe_en_muestra": PER_REPORT, "semilla": SEED,
            "pasajes_muestra": int(len(S)), "textos_distintos_muestra": int(S["h"].nunique()),
            "informes_con_menos_de_200_pasajes": int((nr < PER_REPORT).sum()),
            "fraccion_muestreada": float(len(S) / len(P)),
            "frases_muestra": int(len(F)), "frases_por_pasaje_media": float(F.groupby("h").size().mean()),
            "palabras_por_frase_mediana": float(F["text"].str.split().str.len().median())}
    (OUT / "sample.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
