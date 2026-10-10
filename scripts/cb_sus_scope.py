"""
cb_sus_scope.py
---------------
Comprobacion de alcance del SUS (issue #4): densidad de vocabulario ESG
(SUS 'density', la del paper) recalculada solo sobre los pasajes que el
detector de ClimateBERT marca como climaticos, y sobre el resto.

El SUS se cuenta sobre el texto lematizado, asi que necesita el mismo
TextProcessor (spaCy en_core_web_md, frases protegidas) que el pipeline; corre
en el entorno del pipeline, no en .venv-cb:

    .venv-cb/Scripts/python.exe scripts/cb_analysis.py --export-scope
    <python del pipeline> scripts/cb_sus_scope.py
    .venv-cb/Scripts/python.exe scripts/cb_analysis.py

Entrada:  data/climatebert/scope_texts.json.gz  {doc_key: {"clim": texto, "nonclim": texto}}
Salida:   data/climatebert/sus_scope.csv        doc_key; SUS_clim; SUS_nonclim; SUS_passages
(SUS_passages: menciones y palabras de las dos partes sumadas; control frente
al SUS del paper, que se mide sobre todo el texto extraido).
(cache sin versionar: lleva las claves internas de los informes).
"""

from __future__ import annotations

import gzip
import json
import os
import sys
import time
from pathlib import Path

os.environ["ESG_RUN_TAG"] = "v3"
os.environ["ESG_INCLUDE_SECTORAL"] = "1"

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from loguru import logger  # noqa: E402

import config  # noqa: E402
from esgsi_analyzer import ESGSIAnalyzer  # noqa: E402
from text_processor import TextProcessor  # noqa: E402
from cb_common import CACHE_DIR  # noqa: E402

N_PROC = int(os.environ.get("ESG_CB_PROC", 6))


def main():
    logger.remove()
    with gzip.open(CACHE_DIR / "scope_texts.json.gz", "rt", encoding="utf-8") as fh:
        T = json.load(fh)
    tp = TextProcessor(extra_sw=config.PERSONAL_SW, spacy_model=config.SPACY_MODEL)
    an = ESGSIAnalyzer(keywords=list(config.ESG_KEYWORDS), hedge_words=config.HEDGE_KEYWORDS,
                       quant_patterns=config.QUANT_PATTERNS, sus_mode="density")
    t0 = time.time()
    keys = sorted(T)
    out = {"doc_key": keys}
    m, t = {}, {}
    for part in ["clim", "nonclim"]:
        # Igual que TextProcessor.preprocess, con varios procesos.
        docs = tp.nlp.pipe((tp.prepare(T[k][part]) for k in keys), n_process=N_PROC, batch_size=1)
        clean = [" ".join(tp.lemmas_from_doc(d)) for d in docs]
        an.reset_cache()
        m[part] = an._counts(clean).sum(axis=1).astype(float)
        t[part] = np.array([len(c.split()) for c in clean], dtype=float)
        out[f"SUS_{part}"] = m[part] / np.maximum(t[part], 1) * 100
        print(f"{part}: {time.time() - t0:.0f}s", flush=True)
    # Todos los pasajes: menciones y palabras de las dos partes.
    out["SUS_passages"] = (m["clim"] + m["nonclim"]) / np.maximum(t["clim"] + t["nonclim"], 1) * 100
    pd.DataFrame(out).to_csv(CACHE_DIR / "sus_scope.csv", sep=";", index=False)


if __name__ == "__main__":
    main()
