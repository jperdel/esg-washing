"""
esgm_common.py
--------------
Utilidades compartidas por los scripts esgm_*.py (issue #5: validez convergente
con otros modelos ESG abiertos: FinBERT-tone, FinBERT-ESG, ESGBERT y
ClimateBERT-NetZero).

Reutiliza la segmentacion en pasajes de la comprobacion con ClimateBERT
(issue #4, cb_segment.py): data/climatebert/passages.parquet, y las
predicciones del detector climatico (pred_detector.parquet). Esa cache se lee,
nunca se escribe.

Entradas (solo lectura):
    data/climatebert/passages.parquet, pred_detector.parquet   (cache del #4)
    <clon principal>/results/analysis_v3/doc_level_internal.csv, pillar_by_doc.csv,
        term_lemma_pillar_map.csv
    <clon principal>/data/pdf, data/chunks_lexical_v3_sec      (experimento de recall)
Salidas:
    data/esg_models/      cache sin versionar (muestras, predicciones por hash)
    results/esg_models/   agregados sin nombres de empresa (solo firm_label)

Entorno: .venv-cb (torch CPU + transformers), salvo esgm_recall_extract.py,
que necesita PyMuPDF (el Python del pipeline).
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cb_common import (BASE_DIR, MAIN_DIR, MAX_WORDS, MIN_WORDS, load_scores, nfc,  # noqa: F401
                       paragraphs, setup_hf_env, split_long, text_hash)

CB_CACHE = Path(os.environ.get("ESG_CB_CACHE", BASE_DIR / "data" / "climatebert"))
CACHE = Path(os.environ.get("ESGM_CACHE", BASE_DIR / "data" / "esg_models"))
OUT = Path(os.environ.get("ESGM_OUT", BASE_DIR / "results" / "esg_models"))
AV3 = MAIN_DIR / "results" / "analysis_v3"
PDF_DIR = MAIN_DIR / "data" / "pdf"
CHUNKS_DIR = MAIN_DIR / "data" / "chunks_lexical_v3_sec"

SEED = 20261011
PER_REPORT = 200  # pasajes por informe en la muestra comun

# Modelos de Hugging Face con la revision fijada (commit del repositorio).
# kind: 'bert' = BertForSequenceClassification + BertTokenizerFast en minusculas
# (los repos de FinBERT no traen model_type ni tokenizer_config); 'auto' = Auto*.
MODELS = {
    "tone": dict(repo="yiyanghkust/finbert-tone", rev="4921590d3c0c3832c0efea24c8381ce0bda7844b",
                 kind="bert", unit="sentence"),
    "esg4": dict(repo="yiyanghkust/finbert-esg", rev="f79fefa034aa8a969379e23b755369a94c4cd0d3",
                 kind="bert", unit="passage"),
    "esg9": dict(repo="yiyanghkust/finbert-esg-9-categories",
                 rev="af56509508a62691ad52c7a2d67798a6680502e7", kind="bert", unit="passage"),
    "env": dict(repo="ESGBERT/EnvironmentalBERT-environmental",
                rev="804e3f23cf992623168c37694249c6692cc72e8c", kind="auto", unit="passage"),
    "soc": dict(repo="ESGBERT/SocialBERT-social", rev="e0bef3b8ea33ae02e87bfe8057ee78a9665d0476",
                kind="auto", unit="passage"),
    "gov": dict(repo="ESGBERT/GovernanceBERT-governance", rev="9e9f825dad0bf48f4520269f5f2c0e14b16ff8c8",
                kind="auto", unit="passage"),
    "action": dict(repo="ESGBERT/EnvironmentalBERT-action", rev="43c5ef1382934cb4231dd60588f7653a508205cf",
                   kind="auto", unit="passage"),
    "netzero": dict(repo="climatebert/netzero-reduction", rev="25cf57e30613a2156fee1fe3f917036df4a5c0d1",
                    kind="auto", unit="passage"),
}

BATCH = 16
SHARD = 4000
CUT = 0.5


def label_col(lab: str) -> str:
    return "p_" + re.sub(r"[^a-z0-9]+", "_", lab.lower()).strip("_")


# ---------------------------------------------------------------------------
# Frases (unidad de FinBERT-tone, entrenado con frases de informes de analistas)
# ---------------------------------------------------------------------------

_SENT = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9\"“(•\-–])")


def sentences(text: str, min_words: int = 4) -> list[str]:
    """Frases de un pasaje. Los trozos de menos de min_words palabras (siglas,
    numeraciones) se pegan a la frase anterior para no perder texto."""
    out: list[str] = []
    for s in _SENT.split(text):
        s = s.strip()
        if not s:
            continue
        if out and len(s.split()) < min_words:
            out[-1] = out[-1] + " " + s
        else:
            out.append(s)
    if len(out) > 1 and len(out[0].split()) < min_words:
        out[1] = out[0] + " " + out[1]
        out = out[1:]
    return out


# ---------------------------------------------------------------------------
# Inferencia con cache por hash del texto (como cb_classify.py)
# ---------------------------------------------------------------------------

def load_model(name: str):
    setup_hf_env()
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              BertForSequenceClassification, BertTokenizerFast)
    m = MODELS[name]
    if m["kind"] == "bert":
        tok = BertTokenizerFast.from_pretrained(m["repo"], revision=m["rev"], do_lower_case=True)
        mod = BertForSequenceClassification.from_pretrained(m["repo"], revision=m["rev"])
    else:
        tok = AutoTokenizer.from_pretrained(m["repo"], revision=m["rev"])
        mod = AutoModelForSequenceClassification.from_pretrained(m["repo"], revision=m["rev"])
    mod.eval()
    labels = [mod.config.id2label[i] for i in range(mod.config.num_labels)]
    return tok, mod, labels


def predict(texts: list[str], tok, mod) -> np.ndarray:
    import torch
    order = np.argsort([len(t) for t in texts], kind="stable")
    out = np.zeros((len(texts), mod.config.num_labels), dtype=np.float32)
    with torch.inference_mode():
        for i in range(0, len(texts), BATCH):
            idx = order[i:i + BATCH]
            enc = tok([texts[j] for j in idx], padding=True, truncation=True, max_length=512,
                      return_tensors="pt")
            out[idx] = torch.softmax(mod(**enc).logits, dim=-1).numpy()
    return out


def cache_file(name: str) -> Path:
    return CACHE / f"pred_{name}.parquet"


def load_cache(name: str) -> pd.DataFrame:
    parts = []
    if cache_file(name).exists():
        parts.append(pd.read_parquet(cache_file(name)))
    parts += [pd.read_parquet(f) for f in sorted((CACHE / f"pred_{name}_new").glob("part_*.parquet"))]
    if not parts:
        return pd.DataFrame(columns=["h"])
    return pd.concat(parts, ignore_index=True).drop_duplicates("h", keep="first")


def run_model(name: str, T: pd.DataFrame, tag: str = "") -> dict:
    """Clasifica los textos de T (columnas h, text) que no estan en la cache del
    modelo. Bloques de SHARD textos en pred_<name>_new/ (reanudable)."""
    cache = load_cache(name)
    todo = T.drop_duplicates("h")
    n_dist = len(todo)
    todo = todo[~todo["h"].isin(set(cache["h"]))].reset_index(drop=True)
    print(f"[{name}{tag}] {len(T)} textos, {n_dist} distintos, {len(todo)} sin prediccion", flush=True)
    t0 = time.time()
    if len(todo):
        d = CACHE / f"pred_{name}_new"
        d.mkdir(parents=True, exist_ok=True)
        tok, mod, labels = load_model(name)
        start = len(list(d.glob("part_*.parquet")))
        for s in range(-(-len(todo) // SHARD)):
            sub = todo.iloc[s * SHARD:(s + 1) * SHARD]
            t1 = time.time()
            pr = predict(sub["text"].tolist(), tok, mod)
            res = sub[["h"]].copy()
            for j, lab in enumerate(labels):
                res[label_col(lab)] = pr[:, j]
            res.to_parquet(d / f"part_{start + s:04d}.parquet", index=False)
            el = time.time() - t1
            print(f"[{name}{tag}] bloque {s + 1}/{-(-len(todo) // SHARD)}: {len(sub)} en {el:.0f}s "
                  f"({len(sub) / el:.1f}/s)", flush=True)
        load_cache(name).to_parquet(cache_file(name), index=False)
        for f in d.glob("part_*.parquet"):
            f.unlink()
        d.rmdir()
    el = time.time() - t0
    return {"textos": int(len(T)), "textos_distintos": int(n_dist), "inferidos": int(len(todo)),
            "segundos": round(el, 1), "por_segundo": round(len(todo) / el, 1) if len(todo) and el else None,
            "repo": MODELS[name]["repo"], "revision": MODELS[name]["rev"]}
