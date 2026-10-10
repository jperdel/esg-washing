"""
cb_common.py
------------
Utilidades compartidas por los scripts cb_*.py (issues #2 y #4, validez
convergente frente a los clasificadores abiertos de ClimateBERT; Bingler et al.,
2022, 2024). El piloto (#2) se hizo con las medidas y el corpus anteriores; el
issue #4 lo rehace con las medidas corregidas (SEN por forma exacta, QUANT solo
cifras, HEDGE restringido) y el corpus corregido.

Entradas (solo lectura, en el clon principal; data/ y results/ no se versionan):
    data/chunks_lexical_v3_sec/<PAIS>/<EMPRESA>/<doc>.json   zonas ESG extraidas
    results/analysis_v3/doc_level_internal.csv               scores por informe
Salidas:
    data/climatebert/      (en este worktree, ignorado por git) pasajes y
                           predicciones en cache (por hash del texto del pasaje)
    results/climatebert/   (se versiona con git add -f) agregados sin nombres
                           de empresa: solo firm_label anonimo

Ejecutar con el entorno aislado .venv-cb (torch CPU + transformers):
    .venv-cb/Scripts/python.exe scripts/cb_segment.py
    .venv-cb/Scripts/python.exe scripts/cb_classify.py
    .venv-cb/Scripts/python.exe scripts/cb_analysis.py
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent


def _main_clone() -> Path:
    """Raiz del clon principal (donde viven data/ y results/ del pipeline).
    Si este fichero esta en .claude/worktrees/<x>/scripts, sube hasta el clon;
    si no, es el propio repositorio. Se puede forzar con ESG_MAIN_CLONE."""
    env = os.environ.get("ESG_MAIN_CLONE")
    if env:
        return Path(env)
    parts = BASE_DIR.parts
    if ".claude" in parts and "worktrees" in parts:
        return Path(*parts[: parts.index(".claude")])
    return BASE_DIR


MAIN_DIR = _main_clone()
CHUNKS_DIR = MAIN_DIR / "data" / "chunks_lexical_v3_sec"
SCORES_CSV = MAIN_DIR / "results" / "analysis_v3" / "doc_level_internal.csv"

# Sobrescribibles para pruebas (ESG_CB_CACHE, ESG_CB_OUT).
CACHE_DIR = Path(os.environ.get("ESG_CB_CACHE", BASE_DIR / "data" / "climatebert"))
OUT_DIR = Path(os.environ.get("ESG_CB_OUT", BASE_DIR / "results" / "climatebert"))
# Cache de modelos de Hugging Face (sobrescribible con HF_HOME).
HF_HOME = Path(os.environ.get("HF_HOME", BASE_DIR / ".venv-cb" / "hf"))

# Modelos (Hugging Face, Apache-2.0) y revision fijada para reproducibilidad.
MODELS = {
    "detector": ("climatebert/distilroberta-base-climate-detector",
                 "2c3bc660d45a59e31b35f5d3e365ee4f59fdf76c"),
    "commitment": ("climatebert/distilroberta-base-climate-commitment",
                   "17337c3292df16a8fe93b1505dfe4122d50a4c91"),
    "specificity": ("climatebert/distilroberta-base-climate-specificity",
                    "4ada96ed4bf5c3a7a711282e41f1ab9b29f0ddea"),
    "sentiment": ("climatebert/distilroberta-base-climate-sentiment",
                  "e9f9a94ee4263f5ad5cfc97b8539a497fc88aa7d"),
}

# Segmentacion en pasajes (ver cb_segment.py). Los modelos se entrenaron con
# parrafos de informes de 31-262 palabras (percentiles 1 y 99 de los cuatro
# conjuntos climatebert/climate_*; mediana 64): se descartan los fragmentos
# cortos y se parten los muy largos por frases.
MIN_WORDS = 30
MAX_WORDS = 250

KEY3 = ["País", "Compañía", "Documento"]


def nfc(x) -> str:
    return unicodedata.normalize("NFC", str(x)).strip()


def load_scores() -> pd.DataFrame:
    """Scores por informe (343). Conserva las claves internas para el cruce con
    los JSON; ninguna salida versionada las lleva."""
    df = pd.read_csv(SCORES_CSV, sep=";", encoding="utf-8")
    for c in KEY3:
        df[c] = df[c].map(nfc)
    df["doc_key"] = df["País"] + "/" + df["Compañía"] + "/" + df["Documento"]
    return df


def iter_docs(scores: pd.DataFrame):
    """Itera (doc_key, zonas) sobre los 343 informes puntuados; el JSON del
    duplicado conocido (344 -> 343) no esta en doc_level y se omite."""
    keys = set(scores["doc_key"])
    for p in sorted(CHUNKS_DIR.rglob("*.json")):
        k = nfc(p.parent.parent.name) + "/" + nfc(p.parent.name) + "/" + nfc(p.name)
        if k not in keys:
            continue
        with open(p, encoding="utf-8") as fh:
            js = json.load(fh)
        yield k, [z_["text"] for z_ in js.get("zones", [])]


def iter_raw(scores: pd.DataFrame):
    """Itera (doc_key, texto crudo) con el mismo texto sobre el que el pipeline
    mide SEN, QUANT y HEDGE (campo relevant_text del JSON, espacios colapsados
    como en main.py)."""
    keys = set(scores["doc_key"])
    for p in sorted(CHUNKS_DIR.rglob("*.json")):
        k = nfc(p.parent.parent.name) + "/" + nfc(p.parent.name) + "/" + nfc(p.name)
        if k not in keys:
            continue
        with open(p, encoding="utf-8") as fh:
            js = json.load(fh)
        yield k, re.sub(r"\s+", " ", str(js.get("relevant_text", "")).replace("\x00", "")).strip()


_END = re.compile(r"[.!?;:\"')”’]\s*$")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(])")


def paragraphs(zone_text: str) -> list[str]:
    """Parrafos de una zona (bloques separados por linea en blanco). Un bloque
    sin puntuacion final seguido de otro que empieza en minuscula es la misma
    frase cortada por la maquetacion del PDF: se unen."""
    raw = [re.sub(r"\s+", " ", b).strip() for b in zone_text.split("\n\n")]
    raw = [b for b in raw if b]
    out: list[str] = []
    for b in raw:
        if out and not _END.search(out[-1]) and b[:1].islower():
            out[-1] = out[-1] + " " + b
        else:
            out.append(b)
    return out


def split_long(par: str, max_words: int = MAX_WORDS) -> list[str]:
    """Parte un parrafo de mas de max_words palabras en trozos consecutivos de
    frases completas, de tamano parecido y <= max_words cuando es posible."""
    words = par.split()
    n = len(words)
    if n <= max_words:
        return [par]
    k = -(-n // max_words)
    target = n / k
    sents = _SENT.split(par)
    chunks, cur, cw = [], [], 0
    for s in sents:
        sw = len(s.split())
        if cur and cw + sw > target * 1.15 and len(chunks) < k - 1:
            chunks.append(" ".join(cur))
            cur, cw = [], 0
        cur.append(s)
        cw += sw
    if cur:
        chunks.append(" ".join(cur))
    # Una "frase" de mas de max_words (listas sin puntuacion): corte por palabras.
    out = []
    for c in chunks:
        w = c.split()
        if len(w) <= max_words * 1.5:
            out.append(c)
        else:
            for i in range(0, len(w), max_words):
                out.append(" ".join(w[i:i + max_words]))
    return out


def text_hash(text: str) -> str:
    """Clave de cache de las predicciones: SHA-1 del texto exacto del pasaje.
    Un pasaje que no cambia entre extracciones no se vuelve a clasificar."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def setup_hf_env():
    os.environ.setdefault("HF_HOME", str(HF_HOME))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
