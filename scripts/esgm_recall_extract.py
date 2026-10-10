"""
esgm_recall_extract.py
----------------------
Experimento 3 del issue #5 (recall de la extraccion). La extraccion del paper
(src/lexical_document_filter.LexicalDocumentFilter) solo conserva los parrafos
con >= 1 coincidencia del vocabulario por 100 palabras, mas un parrafo vecino a
cada lado, en zonas de >= 150 caracteres. Aqui se vuelve a leer una muestra de
PDF con la misma lectura de parrafos (_extract_paragraphs), se reconstruyen las
zonas con la misma regla (comprobando que coinciden con los JSON guardados) y se
marca que parrafos se extrajeron. El texto extraido y el no extraido se parten en
pasajes con la misma segmentacion que el #4 (cb_common.paragraphs, >= 30
palabras, split_long) y se muestrea un numero fijo de pasajes de cada parte por
informe para clasificarlos con los detectores E/S/G (esgm_classify.py --set recall).

Muestra de informes: 2 por empresa (98), en un diseno equilibrado por ano: con
las empresas en orden aleatorio (semilla fija), a la empresa i le tocan los anos
YEARS[i mod 7] y YEARS[(i + 3) mod 7], de modo que cada ano tiene 14 informes.

Necesita PyMuPDF (fitz) en la MISMA version que la extraccion del paper (1.27.2, entorno
conda del pipeline; con 1.22.5 los bloques cambian y las zonas no coinciden):
    <python del pipeline> scripts/esgm_recall_extract.py

Salidas (cache, no versionada; llevan claves internas):
    data/esg_models/recall_passages.jsonl.gz doc_key, part (in/out), h, text, n_words, kw_density
    data/esg_models/recall_docs.csv          totales de palabras por informe y parte
"""

from __future__ import annotations

import os
import sys
import time

os.environ["ESG_RUN_TAG"] = "v3"
os.environ["ESG_INCLUDE_SECTORAL"] = "1"

from pathlib import Path  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))
sys.path.insert(0, str(BASE / "scripts"))

import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from loguru import logger  # noqa: E402

import lexical_document_filter as L  # noqa: E402
from cb_common import MAIN_DIR, MIN_WORDS, load_scores, nfc, paragraphs, split_long, text_hash  # noqa: E402

CACHE = Path(os.environ.get("ESGM_CACHE", BASE / "data" / "esg_models"))
PDF_DIR = MAIN_DIR / "data" / "pdf"
CHUNKS_DIR = MAIN_DIR / "data" / "chunks_lexical_v3_sec"
SEED = 20261012
YEARS = list(range(2018, 2025))
N_OUT, N_IN = 80, 40  # pasajes muestreados por informe: no extraidos y extraidos (control)


def pick_reports(scores: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    firms = sorted(scores["firm_label"].unique())
    firms = [firms[i] for i in rng.permutation(len(firms))]
    rows = []
    for i, f in enumerate(firms):
        for y in (YEARS[i % 7], YEARS[(i + 3) % 7]):
            rows.append((f, y))
    pick = pd.DataFrame(rows, columns=["firm_label", "year"])
    return scores.merge(pick, on=["firm_label", "year"], how="inner")


def find_pdf(doc_key: str) -> Path | None:
    pais, comp, doc = doc_key.split("/")
    for p in PDF_DIR.iterdir():
        if nfc(p.name) != pais:
            continue
        for c in p.iterdir():
            if nfc(c.name) != comp:
                continue
            for f in c.iterdir():
                if f.suffix.lower() == ".pdf" and nfc(f.stem) == nfc(Path(doc).stem):
                    return f
    return None


def find_json(doc_key: str) -> Path | None:
    pais, comp, doc = doc_key.split("/")
    for p in CHUNKS_DIR.iterdir():
        if nfc(p.name) != pais:
            continue
        for c in p.iterdir():
            if nfc(c.name) != comp:
                continue
            for f in c.iterdir():
                if nfc(f.name) == doc:
                    return f
    return None


def extracted_mask(ldf: L.LexicalDocumentFilter, paras: list[dict]) -> tuple[np.ndarray, list[str]]:
    """Misma regla que LexicalDocumentFilter._build_zones: devuelve que parrafos
    caen en una zona conservada y los textos de esas zonas."""
    n = len(paras)
    scores = [ldf._kw_density(p["text"]) for p in paras]
    hot = [s >= ldf.kw_threshold for s in scores]
    ctx = ldf.context_paras
    wins = sorted((max(0, i - ctx), min(n - 1, i + ctx)) for i, h in enumerate(hot) if h)
    mask = np.zeros(n, dtype=bool)
    zones = []
    if not wins:
        return mask, zones
    merged = [wins[0]]
    for lo, hi in wins[1:]:
        if lo <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    for lo, hi in merged:
        zt = "\n\n".join(p["text"] for p in paras[lo:hi + 1])
        if len(zt) < ldf.min_zone_len:
            continue
        mask[lo:hi + 1] = True
        zones.append(zt)
    return mask, zones


def segment(texts: list[str]) -> tuple[list[str], int, int]:
    """Pasajes (>= MIN_WORDS palabras) de un tramo de parrafos consecutivos;
    devuelve tambien las palabras del tramo y las que quedan en pasajes."""
    block = "\n\n".join(texts)
    out, w_all, w_kept = [], 0, 0
    for par in paragraphs(block):
        nw = len(par.split())
        w_all += nw
        if nw < MIN_WORDS:
            continue
        w_kept += nw
        out.extend(split_long(par))
    return out, w_all, w_kept


def main():
    logger.remove()
    CACHE.mkdir(parents=True, exist_ok=True)
    scores = load_scores()
    R = pick_reports(scores)
    assert len(R) == 98, len(R)
    ldf = L.LexicalDocumentFilter()
    rng = np.random.default_rng(SEED)
    prow, drow = [], []
    t0 = time.time()
    for r in R.itertuples():
        k = r.doc_key
        pdf, js = find_pdf(k), find_json(k)
        if pdf is None or js is None:
            print("FALTA", r.firm_label, r.year, pdf is None, js is None, flush=True)
            continue
        t1 = time.time()
        paras = ldf._extract_paragraphs(pdf)
        mask, zones = extracted_mask(ldf, paras)
        with open(js, encoding="utf-8") as fh:
            stored = [z["text"] for z in json.load(fh)["zones"]]
        stored_sorted = stored  # los JSON guardan las zonas en orden de start_char
        match = zones == stored_sorted
        rec = {"doc_key": k, "firm_label": r.firm_label, "year": int(r.year), "paragraphs": len(paras),
               "paragraphs_in": int(mask.sum()), "zones": len(zones), "zones_json": len(stored),
               "zones_match_json": bool(match),
               "words_pdf_paragraphs": int(sum(len(p["text"].split()) for p in paras))}
        # Tramos consecutivos con el mismo estado.
        parts = {"in": [], "out": []}
        w = {"in": [0, 0], "out": [0, 0]}
        i = 0
        while i < len(paras):
            j = i
            while j + 1 < len(paras) and mask[j + 1] == mask[i]:
                j += 1
            part = "in" if mask[i] else "out"
            ps, wa, wk = segment([p["text"] for p in paras[i:j + 1]])
            parts[part].extend(ps)
            w[part][0] += wa
            w[part][1] += wk
            i = j + 1
        for part, nmax in (("out", N_OUT), ("in", N_IN)):
            ps = parts[part]
            rec[f"words_{part}"] = w[part][0]
            rec[f"words_{part}_passages"] = w[part][1]
            rec[f"passages_{part}"] = len(ps)
            if not ps:
                continue
            take = rng.choice(len(ps), size=min(nmax, len(ps)), replace=False)
            for t in sorted(take):
                txt = ps[t]
                prow.append({"doc_key": k, "part": part, "h": text_hash(txt), "text": txt,
                             "n_words": len(txt.split()), "kw_density": ldf._kw_density(txt)})
        rec["seconds"] = round(time.time() - t1, 1)
        drow.append(rec)
        print(f"{len(drow)}/98 {r.year} paras={len(paras)} in={mask.sum()} zones={len(zones)}/{len(stored)} "
              f"match={match} out_pass={rec['passages_out']} in_pass={rec['passages_in']} "
              f"{rec['seconds']}s", flush=True)
        if len(drow) % 10 == 0:
            pd.DataFrame(prow).to_json(CACHE / "recall_passages.jsonl.gz", orient="records", lines=True, compression="gzip")
            pd.DataFrame(drow).to_csv(CACHE / "recall_docs.csv", sep=";", index=False)
    pd.DataFrame(prow).to_json(CACHE / "recall_passages.jsonl.gz", orient="records", lines=True, compression="gzip")
    pd.DataFrame(drow).to_csv(CACHE / "recall_docs.csv", sep=";", index=False)
    print(f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
