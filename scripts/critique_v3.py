"""
critique_v3.py
--------------
Recomputes, on the adopted v3 corpus (vocabulary v3 with sectoral entries,
343 unique reports), every quantitative claim of the paper's section 7
(measurement critique: vector normalisation and idf weighting) and section 8
(score-normalisation artefact). Writes a single JSON:

    results/analysis_v3/critique_v3.json

    PYTHONIOENCODING=utf-8 ESG_RUN_TAG=v3 ESG_INCLUDE_SECTORAL=1 \
        python scripts/critique_v3.py

Everything is computed with src/esgsi_analyzer.ESGSIAnalyzer and the lexicons
of src/config.py (the same objects main.py uses), on the corpus loaded and
deduplicated (MD5 of clean_text) by scripts/analysis_v3.load_corpus, exactly
as main.py does. The recomputed components are checked against
results/metrics_v3_sec/results.csv before anything is reported.

Blocks:
  vocab        V and n_max, from the generated lexicons via config.ESG_KEYWORDS
  check        recomputed SUS (three specs), BREADTH, SEN, QUANT vs results.csv
  amplify      L2 invariance I: a synthetic report whose ESG counts are scaled by
               k = 1, 2, 5, 20, re-embedded in the corpus in place of its source
  dilute       L2 invariance II: non-ESG filler appended to a real report
  breadth      what survives L2: correlations of SUS^{tfidf-L2} with BREADTH,
               density, distinct-entry count; the unit-sum bounds
  idf          idf inversion: occurrence shares by idf band; rare-vs-common ratio
  weighting    term weighting vs vector normalisation, component and index level
               (index level: Pearson r, Spearman rho, extreme-decile overlap)
  sublinear    sublinear-TF variants (length-normalised, mean over V), computed by
               validity_v3.sublinear_tf / sublinear_diagnostics (single source:
               1 + ln c, no idf, no L2; full-precision QUANT as in section 9)
  normalisation  score normalisation (z-score vs min-max) for each substance spec,
               label-free: distribution of the index (mean, median, sd,
               quartiles) under each, the mean of each normalised component,
               and Pearson r / Spearman rho / extreme-decile overlap between the
               z-score and min-max versions. No threshold is used anywhere.

Document-choice rule (stated in the paper): the report at the median of
SUS^{dens} (n = 343 is odd, so the median is attained by one report) is the
source of both invariance demonstrations. It is described in the output by
industry, country, document type and year, never by firm name.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

os.environ["ESG_RUN_TAG"] = "v3"
os.environ["ESG_INCLUDE_SECTORAL"] = "1"

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR / "scripts"))

from loguru import logger  # noqa: E402

logger.remove()
logger.add(sys.stderr, level="WARNING")

import config  # noqa: E402
from esgsi_analyzer import ESGSIAnalyzer  # noqa: E402
from analysis_v3 import (  # noqa: E402
    KEY, jsonable, load_corpus, load_results, norm_keys, count_matrix, z,
)
import validity_v3 as V3  # noqa: E402  (single source of the sublinear-TF spec)

RESULTS = BASE_DIR / "results" / "metrics_v3_sec" / "results.csv"
PROCESSED = BASE_DIR / "data" / "clean_v3_sec" / "processed_texts.csv"
CHUNKS = BASE_DIR / "data" / "chunks_lexical_v3_sec"
SECTORS = BASE_DIR / "metadata" / "empresas_supersector.csv"
OUT = BASE_DIR / "results" / "analysis_v3" / "critique_v3.json"

AMPLIFY_K = (1, 2, 5, 20)
DILUTION_MULTIPLE = 2.5      # filler tokens appended = 2.5 x the report's length
N_FILLER_WORDS = 50          # filler drawn from the report's 50 most frequent eligible tokens
IDF_LOW, IDF_HIGH = 1.2, 2.0  # idf bands of the inversion table


def mm(x) -> np.ndarray:
    """Min-max score normalisation to [0, 1] across the cross-section."""
    x = np.asarray(x, dtype=float)
    r = x.max() - x.min()
    return np.zeros_like(x) if r == 0 else (x - x.min()) / r


def pear(a, b) -> float:
    return float(np.corrcoef(np.asarray(a, float), np.asarray(b, float))[0, 1])


def spear(a, b) -> float:
    return float(stats.spearmanr(a, b)[0])


def new_analyzer() -> ESGSIAnalyzer:
    return ESGSIAnalyzer(keywords=config.ESG_KEYWORDS, hedge_words=config.HEDGE_KEYWORDS,
                         quant_patterns=config.QUANT_PATTERNS,
                         ext_weights=config.ESGSI_EXT_WEIGHTS, sus_mode="density")


def sus_all(texts: list[str]) -> dict[str, np.ndarray]:
    """Three SUS specs on a corpus, with a fresh analyzer (caches are corpus-bound)."""
    return new_analyzer().calculate_sus_variants(texts)


def eligible_filler(tokens: list[str], vocab: list[str], n: int) -> list[str]:
    """
    The n most frequent tokens of a report that occur in NO vocabulary entry,
    as a whole word of any entry. A token outside every entry cannot complete a
    vocabulary n-gram, so appending such tokens adds no ESG hit anywhere,
    including across the boundary with the original text.
    """
    vocab_words = {w for e in vocab for w in e.split()}
    cnt = Counter(t for t in tokens if t not in vocab_words and t.isalpha() and len(t) > 2)
    return [w for w, _ in cnt.most_common(n)]


def main():
    t0 = time.time()
    J: dict = {"generado": time.strftime("%Y-%m-%d %H:%M:%S"),
               "entradas": {"results": str(RESULTS), "processed": str(PROCESSED)},
               "regla_documento": "informe en la mediana de SUS_density (n impar)"}

    # ------------------------------------------------------------------ datos
    df = load_results(RESULTS, SECTORS)
    corpus, dedup = load_corpus(PROCESSED)
    raw = norm_keys(pd.read_csv(PROCESSED, sep=";", usecols=KEY + ["clean_text", "raw_text"],
                                encoding="utf-8"))
    raw = raw[raw["clean_text"].isin(set(corpus["clean_text"]))].drop_duplicates("clean_text")
    df = df.merge(raw, on=KEY, how="left", validate="one_to_one")
    if df["clean_text"].isna().any():
        raise ValueError("filas de results.csv sin texto")
    texts = df["clean_text"].astype(str).tolist()
    raws = df["raw_text"].astype(str).tolist()
    n = len(texts)
    vocab = list(config.ESG_KEYWORDS)
    V = len(vocab)
    base = [l.strip().lower() for l in open(BASE_DIR / "metadata" / "esg_terms_lemmatized.txt",
                                            encoding="utf-8") if l.strip() and not l.startswith("#")]
    J["corpus"] = {"n": n, "deduplicacion": {**dedup[0], "descartados": len(dedup) - 1}}
    J["vocab"] = {"V": V, "base": len(set(base)), "sectoriales_extra": V - len(set(base)),
                  "n_max": max(len(k.split()) for k in vocab)}
    print(f"datos n={n}, V={V} ({time.time() - t0:.0f}s)", flush=True)

    # ------------------------------------------------- recomputation / check
    an = new_analyzer()
    sus = an.calculate_sus_variants(texts)
    breadth = an.calculate_breadth_scores(texts)
    sen = an.calculate_sen_scores(texts)
    quant_re = an.calculate_quant_scores(raws)
    # Raw text rebuilt from the extraction JSONs (validity_v3.load_corpus_raw:
    # processed_texts.csv can truncate a raw_text cell at a NUL character) and
    # the full-precision QUANT on it; used by the sublinear block, as in sec. 9.
    rj, _ = V3.load_corpus_raw(PROCESSED, CHUNKS)
    rj = df[KEY].merge(rj[KEY + ["raw_text"]], on=KEY, how="left", validate="one_to_one")
    raws_json = rj["raw_text"].astype(str).tolist()
    quant_full = an.calculate_quant_scores(raws_json)
    # QUANT: results.csv is authoritative (conventions: never reprint a
    # recomputed series). The regex re-run on processed_texts.csv raw_text is
    # only a check, recorded in J["check"]; it matches once the two raw_text
    # cells once truncated at a NUL character have been repaired.
    quant = df["QUANT_Score"].to_numpy(float)
    C, cols = count_matrix(texts, vocab)
    assert cols == vocab
    L = np.array([max(len(t.split()), 1) for t in texts], dtype=float)
    idf = an._idf(texts)
    df_j = (C > 0).sum(axis=0)
    distinct = (C > 0).sum(axis=1)
    esgsi = z(sen) - z(sus["density"])
    J["check"] = {
        "max_abs_dif": {
            "SUS_density": float(np.abs(sus["density"] - df["SUS_density"]).max()),
            "SUS_tfidf_length": float(np.abs(sus["tfidf_length"] - df["SUS_tfidf_length"]).max()),
            "SUS_lagasio": float(np.abs(sus["lagasio"] - df["SUS_lagasio"]).max()),
            "Breadth": float(np.abs(breadth - df["Breadth"]).max()),
            "SEN_Score": float(np.abs(sen - df["SEN_Score"]).max()),
            "QUANT_Score_recalculado": float(np.abs(quant_re - df["QUANT_Score"]).max()),
            "QUANT_precision_completa_JSON": float(np.abs(quant_full - df["QUANT_Score"]).max()),
            "ESGSI": float(np.abs(esgsi - df["ESGSI"]).max()),
        },
        "QUANT_docs_dif_mayor_redondeo": int((np.abs(quant_re - df["QUANT_Score"]) > 6e-5).sum()),
        "QUANT_fuente": "results.csv (QUANT_Score, 4 decimales)",
    }
    print(f"check {J['check']} ({time.time() - t0:.0f}s)", flush=True)

    # Source report for the two invariance demonstrations.
    order = np.argsort(sus["density"], kind="stable")
    src = int(order[n // 2])
    J["documento"] = {"industria": df.loc[src, "industry"], "año": int(df.loc[src, "year"]),
                      "pais": df.loc[src, "country"], "tipo": df.loc[src, "doctype"],
                      "tokens": int(L[src]), "menciones": int(C[src].sum()),
                      "entradas_distintas": int(distinct[src]),
                      "SUS_density": float(sus["density"][src]),
                      "SUS_lagasio": float(sus["lagasio"][src])}

    src_tokens = texts[src].split()
    filler = eligible_filler(src_tokens, vocab, N_FILLER_WORDS)

    # ------------------------------------------------------------- amplify
    # Synthetic report: every vocabulary entry written out as many times as it
    # occurs in the source report, each occurrence followed by one filler token
    # (outside every entry, so occurrences cannot fuse into other n-grams), then
    # filler tokens up to the source's length (the non-ESG remainder). Scaling the block by
    # k multiplies the synthetic count vector by exactly k; presence of every
    # entry is unchanged, so df_j and idf are unchanged across k.
    def block(k: int) -> list[str]:
        parts = []
        f = 0
        for j, e in enumerate(vocab):
            for _ in range(int(C[src, j]) * k):
                parts.append(e)
                parts.append(filler[f % len(filler)])
                f += 1
        return parts

    # The non-ESG remainder is fixed at k = 1 (the source's length minus the
    # written-out block) and is the same for every k.
    n_rest = int(L[src]) - len(" ".join(block(1)).split())
    rest = [filler[i % len(filler)] for i in range(max(n_rest, 0))]

    def synth(k: int) -> str:
        return " ".join(block(k) + rest)

    amp = []
    lag_base = None
    for k in AMPLIFY_K:
        t = texts.copy()
        t[src] = synth(k)
        s = sus_all(t)
        cnt, _ = count_matrix([t[src]], vocab)
        idf_k = TfidfVectorizer(vocabulary=vocab, ngram_range=(1, J["vocab"]["n_max"])).fit(t).idf_
        if lag_base is None:
            lag_base = float(s["lagasio"][src])
            cnt1 = cnt.copy()
        amp.append({"k": k, "tokens": len(t[src].split()), "menciones": int(cnt.sum()),
                    "conteo_proporcional": bool(np.array_equal(cnt, k * cnt1)),
                    "idf_igual_al_corpus_real": float(np.abs(idf_k - idf).max()),
                    "SUS_density": float(s["density"][src]),
                    "SUS_tfidf_length": float(s["tfidf_length"][src]),
                    "SUS_lagasio": float(s["lagasio"][src]),
                    "SUS_lagasio_rel": float(s["lagasio"][src]) / lag_base})
    J["amplify"] = {"filas": amp,
                    "desviacion_max_abs_SUS_lagasio": float(max(abs(a["SUS_lagasio"] - lag_base)
                                                              for a in amp)),
                    "tokens_resto_no_ESG": len(rest),
                    "factor_kL1_Lk": [a["k"] * amp[0]["tokens"] / a["tokens"] for a in amp],
                    "ratio_density_obs": [a["SUS_density"] / amp[0]["SUS_density"] for a in amp]}
    print(f"amplify hecho ({time.time() - t0:.0f}s)", flush=True)

    # -------------------------------------------------------------- dilute
    n_fill = int(round(DILUTION_MULTIPLE * L[src]))
    t = texts.copy()
    t[src] = texts[src] + " " + " ".join(filler[i % len(filler)] for i in range(n_fill))
    s = sus_all(t)
    cnt, _ = count_matrix([t[src]], vocab)
    J["dilute"] = {
        "relleno_tokens": n_fill, "palabras_relleno_distintas": len(filler),
        "original": {"tokens": int(L[src]), "menciones": int(C[src].sum()),
                     "SUS_density": float(sus["density"][src]),
                     "SUS_tfidf_length": float(sus["tfidf_length"][src]),
                     "SUS_lagasio": float(sus["lagasio"][src])},
        "diluido": {"tokens": len(t[src].split()), "menciones": int(cnt.sum()),
                    "SUS_density": float(s["density"][src]),
                    "SUS_tfidf_length": float(s["tfidf_length"][src]),
                    "SUS_lagasio": float(s["lagasio"][src])},
    }
    J["dilute"]["factor_dilucion"] = J["dilute"]["diluido"]["tokens"] / J["dilute"]["original"]["tokens"]
    J["dilute"]["dif_abs_SUS_lagasio"] = abs(J["dilute"]["diluido"]["SUS_lagasio"]
                                             - J["dilute"]["original"]["SUS_lagasio"])
    J["dilute"]["conteos_identicos"] = bool(np.array_equal(cnt[0], C[src]))
    print(f"dilute hecho ({time.time() - t0:.0f}s)", flush=True)

    # ------------------------------------------------------------- breadth
    lag = sus["lagasio"]
    unit_sum = lag * V
    J["breadth"] = {
        "pearson": {"lag_breadth": pear(lag, breadth), "lag_density": pear(lag, sus["density"]),
                    "lag_tfidf_length": pear(lag, sus["tfidf_length"]),
                    "lag_distinct": pear(lag, distinct), "lag_tokens": pear(lag, L),
                    "density_breadth": pear(sus["density"], breadth),
                    "unitsum_sqrt_breadth": pear(unit_sum, np.sqrt(breadth))},
        "spearman": {"lag_breadth": spear(lag, breadth), "lag_density": spear(lag, sus["density"]),
                     "lag_distinct": spear(lag, distinct)},
        "suma_unitaria": {"min": float(unit_sum.min()), "media": float(unit_sum.mean()),
                          "max": float(unit_sum.max()), "cota_superior_sqrtV": float(np.sqrt(V))},
        "entradas_distintas": {"media": float(distinct.mean()), "min": int(distinct.min()),
                               "max": int(distinct.max())},
        "breadth": {"media": float(breadth.mean()), "min": float(breadth.min()),
                    "max": float(breadth.max())},
    }

    # ----------------------------------------------------------------- idf
    occ = C.sum(axis=0)
    tot = occ.sum()
    live = df_j > 0
    low = idf < IDF_LOW
    high = (idf > IDF_HIGH) & live
    # idf < 1.2  <=>  1 + df > (1 + n) / exp(0.2)
    df_cut = (1 + n) / np.exp(IDF_LOW - 1) - 1
    commonest = int(np.argmax(occ))
    lowest_idf = int(np.argmin(idf))
    cand = np.where(df_j >= 2)[0]
    rarest = int(cand[np.lexsort((occ[cand], -idf[cand]))[0]])   # max idf, then fewest occurrences
    J["idf"] = {
        "V": V, "entradas_vivas": int(live.sum()), "entradas_sin_ocurrencias": int((~live).sum()),
        "idf_min": float(idf.min()), "idf_max_vivas": float(idf[live].max()),
        "banda_baja": {"umbral_idf": IDF_LOW, "df_mayor_que": float(df_cut),
                       "df_min_en_banda": int(df_j[low].min()), "entradas": int(low.sum()),
                       "cuota_ocurrencias": float(occ[low].sum() / tot)},
        "banda_alta": {"umbral_idf": IDF_HIGH, "entradas_vivas": int(high.sum()),
                       "cuota_ocurrencias": float(occ[high].sum() / tot),
                       "df_max_en_banda": int(df_j[high].max()) if high.any() else None},
        "banda_media": {"entradas_vivas": int((live & ~low & ~high).sum()),
                        "cuota_ocurrencias": float(occ[live & ~low & ~high].sum() / tot)},
        "ocurrencias_totales": int(tot),
        "mas_frecuente": {"entrada": vocab[commonest], "ocurrencias": int(occ[commonest]),
                          "df": int(df_j[commonest]), "idf": float(idf[commonest])},
        "idf_minimo": {"entrada": vocab[lowest_idf], "ocurrencias": int(occ[lowest_idf]),
                       "df": int(df_j[lowest_idf]), "idf": float(idf[lowest_idf])},
        "mas_rara_df_ge_2": {"regla": "idf maximo con df >= 2; empate: menos ocurrencias",
                             "entrada": vocab[rarest], "ocurrencias": int(occ[rarest]),
                             "df": int(df_j[rarest]), "idf": float(idf[rarest])},
    }
    J["idf"]["ratio_rara_vs_mas_frecuente"] = float(idf[rarest] / idf[commonest])
    top10 = np.argsort(-occ)[:10]
    J["idf"]["top10_ocurrencias"] = [{"entrada": vocab[j], "ocurrencias": int(occ[j]),
                                      "df": int(df_j[j]), "idf": float(idf[j])} for j in top10]

    # ----------------------------------------------------------- weighting
    zsen = z(sen)
    idx = {m: zsen - z(sus[m]) for m in sus}
    J["weighting"] = {
        "pearson_componentes": {"density_tfidf_length": pear(sus["density"], sus["tfidf_length"]),
                                "tfidf_length_lagasio": pear(sus["tfidf_length"], lag),
                                "density_lagasio": pear(sus["density"], lag)},
        "spearman_componentes": {"density_tfidf_length": spear(sus["density"], sus["tfidf_length"]),
                                 "tfidf_length_lagasio": spear(sus["tfidf_length"], lag),
                                 "density_lagasio": spear(sus["density"], lag)},
        "indice_vs_density": {m: {"pearson": pear(v, idx["density"]), "spearman": spear(v, idx["density"]),
                                  "deciles_extremos": V3.decile_overlap(v, idx["density"])}
                              for m, v in idx.items()},
        "indice_tfidf_length_vs_lagasio": {
            "pearson": pear(idx["tfidf_length"], idx["lagasio"]),
            "spearman": spear(idx["tfidf_length"], idx["lagasio"]),
            "deciles_extremos": V3.decile_overlap(idx["tfidf_length"], idx["lagasio"])},
    }

    # ----------------------------------------------------------- sublinear
    # Single source: scripts/validity_v3.sublinear_tf / sublinear_diagnostics
    # (Equation eq:sublinear of section 9: 1 + ln c, no idf, no L2), evaluated
    # against the full-precision QUANT recomputed on the extraction-JSON raw
    # text, exactly as section 9 does. The 4-decimal QUANT_Score column is too
    # coarse for correlations (values ~0.005).
    R = np.array([max(len(t.split()), 1) for t in raws_json], dtype=float)
    subv = V3.sublinear_tf(C, L)
    sub_len, sub_mean = subv["longitud"], subv["media_V"]
    variants = {"density": sus["density"], "tfidf_length": sus["tfidf_length"], "lagasio": lag,
                "sublineal_longitud": sub_len, "sublineal_media_V": sub_mean}
    J["sublinear"] = {
        **V3.sublinear_diagnostics(C, L, R, quant_full),
        "pearson_QUANT": {k: pear(v, quant_full) for k, v in variants.items()},
        "spearman_QUANT": {k: spear(v, quant_full) for k, v in variants.items()},
        "pearson_distintas": {k: pear(v, distinct) for k, v in variants.items()},
        "pearson_density": {k: pear(v, sus["density"]) for k, v in variants.items()},
        "pearson_breadth": {k: pear(v, breadth) for k, v in variants.items()},
        "pearson_tokens": {k: pear(v, L) for k, v in variants.items()},
    }

    # ------------------------------------------------------- normalisation
    # z-score vs min-max, for each substance spec. Label-free: what changes is
    # the location of the index (under min-max its mean is the difference of
    # the two components' min-max means, which depends on their skewness and
    # extremes, not on the reports' relative position) while the ordering is
    # compared with Spearman rho and the extreme-decile overlap.
    def dist(e):
        q1, med, q3 = np.percentile(e, [25, 50, 75])
        return {"media": float(e.mean()), "mediana": float(med), "sd": float(e.std()),
                "q1": float(q1), "q3": float(q3), "min": float(e.min()), "max": float(e.max())}

    norm = {}
    for m in ("density", "tfidf_length", "lagasio"):
        ez = z(sen) - z(sus[m])
        em = mm(sen) - mm(sus[m])
        norm[m] = {
            "zscore": {**dist(ez), "media_SEN_norm": float(z(sen).mean()),
                       "media_SUS_norm": float(z(sus[m]).mean())},
            "minmax": {**dist(em), "media_SEN_norm": float(mm(sen).mean()),
                       "media_SUS_norm": float(mm(sus[m]).mean())},
            "zscore_vs_minmax": {"pearson": pear(ez, em), "spearman": spear(ez, em),
                                 "deciles_extremos": V3.decile_overlap(ez, em)},
            "minmax_vs_zscore_density": {"pearson": pear(em, idx["density"]),
                                         "spearman": spear(em, idx["density"]),
                                         "deciles_extremos": V3.decile_overlap(em, idx["density"])},
        }
    J["normalisation"] = norm
    # Lagasio (2024), Table 2: reported mean ESGSI (min-max, TF-IDF L2 substance).
    J["normalisation_resumen"] = {
        "lagasio_2024_media_publicada": -0.2179,
        "media_minmax_lagasio_v3": norm["lagasio"]["minmax"]["media"],
        "dif_minmax_lagasio_vs_publicada": norm["lagasio"]["minmax"]["media"] - (-0.2179),
        "media_minmax_por_spec": {m: norm[m]["minmax"]["media"] for m in norm},
        "spearman_zscore_vs_minmax_por_spec": {m: norm[m]["zscore_vs_minmax"]["spearman"] for m in norm},
    }
    J["segundos"] = round(time.time() - t0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(jsonable(J), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"listo: {OUT} ({J['segundos']}s)")


if __name__ == "__main__":
    main()
