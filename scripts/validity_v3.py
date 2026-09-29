"""
validity_v3.py
--------------
Cifras de la seccion de validez convergente del paper (seccion 9) sobre la
especificacion adoptada (vocabulario v3 con sectoriales). Recalcula desde el
texto -- no desde las columnas redondeadas de results.csv -- todo lo que la
seccion cita, y escribe

    results/analysis_v3/validity_v3.json

    PYTHONIOENCODING=utf-8 ESG_RUN_TAG=v3 ESG_INCLUDE_SECTORAL=1 \
        python scripts/validity_v3.py [--n-boot 9999]

Entradas (solo lectura): results/metrics_v3_sec/results.csv,
data/clean_v3_sec/processed_texts.csv (clean_text y raw_text; misma
deduplicacion por MD5 de clean_text que main.py), metadata/ESG_terms_v3.csv y
los lexicos lematizados via src/config.py.

Bloques
  0  Reproduccion: las tres especificaciones del SUS, BREADTH y QUANT se
     recalculan con src/esgsi_analyzer.ESGSIAnalyzer y se comparan con
     results.csv (dif. max). QUANT en results.csv tiene 4 decimales, que es
     demasiado grueso para correlacionar (valores ~0.005): aqui se usa el
     recalculo a precision completa.
  a  Correlaciones (Pearson y Spearman) de cada especificacion del SUS, de
     BREADTH y de las dos variantes con TF sublineal con QUANT y con cada
     familia de QUANT (hits de la familia / R_d).
  b  Solapamiento vocabulario-QUANT: que entradas lematizadas del vocabulario
     de puntuacion coinciden con un patron QUANT, derivado aplicando cada
     patron (re.search, sin distinguir mayusculas) a la entrada lematizada y a
     cada termino natural de ESG_terms_v3.csv que lematiza en ella. Su cuota de
     las ocurrencias del SUS (total y por ano). En el lado de QUANT: cuota de
     cada familia en los hits por ano, y cuota de hits (de cualquier familia)
     cuya cadena casada se compone solo de palabras de entradas compartidas
     (p. ej. 'taxonomy', 'co2', '14001' de 'iso 14001').
  c  Bases disjuntas: se quitan del vocabulario las entradas compartidas, QUANT
     se restringe a percentages + large_numbers, y se recalculan las
     correlaciones y el cociente (especificaciones normalizadas por longitud) /
     (especificacion L2).
  d  Incertidumbre agrupada por empresa: bootstrap de empresas (se remuestrean
     las 49 empresas con reemplazo, con sus 7 informes), IC percentil 95 % de
     cada correlacion con QUANT y de las diferencias entre especificaciones, en
     bases completas y disjuntas. Ademas correlaciones entre empresas (medias
     de empresa, n = 49) y dentro de empresa (desviaciones de la media de
     empresa), y correlacion parcial controlando log L_d y log R_d.
  e  TF sublineal (sublinear_tf, fuente unica; critique_v3.py la importa):
     sum_{c>0}(1 + ln c), sin idf ni L2, dividido por L_d y como media sobre el
     vocabulario; terminos distintos por documento; signo de la correlacion con
     el numero de terminos distintos antes y despues de dividir por longitud.
  f  Indice con cada especificacion frente al adoptado (SUS por densidad):
     Pearson r, Spearman rho y solapamiento de los deciles extremos (10 %
     superior e inferior). Medidas de score; ninguna etiqueta ni umbral.

Salida sin nombres de empresa: la deduplicacion y los textos reparados se
describen por recuento y ano.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ["ESG_RUN_TAG"] = "v3"
os.environ["ESG_INCLUDE_SECTORAL"] = "1"

import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR / "scripts"))

import analysis_v3 as A  # noqa: E402

SPECS = ["density", "tfidf_length", "lagasio"]
LN_SPECS = ["density", "tfidf_length"]          # normalizadas por longitud
FAMILIES = ["percentages", "large_numbers", "units", "frameworks"]
DISJOINT_FAMILIES = ["percentages", "large_numbers"]


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------

def load_corpus_raw(processed: Path, chunks: Path) -> tuple[pd.DataFrame, dict]:
    """
    clean_text con la deduplicacion de main.py (MD5 de clean_text) y raw_text
    reconstruido desde los JSON de extraccion exactamente como main.py
    (re.sub(r"\s+", " ", relevant_text).strip()).

    El raw_text de processed_texts.csv NO sirve: algunos textos contienen el
    caracter NUL y el lector C de pandas trunca el campo en el (p. ej. UniCredit
    2023 queda en 99.091 de 795.471 caracteres). main.py calcula QUANT en
    memoria, antes de escribir el CSV, asi que results.csv es correcto; la
    lectura del CSV no lo es.
    """
    df = pd.read_csv(processed, sep=";", usecols=A.KEY + ["clean_text", "raw_text"],
                     encoding="utf-8")
    df = A.norm_keys(df)
    digest = df["clean_text"].astype(str).map(lambda t: hashlib.md5(t.encode("utf-8")).hexdigest())
    dup = digest.duplicated()
    info = {"filas": len(df), "unicos": int((~dup).sum()),
            "descartados": int(dup.sum())}
    df = df[~dup].reset_index(drop=True)
    files = {(A.nfc(f.parent.parent.name), A.nfc(f.parent.name), A.nfc(f.name)): f
             for f in chunks.rglob("*.json")}
    raw, truncated = [], []
    for r in df.itertuples(index=False):
        f = files[(r[1], r[2], r[0])]  # KEY = Documento, Pais, Compania, Ano
        with open(f, encoding="utf-8") as fh:
            t = re.sub(r"\s+", " ", json.load(fh)["relevant_text"]).strip()
        if t != str(r.raw_text):
            truncated.append({"ano": r[3], "caracteres_json": len(t),
                              "caracteres_csv": len(str(r.raw_text))})
        raw.append(t)
    df["raw_text"] = raw
    info["raw_text_csv_distinto_del_json"] = truncated
    return df, info


def pear(x, y) -> float:
    return float(np.corrcoef(np.asarray(x, float), np.asarray(y, float))[0, 1])


def spear(x, y) -> float:
    return float(stats.spearmanr(x, y)[0])


def decile_overlap(a, b, q: float = 0.10) -> dict:
    """
    Solapamiento de los deciles extremos de dos scores: fraccion de los k
    documentos con mayor (menor) valor de `a` que estan tambien entre los k
    mayores (menores) de `b`, con k = round(q * n). Usa
    analysis_v3.decile_overlap si existe (fuente unica); si no, este equivalente.
    """
    if hasattr(A, "decile_overlap"):
        return A.jsonable(A.decile_overlap(a, b, q=q))
    a, b = np.asarray(a, float), np.asarray(b, float)
    k = max(int(round(q * len(a))), 1)
    oa, ob = np.argsort(a, kind="stable"), np.argsort(b, kind="stable")
    top = len(set(oa[-k:]) & set(ob[-k:])) / k
    bot = len(set(oa[:k]) & set(ob[:k])) / k
    return {"q": q, "k": k, "superior": float(top), "inferior": float(bot)}


def sublinear_tf(C: np.ndarray, L: np.ndarray) -> dict[str, np.ndarray]:
    """
    TF sublineal del paper (ecuacion eq:sublinear, seccion 9): el analogo de
    SUS_density con el conteo c_dj sustituido por 1 + ln c_dj (c_dj > 0), SIN
    idf y SIN normalizacion L2.

        numerador_d = sum_{j: c_dj > 0} (1 + ln c_dj)
        longitud_d  = 100 * numerador_d / L_d      (SUS^sub, eq:sublinear)
        media_V_d   = numerador_d / V              (media sobre el vocabulario)

    Fuente unica: scripts/critique_v3.py la importa de aqui.
    """
    V = C.shape[1]
    logC = np.where(C > 0, 1.0 + np.log(np.where(C > 0, C, 1.0)), 0.0)
    num = logC.sum(axis=1)
    return {"numerador": num, "longitud": 100.0 * num / L, "media_V": num / V}


def sublinear_diagnostics(C: np.ndarray, L: np.ndarray, R: np.ndarray,
                          quant: np.ndarray) -> dict:
    """Diagnosticos del TF sublineal citados en el paper (bloque e)."""
    s = sublinear_tf(C, L)
    sub_num, sub_len, sub_mean = s["numerador"], s["longitud"], s["media_V"]
    distinct = (C > 0).sum(axis=1).astype(float)
    return {
        "definicion": "sum_{c_dj>0}(1 + ln c_dj), sin idf ni L2; longitud: *100/L_d; media_V: /V",
        "fuente_QUANT": "recalculo a precision completa sobre raw_text de los JSON de extraccion",
        "terminos_distintos_media": float(distinct.mean()), "V": int(C.shape[1]),
        "terminos_distintos_min_max": [float(distinct.min()), float(distinct.max())],
        "r_numerador_vs_distintos": pear(sub_num, distinct),
        "r_sub_longitud_vs_distintos": pear(sub_len, distinct),
        "r_sub_media_vs_distintos": pear(sub_mean, distinct),
        "r_sub_longitud_vs_QUANT": pear(sub_len, quant),
        "rho_sub_longitud_vs_QUANT": spear(sub_len, quant),
        "r_sub_media_vs_QUANT": pear(sub_mean, quant),
        "rho_sub_media_vs_QUANT": spear(sub_mean, quant),
        "r_sub_longitud_vs_inversa_L": pear(sub_len, 1.0 / L),
        "r_QUANT_vs_inversa_R": pear(quant, 1.0 / R),
        "r_logL_logR": pear(np.log(L), np.log(R)),
        "cv_numerador": float(sub_num.std() / sub_num.mean()),
        "cv_L": float(L.std() / L.mean()),
    }


def family_hits(raw_texts: list[str], patterns: dict[str, str]) -> pd.DataFrame:
    comp = {k: re.compile(v, re.IGNORECASE) for k, v in patterns.items()}
    rows = []
    for t in raw_texts:
        rows.append({k: len(p.findall(t)) for k, p in comp.items()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Bootstrap de empresas
# ---------------------------------------------------------------------------

def rowcorr(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Correlacion de Pearson fila a fila de dos matrices (B, n)."""
    Xc = X - X.mean(axis=1, keepdims=True)
    Yc = Y - Y.mean(axis=1, keepdims=True)
    return (Xc * Yc).sum(axis=1) / np.sqrt((Xc ** 2).sum(axis=1) * (Yc ** 2).sum(axis=1))


def firm_bootstrap(series: dict[str, np.ndarray], target: np.ndarray, firms: np.ndarray,
                   n_boot: int, seed: int, spearman: bool = False) -> dict[str, np.ndarray]:
    """Remuestrea empresas con reemplazo; devuelve r* por serie (vector de B)."""
    codes, uniq = pd.factorize(firms)
    members = [np.where(codes == g)[0] for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    idx = np.stack([np.concatenate([members[g] for g in row]) for row in draws])
    Y = target[idx]
    if spearman:
        Y = stats.rankdata(Y, axis=1)
    out = {}
    for k, v in series.items():
        X = v[idx]
        if spearman:
            X = stats.rankdata(X, axis=1)
        out[k] = rowcorr(X, Y)
    return out


def ci(v: np.ndarray) -> dict:
    lo, hi = np.percentile(v, [2.5, 97.5])
    return {"ic95": [float(lo), float(hi)], "se_boot": float(np.std(v, ddof=1)),
            "media_boot": float(np.mean(v))}


def boot_summary(point: dict[str, float], boot: dict[str, np.ndarray],
                 diffs: list[tuple[str, str]]) -> dict:
    res = {k: {"r": point[k], **ci(boot[k])} for k in boot}
    for a, b in diffs:
        d = boot[a] - boot[b]
        res[f"{a}_menos_{b}"] = {"dif": point[a] - point[b], **ci(d),
                                 "cuota_boot_dif_le_0": float((d <= 0).mean())}
        q = boot[a] / boot[b]
        res[f"{a}_entre_{b}"] = {"cociente": point[a] / point[b],
                                 "ic95": [float(x) for x in np.percentile(q, [2.5, 97.5])]}
    return res


# ---------------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=A.RESULTS_DIR / "metrics_v3_sec" / "results.csv")
    ap.add_argument("--processed", type=Path, default=A.DATA_DIR / "clean_v3_sec" / "processed_texts.csv")
    ap.add_argument("--chunks", type=Path, default=A.DATA_DIR / "chunks_lexical_v3_sec")
    ap.add_argument("--sectors", type=Path, default=A.METADATA_DIR / "empresas_supersector.csv")
    ap.add_argument("--terms-v3", type=Path, default=A.METADATA_DIR / "ESG_terms_v3.csv")
    ap.add_argument("--out", type=Path, default=A.RESULTS_DIR / "analysis_v3" / "validity_v3.json")
    ap.add_argument("--n-boot", type=int, default=A.N_BOOT)
    a = ap.parse_args()
    t0 = time.time()

    import config
    from esgsi_analyzer import ESGSIAnalyzer

    S: dict = {"generado": time.strftime("%Y-%m-%d %H:%M:%S"),
               "entradas": {k: str(v) for k, v in vars(a).items()},
               "bootstrap": {"tipo": "remuestreo de empresas con reemplazo (7 informes cada una)",
                             "reps": a.n_boot, "semilla": A.SEED, "ic": "percentil 95 %"}}

    df = A.load_results(a.results, a.sectors)
    corpus, dinfo = load_corpus_raw(a.processed, a.chunks)
    df = df.merge(corpus, on=A.KEY, how="left", validate="one_to_one")
    if df["clean_text"].isna().any() or df["raw_text"].isna().any():
        raise ValueError("filas de results sin texto")
    n = len(df)
    firms = df["firm"].to_numpy()
    S["corpus"] = {"documentos": n, "empresas": int(df["firm"].nunique()), "deduplicacion": dinfo}
    clean = df["clean_text"].astype(str).tolist()
    raw = df["raw_text"].astype(str).tolist()

    vocab, pillar, sectoral, lem, vinfo, _ = A.load_vocab_and_pillars(a.terms_v3)
    V = len(vocab)

    # --- 0. Reproduccion -------------------------------------------------
    an = ESGSIAnalyzer(vocab, config.HEDGE_KEYWORDS, config.QUANT_PATTERNS)
    sus = an.calculate_sus_variants(clean)
    breadth = an.calculate_breadth_scores(clean)
    quant = an.calculate_quant_scores(raw)
    col = {"density": "SUS_density", "tfidf_length": "SUS_tfidf_length", "lagasio": "SUS_lagasio"}
    repro = {k: float(np.abs(sus[k] - df[col[k]].to_numpy(float)).max()) for k in SPECS}
    repro["Breadth"] = float(np.abs(breadth - df["Breadth"].to_numpy(float)).max())
    repro["QUANT"] = float(np.abs(quant - df["QUANT_Score"].to_numpy(float)).max())
    S["reproduccion_dif_max"] = repro
    S["vocabulario_V"] = V
    if max(repro[k] for k in SPECS) > 1e-5 or repro["QUANT"] > 5.1e-5 or repro["Breadth"] > 1e-3:
        raise RuntimeError(f"no se reproduce results.csv: {repro}")

    R = np.array([max(len(t.split()), 1) for t in raw], dtype=float)
    L = np.array([max(len(t.split()), 1) for t in clean], dtype=float)
    hits = family_hits(raw, config.QUANT_PATTERNS)
    qfam = {f: hits[f].to_numpy(float) / R for f in FAMILIES}
    qfam["restricted"] = hits[DISJOINT_FAMILIES].sum(axis=1).to_numpy(float) / R
    assert np.allclose(hits[FAMILIES].sum(axis=1).to_numpy(float) / R, quant)

    # --- e. TF sublineal (se usa tambien en a) ---------------------------
    C, cols = A.count_matrix(clean, vocab)
    assert cols == vocab
    sub = sublinear_tf(C, L)
    sub_len, sub_mean = sub["longitud"], sub["media_V"]
    S["sublineal"] = sublinear_diagnostics(C, L, R, quant)

    # --- a. Correlaciones con QUANT y familias ---------------------------
    series = {**sus, "breadth": breadth, "sublinear_length": sub_len, "sublinear_mean": sub_mean}
    targets = {"QUANT": quant, **{f"QUANT_{k}": v for k, v in qfam.items()}}
    corr = {}
    for s, x in series.items():
        corr[s] = {t: {"pearson": pear(x, y), "spearman": spear(x, y)} for t, y in targets.items()}
    S["a_correlaciones"] = corr
    S["a_QUANT_restringido_vs_completo"] = {"pearson": pear(qfam["restricted"], quant),
                                             "spearman": spear(qfam["restricted"], quant)}
    S["a_cocientes_QUANT"] = {k: corr[k]["QUANT"]["pearson"] / corr["lagasio"]["QUANT"]["pearson"]
                              for k in LN_SPECS}
    S["a_cocientes_QUANT_spearman"] = {k: corr[k]["QUANT"]["spearman"] / corr["lagasio"]["QUANT"]["spearman"]
                                       for k in LN_SPECS}

    # Correlaciones de cada especificacion con BREADTH (entradas distintas / V)
    S["a_correlaciones_BREADTH"] = {
        k: {"pearson": pear(v, breadth), "spearman": spear(v, breadth)}
        for k, v in series.items() if k != "breadth"}

    # --- b. Solapamiento --------------------------------------------------
    qcomp = {k: re.compile(v, re.IGNORECASE) for k, v in config.QUANT_PATTERNS.items()}
    lem_in = lem[lem["en_vocabulario"]]
    shared = []
    for j, e in enumerate(vocab):
        naturals = sorted(set(lem_in.loc[lem_in["lema"] == e, "termino"].str.lower()))
        fam_lemma = [f for f, p in qcomp.items() if p.search(e)]
        fam_nat = sorted({f for t in naturals for f, p in qcomp.items() if p.search(t)})
        if fam_lemma or fam_nat:
            shared.append({"entrada": e, "terminos_naturales": naturals,
                           "familias_por_lema": fam_lemma, "familias_por_forma_natural": fam_nat,
                           "sectorial": e in sectoral, "pilar": pillar.get(e),
                           "ocurrencias": int(C[:, j].sum()),
                           "documentos": int((C[:, j] > 0).sum())})
    shared_idx = [vocab.index(s["entrada"]) for s in shared]
    tot = C.sum()
    sh_occ = C[:, shared_idx].sum()
    years = df["year"].to_numpy()
    by_year_sus = {}
    for y in sorted(set(years)):
        m = years == y
        by_year_sus[int(y)] = float(C[m][:, shared_idx].sum() / C[m].sum())
    S["b_solapamiento"] = {
        "criterio": "patron QUANT (re.search, IGNORECASE) sobre la entrada lematizada o sobre "
                    "algun termino natural de ESG_terms_v3.csv que lematiza en ella",
        "entradas_compartidas": shared,
        "n_compartidas": len(shared), "V": V,
        "n_por_lema": sum(bool(s["familias_por_lema"]) for s in shared),
        "solo_por_forma_natural": [s["entrada"] for s in shared if not s["familias_por_lema"]],
        "ocurrencias_compartidas": int(sh_occ), "ocurrencias_totales": int(tot),
        "cuota_ocurrencias_SUS": float(sh_occ / tot),
        "cuota_ocurrencias_SUS_por_ano": by_year_sus,
    }

    # Lado QUANT: cuota de familias por ano y cuota de hits que son palabras de entradas compartidas
    shared_words = set()
    for s in shared:
        for t in [s["entrada"]] + s["terminos_naturales"]:
            shared_words.update(re.findall(r"[a-z0-9]+", t.lower()))
    fam_tot = hits[FAMILIES].sum()
    all_hits = float(fam_tot.sum())
    sh_hits = np.zeros(n)
    sh_strings: dict[str, int] = {}
    sh_fam = {f: 0 for f in FAMILIES}
    for i, t in enumerate(raw):
        for f in FAMILIES:
            for mt in qcomp[f].finditer(t):
                w = mt.group(0).lower()
                ws = re.findall(r"[a-z0-9]+", w)
                if ws and all(x in shared_words for x in ws):
                    sh_hits[i] += 1
                    sh_fam[f] += 1
                    sh_strings[w] = sh_strings.get(w, 0) + 1
    fam_year = {}
    for y in sorted(set(years)):
        m = years == y
        hy = hits.loc[m, FAMILIES].sum()
        fam_year[int(y)] = {**{f: float(hy[f] / hy.sum()) for f in FAMILIES},
                            "cadenas_compartidas": float(sh_hits[m].sum() / hy.sum()),
                            "hits_totales": int(hy.sum())}
    S["b_lado_QUANT"] = {
        "hits_por_familia": {f: int(fam_tot[f]) for f in FAMILIES},
        "hits_totales": int(all_hits),
        "cuota_por_familia": {f: float(fam_tot[f] / all_hits) for f in FAMILIES},
        "cuota_por_familia_por_ano": fam_year,
        "hits_cadena_compartida": int(sh_hits.sum()),
        "hits_cadena_compartida_por_familia": sh_fam,
        "cuota_hits_cadena_compartida": float(sh_hits.sum() / all_hits),
        "cadenas_compartidas_casadas": dict(sorted(sh_strings.items(), key=lambda kv: -kv[1])),
        "palabras_de_entradas_compartidas": sorted(shared_words),
        "documentos_QUANT_cero": int((hits[FAMILIES].sum(axis=1) == 0).sum()),
    }

    # --- c. Bases disjuntas ----------------------------------------------
    red_vocab = [t for t in vocab if t not in {s["entrada"] for s in shared}]
    an_red = ESGSIAnalyzer(red_vocab, config.HEDGE_KEYWORDS, config.QUANT_PATTERNS)
    sus_red = an_red.calculate_sus_variants(clean)
    qr = qfam["restricted"]
    dis = {k: {"pearson": pear(sus_red[k], qr), "spearman": spear(sus_red[k], qr)} for k in SPECS}
    S["c_bases_disjuntas"] = {
        "V_reducido": len(red_vocab), "V": V,
        "familias_QUANT": DISJOINT_FAMILIES,
        "correlaciones": dis,
        "correlaciones_bases_completas": {k: corr[k]["QUANT"] for k in SPECS},
        "cocientes_pearson": {k: dis[k]["pearson"] / dis["lagasio"]["pearson"] for k in LN_SPECS},
        "cocientes_spearman": {k: dis[k]["spearman"] / dis["lagasio"]["spearman"] for k in LN_SPECS},
        "caida_relativa_pearson": {k: 1 - dis[k]["pearson"] / corr[k]["QUANT"]["pearson"] for k in SPECS},
        "r_sus_reducido_vs_completo": {k: pear(sus_red[k], sus[k]) for k in SPECS},
        # descomposicion: solo vocabulario reducido / solo QUANT restringido
        "solo_vocabulario_reducido": {k: pear(sus_red[k], quant) for k in SPECS},
        "solo_QUANT_restringido": {k: pear(sus[k], qr) for k in SPECS},
    }

    # --- d. Incertidumbre agrupada por empresa ----------------------------
    diffs = [("density", "lagasio"), ("tfidf_length", "lagasio"), ("tfidf_length", "density")]
    boot_full = firm_bootstrap(sus, quant, firms, a.n_boot, A.SEED)
    boot_dis = firm_bootstrap(sus_red, qr, firms, a.n_boot, A.SEED)
    boot_full_s = firm_bootstrap(sus, quant, firms, a.n_boot, A.SEED, spearman=True)
    boot_dis_s = firm_bootstrap(sus_red, qr, firms, a.n_boot, A.SEED, spearman=True)
    S["d_bootstrap_empresas"] = {
        "pearson_bases_completas": boot_summary({k: corr[k]["QUANT"]["pearson"] for k in SPECS}, boot_full, diffs),
        "pearson_bases_disjuntas": boot_summary({k: dis[k]["pearson"] for k in SPECS}, boot_dis, diffs),
        "spearman_bases_completas": boot_summary({k: corr[k]["QUANT"]["spearman"] for k in SPECS}, boot_full_s, diffs),
        "spearman_bases_disjuntas": boot_summary({k: dis[k]["spearman"] for k in SPECS}, boot_dis_s, diffs),
    }

    g = pd.Series(firms)
    def between(x): return pd.Series(x).groupby(g).mean().to_numpy()
    def within_(x): return A.within(pd.Series(x), g)
    lvl = {}
    for base, ss, q in (("completas", sus, quant), ("disjuntas", sus_red, qr)):
        lvl[base] = {k: {"entre_empresas_r": pear(between(ss[k]), between(q)),
                         "entre_empresas_rho": spear(between(ss[k]), between(q)),
                         "dentro_empresa_r": pear(within_(ss[k]), within_(q))} for k in SPECS}
    S["d_entre_y_dentro"] = {"n_empresas": int(g.nunique()), **lvl}

    Xc = np.column_stack([np.ones(n), np.log(L), np.log(R)])
    def resid(v):
        beta, *_ = np.linalg.lstsq(Xc, v, rcond=None)
        return v - Xc @ beta
    S["d_parcial_logL_logR"] = {
        "completas": {k: pear(resid(sus[k]), resid(quant)) for k in SPECS},
        "disjuntas": {k: pear(resid(sus_red[k]), resid(qr)) for k in SPECS},
        "sublinear_length": pear(resid(sub_len), resid(quant)),
    }

    # --- f. Indice por especificacion --------------------------------------
    # Desde las columnas de results.csv (el ESGSI adoptado es la columna del
    # pipeline). Solo medidas de score: r, rho y deciles extremos.
    zsen = A.z(df["SEN_Score"].to_numpy(float))
    ref = df["ESGSI"].to_numpy(float)
    idx = {"density": {"pearson": 1.0, "spearman": 1.0, "n": n,
                       "deciles_extremos": decile_overlap(ref, ref)}}
    for k in ("tfidf_length", "lagasio"):
        v = zsen - A.z(df[col[k]].to_numpy(float))
        idx[k] = {"pearson": pear(v, ref), "spearman": spear(v, ref), "n": n,
                  "deciles_extremos": decile_overlap(v, ref)}
    S["f_indice_por_especificacion"] = idx
    pairs = {"density_tfidf_length": ("SUS_density", "SUS_tfidf_length"),
             "density_lagasio": ("SUS_density", "SUS_lagasio"),
             "tfidf_length_lagasio": ("SUS_tfidf_length", "SUS_lagasio"),
             "lagasio_breadth": ("SUS_lagasio", "Breadth")}
    S["f_SUS_entre_especificaciones_results_csv"] = {
        k: {"pearson": pear(df[x], df[y]), "spearman": spear(df[x], df[y])}
        for k, (x, y) in pairs.items()}

    S["segundos"] = round(time.time() - t0)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(A.jsonable(S), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"listo: {a.out} ({S['segundos']}s)")


if __name__ == "__main__":
    main()
