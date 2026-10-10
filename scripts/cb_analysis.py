"""
cb_analysis.py
--------------
Paso 3: agrega las predicciones de ClimateBERT por informe y las compara con
los scores del paper (issue #4: validez convergente y discriminante con las
medidas corregidas: SEN por forma exacta, QUANT solo cifras, HEDGE restringido).

Medidas por informe (sobre los pasajes de cb_segment.py):
    clim_share   pasajes climaticos / pasajes (detector, p > 0.5)
    CTI          indice de cheap talk de Bingler et al. (2022, 2024): pasajes
                 climaticos de compromiso no especificos / pasajes climaticos
                 de compromiso
    CTI_prob     lo mismo con probabilidades en vez de clases (sensibilidad)
    spec_share   pasajes climaticos especificos / pasajes climaticos
    sent_net     (oportunidad - riesgo) / pasajes climaticos
    commit_share pasajes de compromiso / pasajes climaticos
Componentes del paper recalculados por alcance (mismas funciones que el
pipeline, src/esgsi_analyzer.ESGSIAnalyzer, sobre el texto crudo):
    SEN_clim, QUANT_clim, HEDGE_clim  solo pasajes climaticos
    SEN_nonclim                       pasajes no climaticos
    SUS_clim, SUS_nonclim             (cb_sus_scope.py, texto lematizado)
SEN sin el vocabulario de temas ESRS (seccion 'ruptura de 2024'):
    SEN_exA   sin la lista A (impactos negativos, incidentes, quejas y
              cauces de denuncia, acoso, trabajo forzoso, corrupcion, soborno,
              violaciones: S1-S4, G1 y la terminologia de impactos de ESRS 1/2)
    SEN_exB   lista A mas salud y seguridad (S1-14) y sanciones (G1-4)
    SEN_exTop10, SEN_exTop25  sin las 10/25 palabras negativas de L&M que mas
              crecen de 2023 a 2024 (cota superior, elegida con los datos)

Comparaciones (solo scores continuos; directrices del autor):
    - Pearson y Spearman: agregado, intra-empresa, entre empresas (49 medias)
      y doble EF; IC 95 % por bootstrap de empresas (B = 2000)
    - matriz multirrasgo-multimetodo (Campbell y Fiske, 1959)
    - ruptura de 2024: medias anuales (IC agrupados por empresa) y
      contraste 2024 frente a 2018-2023 con EF de empresa (con y sin tendencia
      lineal), cluster por empresa y wild cluster bootstrap
    - pendientes con EF de empresa (trend_block de analysis_v3)
    - convergencia a nivel de pasaje (dentro de cada informe)
    - solapamiento de deciles extremos ESGSI / CTI

    .venv-cb/Scripts/python.exe scripts/cb_analysis.py --export-scope   (texto para cb_sus_scope.py)
    .venv-cb/Scripts/python.exe scripts/cb_analysis.py

Salidas versionables (solo identificadores anonimos) en results/climatebert/.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from cb_common import BASE_DIR, CACHE_DIR, OUT_DIR, iter_raw, load_scores, text_hash

sys.path.insert(0, str(BASE_DIR / "scripts"))
sys.path.insert(0, str(BASE_DIR / "src"))
import analysis_v3 as av  # noqa: E402  (fija ESG_RUN_TAG=v3 y sectoriales)
import config  # noqa: E402
from esgsi_analyzer import ESGSIAnalyzer, NEGATIONS, WORD_RE  # noqa: E402

SEED = 20261010
B = int(os.environ.get("ESG_CB_BOOT", 2000))
CUT = 0.5
PILOT_CACHE = CACHE_DIR / "pilot_cache"

CB_VARS = ["sent_net", "spec_share", "clim_share", "CTI", "CTI_prob", "commit_share"]
OUR_VARS = ["Z_SEN", "Z_SUS", "Z_QUANT", "Z_HEDGE", "ESGSI", "ESGSI_ext"]
SCOPE_VARS = ["SEN_clim", "SEN_nonclim", "SUS_clim", "SUS_nonclim", "QUANT_clim", "HEDGE_clim"]

# --- Vocabulario de temas ESRS dentro de la lista Negative de L&M ----------
# Lista A (a priori, por el texto de las normas): palabras de L&M que nombran
# un tema o un dato que las ESRS obligan a divulgar, no una valoracion del
# desempeno. Impactos negativos (ESRS 1, 3.4; ESRS 2 SBM-3, IRO-1: "negative /
# adverse impacts"); S1-S4: incidentes, quejas y cauces para plantear
# inquietudes (S1-3, S1-17: "incidents, complaints", "grievance mechanisms",
# "raise concerns"), acoso (S1-17), trabajo forzoso (S1-1, S2), violaciones de
# derechos humanos (S1-17); G1: corrupcion y soborno (G1-3, G1-4) e inquietudes
# de denunciantes (G1-1).
ESRS_A = {"negative", "negatively", "adverse", "adversely", "incident", "incidents",
          "complaint", "complaints", "grievance", "grievances", "concern", "concerns",
          "harassment", "harassed", "forced", "violation", "violations", "violate", "violated",
          "bribery", "bribe", "bribes", "corruption", "corrupt"}
# Lista B: A mas salud y seguridad (S1-14: accidentes, lesiones, muertes) y
# sanciones (G1-4: condenas y multas por corrupcion; E2/S1: incumplimientos).
ESRS_B = ESRS_A | {"injury", "injuries", "fatality", "fatalities", "fatal", "accident", "accidents",
                   "convicted", "conviction", "convictions", "fines", "penalties", "penalty",
                   "breach", "breaches", "infringement", "infringements", "misconduct", "fraud",
                   "abuse", "abuses", "exploitation", "severe", "severity"}


# Pasajes con terminologia del marco ESRS (a priori, por el texto de las
# normas): la referencia a las propias normas, la doble materialidad y la
# formula "impactos, riesgos y oportunidades" (IRO) con la que ESRS 1 y ESRS 2
# ordenan la divulgacion. Sirve para ver si el sentimiento de ClimateBERT cae
# en 2024 tambien fuera de ese texto prescrito.
FRAME_RE = re.compile(r"\bESRS\b|\bIROs?\b|double[- ]materiality|"
                      r"\bimpacts?,? risks? and opportunities\b", re.IGNORECASE)
# Variante amplia: ademas, el vocabulario de riesgo climatico de TCFD/ESRS E1.
FRAME_WIDE_RE = re.compile(FRAME_RE.pattern + r"|\b(?:physical|transition)(?:al)? (?:climate )?risks?\b",
                           re.IGNORECASE)


def analyzer() -> ESGSIAnalyzer:
    from loguru import logger
    logger.remove()
    return ESGSIAnalyzer(keywords=list(config.ESG_KEYWORDS), hedge_words=config.HEDGE_KEYWORDS,
                         quant_patterns=config.QUANT_PATTERNS, ext_weights=config.ESGSI_EXT_WEIGHTS,
                         sus_mode=config.SUS_MODE, positive_words=config.LM_POSITIVE,
                         negative_words=config.LM_NEGATIVE)


AN = analyzer()


def polarity(text: str) -> float:
    p, n = AN.sen_counts(text)
    return (p - n) / (p + n) if p + n else 0.0


def sen_word_counts(text: str) -> tuple[Counter, Counter, int]:
    """Recuento por palabra de positivas (con la regla de negacion) y negativas,
    con la misma logica que ESGSIAnalyzer.sen_counts."""
    toks = WORD_RE.findall(text.lower())
    pos, neg = Counter(), Counter()
    for i, t in enumerate(toks):
        if t in AN.positive_words:
            if not any(w in NEGATIONS for w in toks[max(0, i - 3):i]):
                pos[t] += 1
        elif t in AN.negative_words:
            neg[t] += 1
    return pos, neg, len(toks)


def zpop(x) -> np.ndarray:
    x = np.asarray(x, float)
    return (x - np.nanmean(x)) / np.nanstd(x)


# ---------------------------------------------------------------------------
# Pasajes e informes
# ---------------------------------------------------------------------------

def passage_frame() -> pd.DataFrame:
    P = pd.read_parquet(CACHE_DIR / "passages.parquet")

    def pr(name):
        return pd.read_parquet(CACHE_DIR / f"pred_{name}.parquet")

    det = pr("detector").rename(columns={"p_yes": "p_clim"}).drop(columns="p_no")
    P = P.merge(det, on="h", how="left", validate="many_to_one")
    assert P["p_clim"].notna().all(), "faltan predicciones del detector"
    P["clim"] = P["p_clim"] > CUT
    com = pr("commitment").rename(columns={"p_yes": "p_commit"}).drop(columns="p_no")
    for x in (com, pr("specificity"), pr("sentiment")):
        P = P.merge(x, on="h", how="left", validate="many_to_one")
    C = P[P["clim"]]
    assert C[["p_commit", "p_spec", "p_opportunity"]].notna().all().all(), "faltan predicciones"
    P["commit"] = P["clim"] & (P["p_commit"] > CUT)
    P["spec"] = P["clim"] & (P["p_spec"] > CUT)
    sc = P[["p_opportunity", "p_neutral", "p_risk"]].to_numpy()
    lab = np.array(["opportunity", "neutral", "risk"])
    P["sent"] = np.where(P["clim"], lab[np.nan_to_num(sc, nan=-1).argmax(axis=1)], None)
    P["frame"] = P["text"].str.contains(FRAME_RE)
    P["frame_wide"] = P["text"].str.contains(FRAME_WIDE_RE)
    return P


def joined(texts) -> str:
    return " ".join(texts)


def doc_frame(P: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k, g in P.groupby("doc_key"):
        c = g[g["clim"]]
        nc = g[~g["clim"]]
        cm = c[c["commit"]]
        n_c = len(c)
        r = {"doc_key": k, "n_passages": len(g), "n_climate": n_c, "n_commit": len(cm),
             "clim_share": n_c / len(g),
             "clim_share_words": c["n_words"].sum() / g["n_words"].sum(),
             "commit_share": len(cm) / n_c if n_c else np.nan,
             "CTI": (~cm["spec"]).mean() if len(cm) else np.nan,
             "CTI_prob": ((c["p_commit"] * c["p_non"]).sum() / c["p_commit"].sum()) if n_c else np.nan,
             "spec_share": c["spec"].mean() if n_c else np.nan,
             "sent_net": ((c["sent"] == "opportunity").sum() - (c["sent"] == "risk").sum()) / n_c
             if n_c else np.nan,
             "sent_net_prob": (c["p_opportunity"] - c["p_risk"]).mean() if n_c else np.nan,
             "opp_share": (c["sent"] == "opportunity").mean() if n_c else np.nan,
             "risk_share": (c["sent"] == "risk").mean() if n_c else np.nan,
             "frame_share_clim": c["frame"].mean() if n_c else np.nan,
             "frame_wide_share_clim": c["frame_wide"].mean() if n_c else np.nan}
        for nm, col in [("exFrame", "frame"), ("exFrameWide", "frame_wide")]:
            cc = c[~c[col]]
            r[f"sent_net_{nm}"] = (((cc["sent"] == "opportunity").sum() - (cc["sent"] == "risk").sum())
                                   / len(cc)) if len(cc) else np.nan
        tc, tn = joined(c["text"]), joined(nc["text"])
        r["SEN_clim"] = polarity(tc) if n_c else np.nan
        r["SEN_nonclim"] = polarity(tn)
        r["QUANT_clim"] = AN.quant_hits(tc) / max(len(tc.split()), 1) if n_c else np.nan
        h, n = AN.hedge_counts(tc)
        r["HEDGE_clim"] = h / max(n, 1) if n_c else np.nan
        rows.append(r)
    D = pd.DataFrame(rows)
    keep = ["doc_key", "firm_label", "year", "country", "industry", "SEN_Score", "SUS_density",
            "QUANT_Score", "HEDGE_Score"] + OUR_VARS
    D = scores[keep].merge(D, on="doc_key", how="left", validate="one_to_one")
    assert len(D) == 343
    f = CACHE_DIR / "sus_scope.csv"
    if f.exists():
        S = pd.read_csv(f, sep=";")
        D = D.merge(S, on="doc_key", how="left", validate="one_to_one")
        D.loc[D["n_climate"].fillna(0) == 0, "SUS_clim"] = np.nan
    else:
        print("AVISO: falta data/climatebert/sus_scope.csv (cb_sus_scope.py); sin SUS por alcance")
        for c in ["SUS_clim", "SUS_nonclim", "SUS_passages"]:
            D[c] = np.nan
    D["firm"] = D["firm_label"]  # cluster = empresa (identificador anonimo)
    return D


def export_scope(P: pd.DataFrame):
    out = {}
    for k, g in P.groupby("doc_key"):
        out[k] = {"clim": "\n\n".join(g.loc[g["clim"], "text"]),
                  "nonclim": "\n\n".join(g.loc[~g["clim"], "text"])}
    with gzip.open(CACHE_DIR / "scope_texts.json.gz", "wt", encoding="utf-8") as fh:
        json.dump(out, fh)
    print(f"scope_texts.json.gz: {len(out)} informes")


# ---------------------------------------------------------------------------
# Correlaciones con bootstrap de empresas
# ---------------------------------------------------------------------------

def _rank(a):
    return np.apply_along_axis(stats.rankdata, 0, a)


def _corr(a, b):
    a = a - a.mean(0)
    b = b - b.mean(0)
    return (a.T @ b) / np.sqrt(np.outer((a ** 2).sum(axis=0), (b ** 2).sum(axis=0)))


def _views(X: np.ndarray, firm: np.ndarray, year: np.ndarray) -> dict:
    """X en las cuatro vistas: agregado, intra-empresa, entre empresas, doble EF."""
    fm = pd.DataFrame(X).groupby(firm).transform("mean").to_numpy()
    fc, yc = pd.factorize(firm)[0], pd.factorize(year)[0]
    nf, ny = np.bincount(fc), np.bincount(yc)

    def _dm(A, c, n):
        M = np.stack([np.bincount(c, weights=A[:, j]) for j in range(A.shape[1])], 1) / n[:, None]
        return A - M[c]

    tw = X - X.mean(0)
    for _ in range(30):  # doble EF exacto por proyecciones alternadas (panel no equilibrado)
        tw = _dm(_dm(tw, fc, nf), yc, ny)
    bt = pd.DataFrame(X).groupby(firm).mean().to_numpy()
    return {"pooled": X, "within": X - fm, "between": bt, "twoway": tw}


def corr_all(X, Y, firm, year) -> dict:
    vx, vy = _views(X, firm, year), _views(Y, firm, year)
    out = {}
    for v in vx:
        out[(v, "pearson")] = _corr(vx[v], vy[v])
        out[(v, "spearman")] = _corr(_rank(vx[v]), _rank(vy[v]))
    return out


def boot_corr(D: pd.DataFrame, xs: list[str], ys: list[str]) -> pd.DataFrame:
    D = D.dropna(subset=xs + ys).reset_index(drop=True)
    firms = D["firm"].to_numpy()
    uf = np.unique(firms)
    idx_by_firm = {f: np.where(firms == f)[0] for f in uf}
    year = D["year"].to_numpy()
    X, Y = D[xs].to_numpy(float), D[ys].to_numpy(float)
    est = corr_all(X, Y, firms, year)
    rng = np.random.default_rng(SEED)
    draws = {k: [] for k in est}
    for _ in range(B):
        pick = rng.choice(uf, size=len(uf), replace=True)
        idx = np.concatenate([idx_by_firm[f] for f in pick])
        newfirm = np.repeat(np.arange(len(pick)), [len(idx_by_firm[f]) for f in pick])
        r = corr_all(X[idx], Y[idx], newfirm, year[idx])
        for k in draws:
            draws[k].append(r[k])
    rows = []
    for (v, m), R in est.items():
        A = np.stack(draws[(v, m)])
        lo, hi = np.nanpercentile(A, 2.5, axis=0), np.nanpercentile(A, 97.5, axis=0)
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                rows.append({"cb": x, "score": y, "view": v, "method": m, "r": R[i, j],
                             "ci_lo": lo[i, j], "ci_hi": hi[i, j],
                             "n": len(uf) if v == "between" else len(D)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# MTMM
# ---------------------------------------------------------------------------

MTMM_DICT = ["Z_SEN", "Z_SUS", "Z_QUANT", "Z_HEDGE", "ESGSI", "ESGSI_ext"]
MTMM_CB = ["sent_net", "spec_share", "clim_share", "CTI"]
TRAITS = {"Z_SEN": {"tone"}, "sent_net": {"tone"},
          "Z_SUS": {"substance"}, "Z_QUANT": {"substance"}, "spec_share": {"substance"},
          "clim_share": {"substance"},
          "Z_HEDGE": {"vagueness"}, "CTI": {"vagueness", "composite"},
          "ESGSI": {"composite"}, "ESGSI_ext": {"composite"}}
COMPOSITES = {"ESGSI", "ESGSI_ext"}
# Pares de validez (mismo rasgo, distinto metodo) con el signo esperado.
VALIDITY = [("Z_SEN", "sent_net", +1), ("Z_SUS", "spec_share", +1), ("Z_SUS", "clim_share", +1),
            ("Z_QUANT", "spec_share", +1), ("Z_QUANT", "clim_share", +1),
            ("Z_HEDGE", "CTI", +1), ("ESGSI", "CTI", +1), ("ESGSI_ext", "CTI", +1)]


def mtmm(D: pd.DataFrame, vars_: list[str], view: str, method: str) -> pd.DataFrame:
    D = D.dropna(subset=vars_)
    v = _views(D[vars_].to_numpy(float), D["firm"].to_numpy(), D["year"].to_numpy())[view]
    if method == "spearman":
        v = _rank(v)
    return pd.DataFrame(_corr(v, v), index=vars_, columns=vars_)


def campbell_fiske(M: pd.DataFrame, dict_vars, cb_vars, validity, traits=TRAITS) -> list[dict]:
    """Criterio 1 de Campbell y Fiske: el valor de validez (con su signo esperado)
    debe superar el maximo |r| heterorrasgo-heterometodo de su fila y su columna
    (variables del otro metodo que no comparten rasgo).
    Los indices compuestos contienen a los componentes (ESGSI = Z(SEN) - Z(SUS)),
    asi que no cuentan como 'otro rasgo' al evaluar un par de componentes."""
    out = []
    for a, b, s in validity:
        v = s * M.loc[a, b]
        skip = COMPOSITES if a not in COMPOSITES else set()
        het = [abs(M.loc[a, c]) for c in cb_vars if c != b and not traits[c] & traits[a]] + \
              [abs(M.loc[c, b]) for c in dict_vars if c != a and c not in skip
               and not traits[c] & traits[b]]
        out.append({"par": f"{a}~{b}", "validez": float(v), "max_het_het": float(max(het)),
                    "cumple": bool(v > max(het))})
    return out


# ---------------------------------------------------------------------------
# Ruptura de 2024
# ---------------------------------------------------------------------------

def break_block(df: pd.DataFrame, col: str, n_boot: int) -> dict:
    """2024 frente a 2018-2023 con EF de empresa, en DT agregadas de la medida.
    (a) y_it = a_i + b*1[2024] + e; (b) con tendencia lineal comun:
    y_it = a_i + c*t + b*1[2024] + e (salto sobre la tendencia de 2018-2023).
    Errores agrupados por empresa (t con G-1 gl) y wild cluster bootstrap
    (Rademacher, H0 impuesta; en (b) por Frisch-Waugh-Lovell)."""
    d = df.dropna(subset=[col]).copy()
    d["_y"] = zpop(d[col])
    d24 = (d["year"] == 2024).astype(float)
    g = d["firm"].to_numpy()
    yd = av.within(d["_y"], d["firm"])
    xd = av.within(d24, d["firm"])
    td = av.within(d["year"].astype(float), d["firm"])
    m1 = av.cluster_fit(yd, xd[:, None], g)
    w1 = av.wild_cluster_p(yd, xd, g, n_boot, av.SEED)
    m2 = av.cluster_fit(yd, np.column_stack([td, xd]), g)
    # FWL: residualiza y y la dummy sobre la tendencia (intra-empresa).
    ry = yd - td * (td @ yd) / (td @ td)
    rx = xd - td * (td @ xd) / (td @ td)
    w2 = av.wild_cluster_p(ry, rx, g, n_boot, av.SEED)
    pre = d[d["year"] <= 2023]
    fp = av.fe_slope(pre.assign(_y=zpop(pre[col])), "_y")
    mean_pre = float(d.loc[d["year"] <= 2023, "_y"].mean())
    mean_24 = float(d.loc[d["year"] == 2024, "_y"].mean())
    return {"n": int(len(d)), "media_z_2018_2023": mean_pre, "media_z_2024": mean_24,
            "salto_ef": {"b": float(m1.params[0]), "se": float(m1.bse[0]),
                         "ci95": [float(x) for x in m1.conf_int()[0]], "p": float(m1.pvalues[0]),
                         "p_wild": w1["p_wild"]},
            "salto_sobre_tendencia": {"b": float(m2.params[1]), "se": float(m2.bse[1]),
                                      "ci95": [float(x) for x in m2.conf_int()[1]],
                                      "p": float(m2.pvalues[1]), "p_wild": w2["p_wild"],
                                      "tendencia": float(m2.params[0])},
            "pendiente_2018_2023_z_pre": fp}


def sen_decomposition(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Recuentos L&M por palabra en el texto de cada informe; comprueba que
    reproducen SEN_Score; devuelve variantes de SEN sin listas de palabras."""
    rows, neg_rows, pos_rows = [], {}, {}
    for k, raw in iter_raw(scores):
        pos, neg, n = sen_word_counts(raw)
        p0, n0 = AN.sen_counts(raw)
        assert (sum(pos.values()), sum(neg.values())) == (p0, n0)
        rows.append({"doc_key": k, "tokens": n, "P": p0, "N": n0,
                     "SEN_recomputed": (p0 - n0) / (p0 + n0) if p0 + n0 else 0.0})
        neg_rows[k], pos_rows[k] = neg, pos
    R = pd.DataFrame(rows)
    R = scores[["doc_key", "firm_label", "year", "SEN_Score", "Z_SEN"]].merge(
        R, on="doc_key", how="inner", validate="one_to_one")
    assert len(R) == 343
    chk = {"max_abs_dif_SEN_Score": float((R["SEN_recomputed"] - R["SEN_Score"]).abs().max()),
           "corr_Z_SEN": float(np.corrcoef(zpop(R["SEN_recomputed"]), R["Z_SEN"])[0, 1])}
    NW = pd.DataFrame.from_dict(neg_rows, orient="index").fillna(0)
    PW = pd.DataFrame.from_dict(pos_rows, orient="index").fillna(0)
    NW, PW = NW.loc[R["doc_key"]], PW.loc[R["doc_key"]]
    yr = R["year"].to_numpy()
    tok = R.set_index("doc_key")["tokens"]

    def rate(W, y):
        m = yr == y
        return W[m].sum(axis=0) / tok[m].sum() * 1e4  # por 10.000 palabras

    T = pd.DataFrame({"neg_rate_2018_2023": NW[yr <= 2023].sum(axis=0) / tok[yr <= 2023].sum() * 1e4,
                      "neg_rate_2023": rate(NW, 2023), "neg_rate_2024": rate(NW, 2024)})
    T["cambio_2023_2024"] = T["neg_rate_2024"] - T["neg_rate_2023"]
    T["cambio_2024_vs_2018_2023"] = T["neg_rate_2024"] - T["neg_rate_2018_2023"]
    tot = T["cambio_2023_2024"].sum()
    T["cuota_del_cambio_total"] = T["cambio_2023_2024"] / tot
    T["lista_A"] = T.index.isin(ESRS_A)
    T["lista_B"] = T.index.isin(ESRS_B)
    T = T.sort_values("cambio_2023_2024", ascending=False)
    Tp = pd.DataFrame({"pos_rate_2023": rate(PW, 2023), "pos_rate_2024": rate(PW, 2024)})
    Tp["cambio_2023_2024"] = Tp["pos_rate_2024"] - Tp["pos_rate_2023"]
    Tp = Tp.sort_values("cambio_2023_2024")
    top = list(T.index)
    lists = {"exA": ESRS_A, "exB": ESRS_B, "exTop10": set(top[:10]), "exTop25": set(top[:25])}
    for name, L in lists.items():
        cols = [c for c in NW.columns if c in L]
        n_ex = NW[cols].sum(axis=1).to_numpy()
        P, N = R["P"].to_numpy(), R["N"].to_numpy() - n_ex
        R[f"SEN_{name}"] = np.where(P + N > 0, (P - N) / np.maximum(P + N, 1), 0.0)
        R[f"cuota_N_{name}"] = n_ex / np.maximum(R["N"].to_numpy(), 1)
    info = {"comprobacion": chk,
            "lista_A": sorted(ESRS_A), "lista_B_extra": sorted(ESRS_B - ESRS_A),
            "lista_A_en_LM_negativas": sorted(ESRS_A & AN.negative_words),
            "lista_B_en_LM_negativas": sorted(ESRS_B & AN.negative_words),
            "top10": top[:10], "top25": top[:25],
            "tasa_negativas_por_10k": {int(y): float(NW[yr == y].sum().sum() / tok[yr == y].sum() * 1e4)
                                       for y in sorted(set(yr))},
            "tasa_positivas_por_10k": {int(y): float(PW[yr == y].sum().sum() / tok[yr == y].sum() * 1e4)
                                       for y in sorted(set(yr))},
            "cambio_total_negativas_2023_2024": float(tot),
            "cuota_cambio_lista_A": float(T.loc[T["lista_A"], "cambio_2023_2024"].sum() / tot),
            "cuota_cambio_lista_B": float(T.loc[T["lista_B"], "cambio_2023_2024"].sum() / tot),
            "cuota_N_lista_A_por_ano": R.groupby("year")["cuota_N_exA"].mean().to_dict(),
            "cuota_N_lista_B_por_ano": R.groupby("year")["cuota_N_exB"].mean().to_dict()}
    return R, T, info | {"positivas_que_mas_caen": Tp.head(10)["cambio_2023_2024"].to_dict()}


# ---------------------------------------------------------------------------
# Nivel de pasaje
# ---------------------------------------------------------------------------

def passage_level(P: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    C = P[P["clim"]].copy()
    pc = [AN.sen_counts(t) for t in C["text"]]
    C["lm_pos"] = [a for a, _ in pc]
    C["lm_neg"] = [b for _, b in pc]
    C["lm_pol"] = [(a - b) / (a + b) if a + b else 0.0 for a, b in pc]
    C["quant"] = [AN.quant_hits(t) / max(len(t.split()), 1) for t in C["text"]]
    C["hedge"] = [h / max(n, 1) for h, n in (AN.hedge_counts(t) for t in C["text"])]
    C["cb_net"] = C["p_opportunity"] - C["p_risk"]
    C["year"] = C["year"].astype(int)
    pairs = [("lm_pol", "cb_net"), ("lm_pol", "p_spec"), ("quant", "p_spec"), ("quant", "cb_net"),
             ("hedge", "p_spec"), ("hedge", "cb_net"), ("hedge", "p_commit")]

    def within_rho(G, a, b):
        return G.groupby("doc_key").apply(
            lambda g: stats.spearmanr(g[a], g[b])[0] if len(g) >= 10 and g[a].nunique() > 1 else np.nan,
            include_groups=False).dropna()

    res = {}
    for a, b in pairs:
        rr = within_rho(C, a, b)
        res[f"{a}~{b}"] = {"informes": int(len(rr)), "mediana_rho": float(rr.median()),
                           "p25": float(rr.quantile(.25)), "p75": float(rr.quantile(.75)),
                           "cuota_positiva": float((rr > 0).mean()),
                           "rho_agregado_todos_los_pasajes": float(stats.spearmanr(C[a], C[b])[0]),
                           "pasajes": int(len(C))}
    # Solo pasajes con al menos una palabra de tono de L&M.
    Ct = C[(C["lm_pos"] + C["lm_neg"]) > 0]
    rr = within_rho(Ct, "lm_pol", "cb_net")
    res["lm_pol~cb_net|con_tono"] = {"informes": int(len(rr)), "mediana_rho": float(rr.median()),
                                     "p25": float(rr.quantile(.25)), "p75": float(rr.quantile(.75)),
                                     "cuota_positiva": float((rr > 0).mean()),
                                     "pasajes": int(len(Ct)),
                                     "cuota_pasajes_con_tono": float(len(Ct) / len(C))}
    # Por ano (mediana de la rho intra-informe).
    rr = within_rho(C, "lm_pol", "cb_net").rename("rho").reset_index()
    rr = rr.merge(C[["doc_key", "year"]].drop_duplicates(), on="doc_key")
    res["lm_pol~cb_net_por_ano"] = rr.groupby("year")["rho"].median().to_dict()
    return res, C


# ---------------------------------------------------------------------------

def coverage_vs_pilot(P: pd.DataFrame, D: pd.DataFrame) -> dict:
    if not (PILOT_CACHE / "passages.parquet").exists():
        return {}
    old = pd.read_parquet(PILOT_CACHE / "passages.parquet", columns=["doc_key", "text"])
    oldh = set(old["text"].map(text_hash))
    P = P.assign(nuevo=~P["h"].isin(oldh))
    by = P.groupby("doc_key")["nuevo"].mean()
    lab = D.set_index("doc_key")[["firm_label", "year"]]
    changed = by[by > 0].sort_values(ascending=False)
    return {"pasajes_piloto": int(len(old)), "pasajes_ahora": int(len(P)),
            "pasajes_con_texto_nuevo": int(P["nuevo"].sum()),
            "informes_con_algun_pasaje_nuevo": int(len(changed)),
            "informes_con_mas_del_50pct_nuevo": [
                {"firm_label": lab.loc[k, "firm_label"], "year": int(lab.loc[k, "year"]),
                 "cuota_nueva": float(v)} for k, v in changed[changed > .5].items()],
            "cuota_nueva_mediana_en_los_demas": float(changed[changed <= .5].median())
            if (changed <= .5).any() else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-scope", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scores = load_scores()
    P = passage_frame()
    if a.export_scope:
        export_scope(P)
        return
    D = doc_frame(P, scores)
    P = P.merge(D[["doc_key", "year"]], on="doc_key", how="left")

    # --- SEN por palabra y variantes sin vocabulario ESRS
    R, T, sen_info = sen_decomposition(scores)
    var_cols = ["SEN_exA", "SEN_exB", "SEN_exTop10", "SEN_exTop25", "cuota_N_exA", "cuota_N_exB"]
    D = D.merge(R[["doc_key"] + var_cols], on="doc_key", how="left", validate="one_to_one")
    for c in ["SEN_exA", "SEN_exB", "SEN_exTop10", "SEN_exTop25", "SEN_clim", "SEN_nonclim"]:
        D[f"Z_{c}"] = zpop(D[c])
    T.round(4).to_csv(OUT_DIR / "sen_negative_words_2023_2024.csv", sep=";", index_label="word")

    pub = D.drop(columns=["doc_key", "firm"])
    pub.to_csv(OUT_DIR / "doc_level_cb.csv", sep=";", index=False)
    Dall = D.copy()
    sin_cti = D[D["CTI"].isna()]
    D = D[D["CTI"].notna()].reset_index(drop=True)

    S: dict = {"cobertura": {
        "informes": int(len(pub)), "pasajes": int(len(P)),
        "pasajes_climaticos": int(P["clim"].sum()), "cuota_climatica": float(P["clim"].mean()),
        "pasajes_compromiso": int(P["commit"].sum()),
        "climaticos_por_informe": {"min": int(Dall["n_climate"].min()),
                                   "mediana": float(Dall["n_climate"].median()),
                                   "max": int(Dall["n_climate"].max())},
        "compromiso_por_informe": {"min": int(D["n_commit"].min()),
                                   "p05": float(D["n_commit"].quantile(.05)),
                                   "mediana": float(D["n_commit"].median())},
        "informes_con_menos_de_10_compromisos": int((D["n_commit"] < 10).sum()),
        "informes_sin_CTI": sin_cti[["firm_label", "year", "n_passages", "n_climate", "n_commit"]]
        .to_dict("records"),
        "informes_analizados": int(len(D)),
        "frente_al_piloto": coverage_vs_pilot(P, Dall),
    }}
    S["runtime_inferencia"] = json.loads((CACHE_DIR / "runtime.json").read_text())
    S["descriptivos"] = {c: {"media": float(D[c].mean()), "dt": float(D[c].std()),
                             "min": float(D[c].min()), "max": float(D[c].max())}
                         for c in ["CTI", "CTI_prob", "spec_share", "sent_net", "commit_share",
                                   "clim_share"] + SCOPE_VARS}
    S["corr_CTI_CTI_prob_spearman"] = float(stats.spearmanr(D["CTI"], D["CTI_prob"])[0])
    S["corr_sent_net_prob_spearman"] = float(stats.spearmanr(D["sent_net"], D["sent_net_prob"])[0])
    S["sus_alcance_control"] = {
        "rho_SUS_passages_vs_SUS_density": float(stats.spearmanr(Dall["SUS_passages"], Dall["SUS_density"],
                                                                 nan_policy="omit")[0])
        if Dall["SUS_passages"].notna().any() else None}
    S["sen"] = sen_info

    # --- Correlaciones
    C1 = boot_corr(D, CB_VARS, OUR_VARS).assign(block="paper_scores")
    C2 = boot_corr(D.dropna(subset=SCOPE_VARS) if D["SUS_clim"].notna().any() else D,
                   ["sent_net", "spec_share", "CTI"],
                   [v for v in SCOPE_VARS if D[v].notna().any()]).assign(block="climate_scope_dictionary")
    C3 = boot_corr(D, ["sent_net", "sent_net_prob"], ["Z_SEN", "Z_SEN_exA", "Z_SEN_exB", "Z_SEN_exTop25"]) \
        .assign(block="sen_variants")
    C = pd.concat([C1, C2, C3], ignore_index=True)
    C.to_csv(OUT_DIR / "correlations.csv", sep=";", index=False)
    D10 = D[D["n_commit"] >= 10]
    S["sensibilidad_n_commit_ge_10"] = {
        "informes": int(len(D10)),
        "rho_CTI": {y: float(stats.spearmanr(D10["CTI"], D10[y])[0]) for y in OUR_VARS}}

    # --- MTMM
    vars_ = MTMM_DICT + MTMM_CB
    S["mtmm"] = {}
    for view, meth in [("pooled", "spearman"), ("within", "pearson"), ("between", "pearson"),
                       ("twoway", "pearson")]:
        M = mtmm(D, vars_, view, meth)
        M.round(4).to_csv(OUT_DIR / f"mtmm_{view}_{meth}.csv", sep=";")
        S["mtmm"][f"{view}_{meth}"] = campbell_fiske(M, MTMM_DICT, MTMM_CB, VALIDITY)
    sv = ["SEN_clim", "SUS_clim", "QUANT_clim", "HEDGE_clim", "sent_net", "spec_share", "CTI"]
    if D["SUS_clim"].notna().any():
        Ms = mtmm(D, sv, "pooled", "spearman")
        Ms.round(4).to_csv(OUT_DIR / "mtmm_climate_scope_pooled_spearman.csv", sep=";")
        tr = {"SEN_clim": {"tone"}, "sent_net": {"tone"}, "SUS_clim": {"substance"},
              "QUANT_clim": {"substance"}, "spec_share": {"substance"}, "HEDGE_clim": {"vagueness"},
              "CTI": {"vagueness"}}
        S["mtmm"]["climate_scope_pooled_spearman"] = campbell_fiske(
            Ms, sv[:4], sv[4:], [("SEN_clim", "sent_net", 1), ("SUS_clim", "spec_share", 1),
                                 ("QUANT_clim", "spec_share", 1), ("HEDGE_clim", "CTI", 1)], tr)

    # --- Ruptura de 2024 y medias anuales
    brk_vars = ["sent_net", "sent_net_prob", "opp_share", "risk_share", "sent_net_exFrame",
                "sent_net_exFrameWide", "frame_share_clim", "frame_wide_share_clim", "clim_share", "spec_share", "CTI", "commit_share",
                "Z_SEN", "Z_SEN_clim", "Z_SEN_nonclim", "Z_SEN_exA", "Z_SEN_exB", "Z_SEN_exTop10",
                "Z_SEN_exTop25", "Z_SUS", "ESGSI"]
    S["ruptura_2024"] = {c: break_block(Dall, c, av.N_BOOT) for c in brk_vars}
    # Amplitud: cuantas empresas caen en 2024 frente a su media de 2018-2023, y
    # si la caida es mayor donde el texto climatico de 2024 usa mas el marco ESRS.
    S["ruptura_2024_por_empresa"] = {}
    f24 = Dall[Dall["year"] == 2024].set_index("firm")["frame_share_clim"]
    for c in ["sent_net", "opp_share", "risk_share", "spec_share", "commit_share", "CTI", "clim_share",
              "SEN_Score", "SEN_exA"]:
        w = Dall.pivot_table(index="firm", columns="year", values=c)
        dif = (w[2024] - w.loc[:, 2018:2023].mean(axis=1)).dropna()
        S["ruptura_2024_por_empresa"][c] = {
            "empresas": int(len(dif)), "cuota_que_baja": float((dif < 0).mean()),
            "rho_cambio_vs_cuota_marco_ESRS_2024": float(stats.spearmanr(dif, f24.loc[dif.index])[0])}
    yc = []
    for c in brk_vars:
        d = Dall.dropna(subset=[c]).copy()
        d[c] = zpop(d[c])  # todas en DT agregadas para comparar
        yc.append(av.yearly_ci(d, c))
    YC = pd.concat(yc, ignore_index=True)
    YC.round(4).to_csv(OUT_DIR / "yearly_ci_z.csv", sep=";", index=False)
    yr = Dall.groupby("year")[["CTI", "CTI_prob", "spec_share", "sent_net", "opp_share", "risk_share",
                               "sent_net_exFrame", "sent_net_exFrameWide", "frame_share_clim",
                               "frame_wide_share_clim", "commit_share",
                               "clim_share", "SEN_Score", "SEN_clim", "SEN_nonclim", "SEN_exA",
                               "SEN_exB", "ESGSI", "ESGSI_ext"]].mean()
    yr.round(4).to_csv(OUT_DIR / "yearly_means_raw.csv", sep=";")

    # --- Tendencias con EF de empresa
    S["pendientes"] = {}
    for c in ["CTI", "CTI_prob", "spec_share", "sent_net", "commit_share", "clim_share"]:
        tb = av.trend_block(D, c, av.N_BOOT, av.SEED)
        tb["ef_2018_2023"] = av.fe_slope(D[D["year"] <= 2023], c)
        tb["dt_agregada"] = float(D[c].std(ddof=0))
        S["pendientes"][c] = tb

    # --- Deciles extremos
    S["deciles"] = {}
    for a_ in ["ESGSI", "ESGSI_ext"]:
        for b_ in ["CTI", "CTI_prob"]:
            ov = av.decile_overlap(D[a_], D[b_])
            k, n = ov["k"], ov["n"]
            xt, xb = round(ov["top"] * k), round(ov["bottom"] * k)
            hg = stats.hypergeom(n, k, k)
            S["deciles"][f"{a_}~{b_}"] = {"k": k, "top": xt, "bottom": xb, "esperado": k * k / n,
                                          "p_top": float(hg.sf(xt - 1)), "p_bottom": float(hg.sf(xb - 1))}

    # --- Nivel de pasaje
    S["pasaje"], _ = passage_level(P)
    S["segundos_analisis"] = round(time.time() - t0, 1)

    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(av.jsonable(S), fh, indent=2, ensure_ascii=False)

    # --- Consola
    pd.set_option("display.width", 250)
    for blk in C["block"].unique():
        t = C[C["block"] == blk].copy()
        t = t[(t["method"] == "spearman") & (t["view"] == "pooled") | (t["method"] == "pearson") &
              (t["view"] != "pooled")]
        t["txt"] = t.apply(lambda r: f"{r.r:+.2f} [{r.ci_lo:+.2f},{r.ci_hi:+.2f}]", axis=1)
        print(t.pivot_table(index=["cb", "score"], columns=["view"], values="txt",
                            aggfunc="first")[["pooled", "within", "between", "twoway"]].to_string())
    print(json.dumps(av.jsonable({k: S[k] for k in ["cobertura", "mtmm", "deciles", "pasaje", "sen",
                                                   "sensibilidad_n_commit_ge_10",
                                                   "sus_alcance_control"]}), indent=1,
                     ensure_ascii=False))
    for c, v in S["ruptura_2024"].items():
        e, f = v["salto_ef"], v["salto_sobre_tendencia"]
        print(f"{c:15s} pre={v['media_z_2018_2023']:+.3f} 2024={v['media_z_2024']:+.3f} "
              f"salto={e['b']:+.3f} [{e['ci95'][0]:+.3f},{e['ci95'][1]:+.3f}] p={e['p']:.3g} "
              f"pw={e['p_wild']:.3g} | s/tend={f['b']:+.3f} p={f['p']:.3g} pw={f['p_wild']:.3g} "
              f"| pend.pre={v['pendiente_2018_2023_z_pre']['pendiente']:+.3f} "
              f"p={v['pendiente_2018_2023_z_pre']['p']:.3g}")
    for c, v in S["pendientes"].items():
        e = v["ef_cluster"]
        print(c, f"{e['pendiente']:+.4f}", [round(x, 4) for x in e["ci95"]], f"p={e['p']:.3g}",
              f"pw={v['ef_wild_bootstrap']['p_wild']:.3g}",
              f"pre: {v['ef_2018_2023']['pendiente']:+.4f} p={v['ef_2018_2023']['p']:.3g}")
    print(YC.pivot_table(index="variable", columns="year", values="mean").round(3).to_string())
    print(T.head(30).round(3).to_string())
    print(f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
