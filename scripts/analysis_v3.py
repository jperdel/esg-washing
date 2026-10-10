"""
analysis_v3.py
--------------
Capa de analisis del paper sobre la especificacion adoptada (vocabulario v3
con sectoriales, SUS_MODE='density'). Lee las salidas del pipeline y escribe
todo lo que el paper puede citar en un unico directorio:

    results/analysis_v3/
        summary.json            todos los escalares citables (una sola fuente)
        *.csv                   tablas intermedias (separador ';')
        tex/*.tex               fragmentos tabular listos para \\input (booktabs)
        pdf_wordcounts.csv      cache de palabras del texto completo de cada PDF

    PYTHONIOENCODING=utf-8 python scripts/analysis_v3.py
    python scripts/analysis_v3.py --results <csv> --processed <csv> ...  (ver --help)

Bloques (letras del encargo):
    a  descriptivos y medias de los scores por grupo
    b  especificaciones del SUS: correlaciones; indice con cada una comparado
       como score (Pearson, Spearman, solapamiento de deciles extremos)
    c  serie temporal, regresiones de tendencia (MCO, cluster por empresa, efectos
       fijos de empresa, wild cluster bootstrap), Alemania sola
    d  corte transversal: pais, tipo de documento, supersector, industria ICB;
       pendientes por grupo, descomposicion de la varianza, comparacion intra-pais
    e  nivel empresa (identificadores anonimos): medias, pendientes, dispersion,
       persistencia del ranking (Spearman entre anos)
    f  pilares: recuentos por pilar con el vocabulario lematizado exactamente como
       el analizador; terminos mas pesados y distintivos; tendencia sin marcos
    g  estadisticas de extraccion y rendimiento frente al texto completo del PDF
    w  rejilla de pesos del ESGSI extendido (w_q, w_h en {0, .25, .5, .75, 1})

Directrices (paper/_research/directrices.md): el indice es un score continuo;
no hay umbral ni etiquetas, ni recuentos de "senalados". Las comparaciones
entre scores son Pearson r, Spearman rho y solapamiento de los deciles extremos
(decile_overlap). El ESGSI extendido es una metrica propia: cada corte del
ESGSI se calcula tambien para ESGSI_ext. Ninguna salida citable lleva nombres
de empresa: las empresas son identificadores anonimos '<Industria>-NN' (NN =
posicion por ESGSI medio dentro de la industria); el nombre de carpeta solo
aparece como clave de cruce en ficheros '*_internal.*', que no van al paper.

Convenciones: z-score con desviacion tipica POBLACIONAL (np.std, ddof=0), como
src/esgsi_analyzer.py. Toda regresion sobre el panel lleva errores agrupados
por empresa (49 clusters); la MCO convencional se da solo al lado y etiquetada
como tal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import unicodedata
import warnings
from pathlib import Path

# El vocabulario adoptado: debe fijarse ANTES de importar src/config.py.
os.environ["ESG_RUN_TAG"] = "v3"
os.environ["ESG_INCLUDE_SECTORAL"] = "1"

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.feature_extraction.text import CountVectorizer

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR / "scripts"))

METADATA_DIR = BASE_DIR / "metadata"
RESULTS_DIR = BASE_DIR / "results"
DATA_DIR = BASE_DIR / "data"

SEED = 20260925
N_BOOT = 9999

# ---------------------------------------------------------------------------
# Etiquetas y agrupaciones
# ---------------------------------------------------------------------------

COUNTRY_EN = {
    "ALEMANIA": "Germany", "BELGICA": "Belgium", "ESPAÑA": "Spain",
    "FINLANDIA": "Finland", "FRANCIA": "France", "ITALIA": "Italy",
    "PAISES BAJOS": "Netherlands",
}
DOCTYPE_EN = {
    "Annual report": "Annual report", "URD": "URD",
    "Informe de sostenibilidad": "Sustainability report",
}

# Supersector STOXX/ICB -> industria ICB (dos niveles; ambos se conservan).
SUPERSECTOR_TO_INDUSTRY = {
    "Banks": "Financials", "Insurance": "Financials", "Financial Services": "Financials",
    "Industrial Goods and Services": "Industrials", "Construction and Materials": "Industrials",
    "Automobiles and Parts": "Consumer Discretionary",
    "Consumer Products and Services": "Consumer Discretionary",
    "Retail": "Consumer Discretionary",
    "Food, Beverage and Tobacco": "Consumer Staples",
    "Personal Care, Drug and Grocery Stores": "Consumer Staples",
    "Chemicals": "Basic Materials", "Energy": "Energy", "Utilities": "Utilities",
    "Health Care": "Health Care", "Healthcare": "Health Care",
    "Technology": "Technology", "Telecommunications": "Telecommunications",
}

# Empresas: identificador anonimo '<Industria>-NN', NN = posicion (01 = mayor
# ESGSI medio 2018-2024) dentro de la industria ICB. Ver firm_ids().

# Rejilla de pesos del ESGSI extendido y pesos adoptados (src/config.py).
EXT_WEIGHT_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
EXT_DEFAULT = (0.5, 0.5)
DECILE_Q = 0.10

# Variables descriptivas: columna de results.csv -> nombre en el paper.
DESC_VARS = {
    "SUS_density": r"\susdens", "SUS_tfidf_length": r"\sustfidf", "SUS_lagasio": r"\suslag",
    "Breadth": r"\BREADTH", "SEN_Score": r"\SEN", "QUANT_Score": r"\QUANT",
    "HEDGE_Score": r"\HEDGE", "ESGSI": r"\ESGSI", "ESGSI_ext": r"\ESGSIext",
}

# Entradas que NOMBRAN un regimen de divulgacion, un estandar, un marco o una
# agencia de rating (forma natural, tal como aparecen en ESG_terms_v3.csv).
# Se lematizan con el pipeline real antes de quitarlas del vocabulario.
# Criterio: su frecuencia puede crecer por pura cronologia regulatoria
# (CSRD/ESRS, Taxonomia, SFDR, ISSB/TNFD no existian o no obligaban en 2018),
# no por mas contenido sustantivo.
REGIME_TERMS = [
    # regulacion y estandares UE
    "csrd", "esrs", "eu taxonomy", "taxonomy alignment", "taxonomy eligible",
    "sfdr", "double materiality",
    # estandares y marcos voluntarios
    "global reporting initiative", "sasb", "issb", "tcfd", "tnfd", "iirc",
    "cdp", "un global compact", "sdg", "2030 agenda", "science based target",
    # agencias de rating ESG
    "ecovadis", "ftse4good", "msci esg", "sustainalytics",
]

# Duplicado conocido (344 PDF -> 343 informes unicos). Sin nombre de empresa:
# el detalle con la carpeta va a dedup_internal.json.
KNOWN_DUPLICATE = "un informe de 2019 con dos PDF (copia '_repaired') de texto extraido identico"


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def nfc(x) -> str:
    return unicodedata.normalize("NFC", str(x)).strip()


def z(x) -> np.ndarray:
    """Z-score con desviacion tipica poblacional, identico a ESGSIAnalyzer._z_score."""
    x = np.asarray(x, dtype=float)
    s = np.std(x)
    return np.zeros_like(x) if s == 0 else (x - np.mean(x)) / s


def decile_overlap(a, b, q: float = DECILE_Q) -> dict:
    """
    Solapamiento de los deciles extremos de dos scores sobre los mismos
    informes: k = round(q * n) informes con mayor (menor) valor en cada score;
    'top' ('bottom') = cuota de los k de a que tambien estan entre los k de b.
    Empates: orden estable por posicion (con scores continuos no los hay).
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n = len(a)
    k = max(int(round(q * n)), 1)
    oa, ob = np.argsort(a, kind="stable"), np.argsort(b, kind="stable")
    top = len(set(oa[-k:]) & set(ob[-k:])) / k
    bottom = len(set(oa[:k]) & set(ob[:k])) / k
    return {"top": float(top), "bottom": float(bottom), "k": k, "n": n, "q": q}


def compare_scores(a, b, q: float = DECILE_Q) -> dict:
    """Comparacion de dos scores: Pearson r, Spearman rho y solapamiento de deciles extremos."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ov = decile_overlap(a, b, q)
    return {"r": float(np.corrcoef(a, b)[0, 1]), "rho": float(stats.spearmanr(a, b)[0]),
            "solape_top": ov["top"], "solape_bottom": ov["bottom"], "k_decil": ov["k"]}


def firm_ids(df: pd.DataFrame) -> dict:
    """
    Identificador anonimo por empresa: '<Industria>-NN', NN = posicion (01 =
    mayor) del ESGSI medio 2018-2024 dentro de la industria ICB. Devuelve
    {nombre de carpeta: identificador}; el nombre no sale de los '*_internal'.
    """
    fm = df.groupby("firm").agg(industry=("industry", "first"), m=("ESGSI", "mean")).reset_index()
    fm = fm.sort_values(["industry", "m"], ascending=[True, False])
    fm["k"] = fm.groupby("industry").cumcount() + 1
    return {r.firm: f"{r.industry}-{r.k:02d}" for r in fm.itertuples()}


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, pd.DataFrame):
        return jsonable(o.to_dict(orient="records"))
    if isinstance(o, pd.Series):
        return jsonable(o.to_dict())
    return o


# --- Formato LaTeX, como el paper: cifras en modo matematico, miles con {,} ---

def tnum(x, d: int = 3, sign: bool = False) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    s = f"{x:+,.{d}f}" if sign else f"{x:,.{d}f}"
    s = s.replace(",", "{,}")
    return f"${s}$"


def tint(x) -> str:
    return f"{int(x):,}".replace(",", "{,}")


def tp(p: float) -> str:
    """p-valor: tres decimales, o notacion a x 10^-k si es pequeno."""
    if p is None or not np.isfinite(p):
        return "--"
    if p >= 0.001:
        return f"${p:.3f}$"
    m, e = f"{p:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def tpw(p: float, n_boot: int) -> str:
    """p del wild bootstrap: si ningun t* supera al observado, cota 1/(B+1)."""
    return tp(p) if p > 0 else f"$<{1 / (n_boot + 1):.4f}$"


def tfrac(k: int, n: int) -> str:
    return f"{int(k)}/{int(n)}"


def tpct(x: float, d: int = 1, sign: bool = False) -> str:
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return f"${s}\\%$"


def tex_escape(s: str) -> str:
    return (str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
            .replace("#", r"\#"))


def write_tabular(path: Path, colspec: str, header: list[str], rows: list[list[str]],
                  midrules: list[int] | None = None, note: str | None = None):
    """Fragmento tabular (sin table/caption: el caption y el label van en la seccion)."""
    midrules = set(midrules or [])
    lines = [f"% GENERADO POR scripts/analysis_v3.py -- NO EDITAR A MANO.",
             f"\\begin{{tabular}}{{@{{}}{colspec}@{{}}}}", r"\toprule",
             " & ".join(header) + r" \\", r"\midrule"]
    for i, r in enumerate(rows):
        if i in midrules:
            lines.append(r"\midrule")
        lines.append(" & ".join(r) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if note:
        lines.insert(1, f"% {note}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_csv(df: pd.DataFrame, path: Path):
    df.to_csv(path, sep=";", index=False, encoding="utf-8")


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------

KEY = ["Documento", "País", "Compañía", "Año"]


def norm_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ("Documento", "País", "Compañía"):
        df[c] = df[c].map(nfc)
    df["Año"] = df["Año"].astype(str).str.strip()
    return df


def load_results(path: Path, sectors_path: Path) -> pd.DataFrame:
    res = norm_keys(pd.read_csv(path, sep=";", encoding="utf-8"))
    sec = pd.read_csv(sectors_path, sep=";", encoding="utf-8")
    sec.columns = [nfc(c) for c in sec.columns]
    sec = sec.rename(columns={"Empresa": "firm", "Supersector STOXX/ICB": "supersector"})
    sec["firm"] = sec["firm"].map(nfc)
    sec["supersector"] = sec["supersector"].map(nfc)
    res["firm"] = res["Compañía"]
    missing = sorted(set(res["firm"]) - set(sec["firm"]))
    if missing:
        raise ValueError(f"empresas sin supersector en {sectors_path.name}: {missing}")
    df = res.merge(sec[["firm", "supersector"]], on="firm", how="left", validate="many_to_one")
    unmapped = sorted(set(df["supersector"]) - set(SUPERSECTOR_TO_INDUSTRY))
    if unmapped:
        raise ValueError(f"supersectores sin industria ICB: {unmapped}")
    df["industry"] = df["supersector"].map(SUPERSECTOR_TO_INDUSTRY)
    # firm_label: identificador ANONIMO (nunca el nombre); 'firm' es la clave interna.
    df["firm_label"] = df["firm"].map(firm_ids(df))
    df["country"] = df["País"].map(COUNTRY_EN)
    df["doctype"] = df["TipoDocumento"].map(DOCTYPE_EN).fillna(df["TipoDocumento"])
    if df["country"].isna().any():
        raise ValueError(f"pais sin traducir: {df.loc[df['country'].isna(), 'País'].unique()}")
    df["year"] = df["Año"].astype(int)
    return df.reset_index(drop=True)


def load_corpus(processed: Path, with_raw: bool = False) -> tuple[pd.DataFrame, list[dict]]:
    """Mismo corpus y misma deduplicacion (MD5 de clean_text) que main.py.
    with_raw=True conserva tambien raw_text (QUANT y HEDGE se miden sobre el)."""
    usecols = KEY + ["clean_text"] + (["raw_text"] if with_raw else [])
    df = pd.read_csv(processed, sep=";", usecols=usecols, encoding="utf-8")
    df = norm_keys(df)
    digest = df["clean_text"].astype(str).map(lambda t: hashlib.md5(t.encode("utf-8")).hexdigest())
    dup = digest.duplicated()
    dropped = []
    for i in np.where(dup)[0]:
        first = df.loc[(digest == digest.iloc[i]) & ~dup].iloc[0]
        dropped.append({"descartado": df.loc[i, "Documento"], "empresa": df.loc[i, "Compañía"],
                        "año": df.loc[i, "Año"], "identico_a": first["Documento"]})
    n_raw = len(df)
    df = df[~dup].reset_index(drop=True)
    return df, [{"filas_processed_texts": n_raw, "unicos": len(df)}] + dropped


# ---------------------------------------------------------------------------
# Vocabulario y pilares
# ---------------------------------------------------------------------------

def load_vocab_and_pillars(terms_v3: Path):
    """
    Vocabulario TF-IDF exactamente como el analizador (config.ESG_KEYWORDS con
    ESG_RUN_TAG=v3, ESG_INCLUDE_SECTORAL=1) y mapa lema -> pilar construido
    pasando cada termino natural de ESG_terms_v3.csv por el MISMO TextProcessor
    que usa scripts/build_lexicons.py.
    """
    import config
    from build_lexicons import SPACY_MODEL, _read_terms
    from text_processor import TextProcessor

    vocab = list(config.ESG_KEYWORDS)
    base = _read_terms(METADATA_DIR / "esg_terms_lemmatized.txt")
    sect = _read_terms(METADATA_DIR / "esg_terms_sectorial_lemmatized.txt")
    sectoral = {t for t in sect if t not in set(base)}

    processor = TextProcessor(extra_sw=_read_terms(METADATA_DIR / "personal_stopwords.txt"),
                              spacy_model=SPACY_MODEL)
    v3 = pd.read_csv(terms_v3, sep=";", encoding="utf-8-sig", dtype=str).fillna("")
    rows = []
    for r in v3.itertuples():
        if r.tipo != "termino":
            continue
        lemma = processor.preprocess(r.termino.lower()).strip()
        rows.append({"termino": r.termino, "lema": lemma, "pilar": r.pilar,
                     "sectorial_csv": r.sectorial == "1", "en_vocabulario": lemma in set(vocab)})
    lem = pd.DataFrame(rows)

    pillar: dict[str, str] = {}
    conflicts = []
    for r in lem[lem["en_vocabulario"]].itertuples():
        if r.lema in pillar and pillar[r.lema] != r.pilar:
            conflicts.append({"lema": r.lema, "pilar_asignado": pillar[r.lema],
                              "pilar_alternativo": r.pilar, "termino": r.termino})
            continue
        pillar.setdefault(r.lema, r.pilar)
    unmapped_vocab = [t for t in vocab if t not in pillar]
    not_in_vocab = lem[~lem["en_vocabulario"]][["termino", "lema", "pilar"]].to_dict(orient="records")

    regime_lemmas = {}
    for t in REGIME_TERMS:
        lemma = processor.preprocess(t).strip()
        regime_lemmas[t] = lemma if lemma in set(vocab) else None

    info = {
        "entradas_vocabulario": len(vocab),
        "entradas_base": len(set(base)), "entradas_sectoriales_extra": len(sectoral),
        "entradas_por_pilar": pd.Series([pillar.get(t, "?") for t in vocab]).value_counts().to_dict(),
        "vocabulario_sin_pilar": unmapped_vocab,
        "terminos_v3_fuera_del_vocabulario_tfidf": not_in_vocab,
        "conflictos_de_pilar": conflicts,
        "regimenes_lista_natural": REGIME_TERMS,
        "regimenes_lema": regime_lemmas,
    }
    return vocab, pillar, sectoral, lem, info, processor


def count_matrix(texts: list[str], vocab: list[str]):
    """Igual que ESGSIAnalyzer: CountVectorizer(vocabulary=..., ngram_range=(1, max_n))."""
    max_n = max((len(k.split()) for k in vocab), default=1)
    cv = CountVectorizer(vocabulary=vocab, ngram_range=(1, max_n))
    C = cv.fit_transform(texts).toarray().astype(float)
    return C, list(cv.get_feature_names_out())


# ---------------------------------------------------------------------------
# Regresiones
# ---------------------------------------------------------------------------

def within(v: pd.Series, g: pd.Series) -> np.ndarray:
    return (v - v.groupby(g).transform("mean")).to_numpy(dtype=float)


def cluster_fit(y, X, groups):
    """MCO con errores agrupados; inferencia t con G-1 grados de libertad."""
    return sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups}, use_t=True)


def wild_cluster_p(yd: np.ndarray, xd: np.ndarray, groups: np.ndarray,
                   n_boot: int, seed: int) -> dict:
    """
    Wild cluster bootstrap restringido (WCR, Rademacher) para la pendiente de
    efectos fijos, H0: beta = 0. En el espacio 'within' el modelo restringido
    solo tiene los efectos fijos, asi que sus residuos son el propio y
    desviado de la media de la empresa. p = P(|t*| >= |t|), t con varianza
    agrupada (el factor de correccion de pequena muestra se cancela).
    """
    codes, uniq = pd.factorize(groups)
    G = len(uniq)
    sxx = (xd ** 2).sum()

    def tstat(Y):  # Y: (B, N)
        beta = Y @ xd / sxx
        U = Y - beta[:, None] * xd[None, :]
        S = np.zeros((Y.shape[0], G))
        for g in range(G):
            m = codes == g
            S[:, g] = (U[:, m] * xd[m]).sum(axis=1)
        se = np.sqrt((S ** 2).sum(axis=1)) / sxx
        return beta / se

    t_obs = tstat(yd[None, :])[0]
    rng = np.random.default_rng(seed)
    W = rng.choice([-1.0, 1.0], size=(n_boot, G))
    Ystar = yd[None, :] * W[:, codes]
    t_b = tstat(Ystar)
    p = float((np.abs(t_b) >= abs(t_obs)).mean())
    return {"t": float(t_obs), "p_wild": p, "reps": n_boot, "clusters": G,
            "pesos": "Rademacher", "semilla": seed, "tipo": "WCR (H0 impuesta)"}


def trend_block(df: pd.DataFrame, col: str, n_boot: int, seed: int) -> dict:
    """(i) MCO agrupada convencional, (ii) MCO agrupada con cluster por empresa,
    (iii) efectos fijos de empresa con cluster, (iv) wild cluster bootstrap."""
    y = df[col].to_numpy(dtype=float)
    X = sm.add_constant(df["year"].to_numpy(dtype=float))
    g = df["firm"].to_numpy()
    ols = sm.OLS(y, X).fit()
    clu = cluster_fit(y, X, g)
    yd, xd = within(df[col], df["firm"]), within(df["year"].astype(float), df["firm"])
    fe = cluster_fit(yd, xd[:, None], g)
    ci_fe = fe.conf_int()[0]
    out = {
        "n": len(df), "firmas": int(df["firm"].nunique()),
        "mco_convencional": {"pendiente": ols.params[1], "se": ols.bse[1], "p": ols.pvalues[1],
                             "r2": ols.rsquared},
        "mco_cluster": {"pendiente": clu.params[1], "se": clu.bse[1], "p": clu.pvalues[1],
                        "ci95": list(clu.conf_int()[1])},
        "ef_cluster": {"pendiente": fe.params[0], "se": fe.bse[0], "p": fe.pvalues[0],
                       "ci95": list(ci_fe), "r2_within": fe.rsquared,
                       "gl_inferencia": int(df["firm"].nunique() - 1)},
    }
    out["ef_wild_bootstrap"] = wild_cluster_p(yd, xd, g, n_boot, seed)
    return out


def yearly_ci(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Media anual con IC 95 % agrupado por empresa (regresion sobre dummies de ano)."""
    D = pd.get_dummies(df["year"]).astype(float)
    m = cluster_fit(df[col].to_numpy(float), D.to_numpy(), df["firm"].to_numpy())
    ci = m.conf_int()
    return pd.DataFrame({"year": D.columns.astype(int), "variable": col, "mean": m.params,
                         "ci_lo": ci[:, 0], "ci_hi": ci[:, 1], "se_cluster": m.bse})


def group_slopes(df: pd.DataFrame, col: str, grp: str) -> tuple[pd.DataFrame, dict]:
    """
    Pendiente por grupo con efectos fijos de empresa: y_it = a_i + b_g(i)*t + e,
    errores agrupados por empresa. En un panel equilibrado b_g coincide con la
    media de las pendientes MCO de las empresas del grupo, asi que se da
    tambien el intervalo t sobre esas pendientes (n_firmas-1 gl), que es el
    honesto cuando el grupo tiene pocas empresas.
    """
    yd = within(df[col], df["firm"])
    xd = within(df["year"].astype(float), df["firm"])
    groups = sorted(df[grp].unique())
    X = np.column_stack([xd * (df[grp] == gname).to_numpy() for gname in groups])
    m = cluster_fit(yd, X, df["firm"].to_numpy())
    ci = m.conf_int()
    fs = firm_slopes(df, col)
    fs = fs.merge(df[["firm", grp]].drop_duplicates(), on="firm")
    rows = []
    for j, gname in enumerate(groups):
        s = fs.loc[fs[grp] == gname, "slope"].to_numpy()
        nf = len(s)
        if nf > 1:
            h = stats.t.ppf(0.975, nf - 1) * s.std(ddof=1) / np.sqrt(nf)
            p_t = float(stats.ttest_1samp(s, 0).pvalue)
        else:
            h, p_t = np.nan, np.nan
        rows.append({grp: gname, "n_firms": nf, "n_docs": int((df[grp] == gname).sum()),
                     "slope": m.params[j], "se_cluster": m.bse[j], "ci_lo": ci[j, 0],
                     "ci_hi": ci[j, 1], "p_cluster": m.pvalues[j],
                     "firmslope_mean": s.mean(), "firmslope_ci_lo": s.mean() - h,
                     "firmslope_ci_hi": s.mean() + h, "firmslope_p": p_t})
    R = np.zeros((len(groups) - 1, len(groups)))
    for i in range(len(groups) - 1):
        R[i, 0], R[i, i + 1] = 1, -1
    # Con grupos de 1-2 empresas la varianza agrupada no es de rango completo;
    # statsmodels usa entonces el rango efectivo (df_num) y avisa. Es esperado.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ft = m.f_test(R)
    eq = {"F": float(np.squeeze(ft.fvalue)), "p": float(ft.pvalue), "df_num": int(ft.df_num),
          "df_den": float(ft.df_denom)}
    return pd.DataFrame(rows), eq


def firm_slopes(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = []
    for f, d in df.groupby("firm"):
        sl = stats.linregress(d["year"], d[col])
        out.append({"firm": f, "slope": sl.slope})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Bloques del analisis
# ---------------------------------------------------------------------------

def fe_slope(df: pd.DataFrame, col: str) -> dict:
    """Pendiente temporal con efectos fijos de empresa y errores agrupados por
    empresa (igual que trend_block['ef_cluster'], sin el bootstrap)."""
    yd, xd = within(df[col], df["firm"]), within(df["year"].astype(float), df["firm"])
    m = cluster_fit(yd, xd[:, None], df["firm"].to_numpy())
    ci = m.conf_int()[0]
    return {"pendiente": float(m.params[0]), "se": float(m.bse[0]), "p": float(m.pvalues[0]),
            "ci95": [float(ci[0]), float(ci[1])]}


SCORES = ("ESGSI", "ESGSI_ext")
SCORE_TEX = {"ESGSI": r"$\ESGSI$", "ESGSI_ext": r"$\ESGSIext$"}


def describe(s: pd.Series) -> dict:
    q = s.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    return {"n": int(s.count()), "mean": s.mean(), "sd": s.std(ddof=1), "min": s.min(),
            "p05": q[0.05], "p25": q[0.25], "median": q[0.5], "p75": q[0.75],
            "p95": q[0.95], "max": s.max()}


def means_by(df: pd.DataFrame, by: str | None) -> pd.DataFrame:
    """Medias (y sd) de los scores y componentes por grupo."""
    g = df.groupby(by) if by else df.assign(_all="All").groupby("_all")
    out = g.agg(n_docs=("ESGSI", "size"), n_firms=("firm", "nunique"),
                mean_ESGSI=("ESGSI", "mean"), sd_ESGSI=("ESGSI", "std"),
                mean_ESGSI_ext=("ESGSI_ext", "mean"), sd_ESGSI_ext=("ESGSI_ext", "std"),
                mean_SEN=("SEN_Score", "mean"), mean_SUS=("SUS_density", "mean"),
                mean_Breadth=("Breadth", "mean"), mean_QUANT=("QUANT_Score", "mean"),
                mean_HEDGE=("HEDGE_Score", "mean")).reset_index()
    out = out.rename(columns={out.columns[0]: "group"})
    out.insert(0, "dimension", by or "all")
    return out


def block_a(df, out, tex, S):
    desc = pd.DataFrame([{"variable": v, **describe(df[v])} for v in DESC_VARS])
    save_csv(desc, out / "descriptives.csv")
    S["a_descriptivos"] = {r["variable"]: {k: r[k] for k in desc.columns if k != "variable"}
                           for r in desc.to_dict(orient="records")}
    dec = {"SUS_density": 3, "SUS_tfidf_length": 3, "SUS_lagasio": 4, "Breadth": 2,
           "SEN_Score": 3, "QUANT_Score": 4, "HEDGE_Score": 4, "ESGSI": 3, "ESGSI_ext": 3}
    rows = [[f"${DESC_VARS[r.variable]}$"] +
            [tnum(getattr(r, k), dec[r.variable]) for k in ("mean", "sd", "min", "p25", "median", "p75", "max")]
            for r in desc.itertuples()]
    write_tabular(tex / "descriptives.tex", "lrrrrrrr",
                  ["", "Mean", "SD", "Min", "P25", "Median", "P75", "Max"], rows,
                  midrules=[3, 4, 7], note=f"n = {len(df)} documentos; SD muestral (ddof=1)")

    parts = [means_by(df, None)] + [means_by(df, c) for c in
                                    ("year", "country", "doctype", "supersector", "industry")]
    gm = pd.concat(parts, ignore_index=True)
    save_csv(gm, out / "group_means.csv")
    S["a_medias_por_grupo"] = {
        dim: {str(r["group"]): {"n": int(r["n_docs"]), "firmas": int(r["n_firms"]),
                                "ESGSI": r["mean_ESGSI"], "ESGSI_sd": r["sd_ESGSI"],
                                "ESGSI_ext": r["mean_ESGSI_ext"], "ESGSI_ext_sd": r["sd_ESGSI_ext"],
                                "SEN": r["mean_SEN"], "SUS_density": r["mean_SUS"],
                                "QUANT": r["mean_QUANT"], "HEDGE": r["mean_HEDGE"]}
              for r in gm[gm.dimension == dim].to_dict(orient="records")}
        for dim in gm["dimension"].unique() if dim != "all"}
    rows, mids = [], []
    for dim, lab in (("year", "Year"), ("country", "Country"), ("doctype", "Document type"),
                     ("industry", "Industry (ICB)")):
        sub = gm[gm.dimension == dim]
        sub = sub.sort_values("group") if dim == "year" else sub.sort_values("n_docs", ascending=False)
        mids.append(len(rows))
        rows.append([f"\\multicolumn{{5}}{{@{{}}l}}{{\\textit{{{lab}}}}}"])
        for r in sub.itertuples():
            rows.append([f"\\quad {tex_escape(r.group)}", tint(r.n_firms), tint(r.n_docs),
                         tnum(r.mean_ESGSI, 3), tnum(r.mean_ESGSI_ext, 3)])
    write_tabular(tex / "group_means.tex", "lrrrr",
                  ["", "Firms", "Docs", r"Mean $\ESGSI$", r"Mean $\ESGSIext$"], rows,
                  midrules=[m for m in mids if m > 0], note="medias de los informes del grupo (z)")


def block_b(df, out, tex, S):
    specs = ["SUS_density", "SUS_tfidf_length", "SUS_lagasio", "Breadth"]
    pear = df[specs].corr(method="pearson")
    spear = df[specs].corr(method="spearman")
    save_csv(pear.reset_index().rename(columns={"index": "var"}), out / "sus_corr_pearson.csv")
    save_csv(spear.reset_index().rename(columns={"index": "var"}), out / "sus_corr_spearman.csv")

    zsen = z(df["SEN_Score"])
    rec = zsen - z(df["SUS_density"])
    gap = float(np.abs(rec - df["ESGSI"]).max())
    idx = {"density": df["ESGSI"].to_numpy(float),
           "tfidf_length": zsen - z(df["SUS_tfidf_length"]),
           "lagasio": zsen - z(df["SUS_lagasio"])}
    comp = pd.DataFrame([{"spec": k, **compare_scores(v, idx["density"]), "n": len(v)}
                         for k, v in idx.items()])
    save_csv(comp, out / "index_by_sus_spec.csv")
    ext_cmp = {**compare_scores(df["ESGSI"], df["ESGSI_ext"]), "n": len(df)}
    S["b_especificaciones_sus"] = {
        "pearson": pear.to_dict(), "spearman": spear.to_dict(),
        "recalculo_ESGSI_density_dif_max": gap,
        "indice_por_especificacion": {r["spec"]: r for r in comp.to_dict(orient="records")},
        "ESGSI_vs_ESGSI_ext": ext_cmp,
        "nota": "comparaciones frente a ESGSI(density): r, rho, solapamiento del 10 % superior e inferior",
    }
    names = {"SUS_density": r"$\susdens$", "SUS_tfidf_length": r"$\sustfidf$",
             "SUS_lagasio": r"$\suslag$", "Breadth": r"$\BREADTH$"}
    rows = []
    for i, a in enumerate(specs):
        cells = []
        for j, b in enumerate(specs):
            if i == j:
                cells.append("1")
            elif j > i:          # triangulo superior: Pearson
                cells.append(tnum(pear.loc[a, b], 3))
            else:                # triangulo inferior: Spearman
                cells.append(tnum(spear.loc[a, b], 3))
        rows.append([names[a]] + cells)
    write_tabular(tex / "sus_corr.tex", "lrrrr", [""] + [names[s] for s in specs], rows,
                  note="triangulo superior Pearson, inferior Spearman")
    lab = {"density": r"$\ESGSI(\susdens)$ (adopted)", "tfidf_length": r"$\ESGSI(\sustfidf)$",
           "lagasio": r"$\ESGSI(\suslag)$"}

    def ov(c):
        k = c["k_decil"]
        return [tfrac(round(c["solape_top"] * k), k), tfrac(round(c["solape_bottom"] * k), k)]

    rows = [[lab[r["spec"]], tnum(r["r"], 4), tnum(r["rho"], 4)] + ov(r) for r in comp.to_dict(orient="records")]
    rows.append([r"$\ESGSIext$", tnum(ext_cmp["r"], 4), tnum(ext_cmp["rho"], 4)] + ov(ext_cmp))
    write_tabular(tex / "index_by_sus_spec.tex", "lrrrr",
                  ["Index", r"$r$", r"$\rho$", r"Top 10\,\% shared", r"Bottom 10\,\% shared"], rows,
                  midrules=[3], note="frente a ESGSI(density); decil extremo = 34 de 343 informes")
    return idx


def block_c(df, idx, extra, out, tex, S, n_boot):
    d = df.assign(ESGSI_tfidf_length=idx["tfidf_length"], ESGSI_lagasio=idx["lagasio"],
                  Z_SEN=z(df["SEN_Score"]), Z_SUS=z(df["SUS_density"]),
                  Z_QUANT=z(df["QUANT_Score"]), Z_HEDGE=z(df["HEDGE_Score"]),
                  tokens=extra["tokens"], mentions=extra["mentions"], distinct=extra["distinct"])
    cols = ["ESGSI", "ESGSI_lagasio", "ESGSI_tfidf_length", "ESGSI_ext", "Z_SEN", "Z_SUS",
            "Z_QUANT", "Z_HEDGE", "SUS_density", "SEN_Score", "QUANT_Score", "HEDGE_Score",
            "Breadth", "tokens", "mentions", "distinct"]
    score_cols = ["year", "country", "doctype", "supersector", "industry",
                  "SUS_density", "SUS_tfidf_length", "SUS_lagasio", "Breadth", "SEN_Score",
                  "QUANT_Score", "HEDGE_Score", "ESGSI", "ESGSI_ext", "ESGSI_tfidf_length",
                  "ESGSI_lagasio", "Z_SEN", "Z_SUS", "Z_QUANT", "Z_HEDGE", "tokens", "mentions", "distinct"]
    # Tabla a nivel documento: anonima (la usan las figuras) e interna (con claves).
    save_csv(d[["firm_label"] + score_cols].rename(columns={"firm_label": "firm_id"}), out / "doc_level.csv")
    save_csv(d[KEY + ["firm", "firm_label"] + score_cols], out / "doc_level_internal.csv")
    yr = d.groupby("year")[cols].mean().reset_index()
    save_csv(yr, out / "yearly_means.csv")
    mono = {}
    for c in cols:
        dif = np.diff(yr[c].to_numpy())
        mono[c] = {"decreciente": bool((dif < 0).all()), "creciente": bool((dif > 0).all()),
                   "2018": float(yr[c].iloc[0]), "2024": float(yr[c].iloc[-1]),
                   "cambio_pct_2018_2024": float(100 * (yr[c].iloc[-1] / yr[c].iloc[0] - 1))
                   if yr[c].iloc[0] != 0 else None}
    ci = pd.concat([yearly_ci(d, c) for c in ("ESGSI", "ESGSI_ext", "Z_SEN", "Z_SUS", "Z_QUANT",
                                              "Z_HEDGE", "SUS_density", "SEN_Score")], ignore_index=True)
    save_csv(ci, out / "yearly_ci.csv")

    trends = {c: trend_block(d, c, n_boot, SEED)
              for c in ("ESGSI", "ESGSI_ext", "SEN_Score", "SUS_density", "QUANT_Score",
                        "HEDGE_Score", "Z_SEN", "Z_SUS", "Z_QUANT", "Z_HEDGE", "ESGSI_lagasio")}
    ger = d[d["country"] == "Germany"]
    trends["ESGSI_Alemania"] = trend_block(ger, "ESGSI", n_boot, SEED)
    trends["ESGSI_ext_Alemania"] = trend_block(ger, "ESGSI_ext", n_boot, SEED)
    S["c_temporal"] = {"medias_anuales": yr.set_index("year").to_dict(),
                       "monotonia": mono, "tendencias": trends}

    # --- tablas ---
    years = sorted(d["year"].unique())
    names = [("ESGSI", r"$\ESGSI(\susdens)$ (adopted)", 3), ("ESGSI_tfidf_length", r"$\ESGSI(\sustfidf)$", 3),
             ("ESGSI_lagasio", r"$\ESGSI(\suslag)$", 3), ("ESGSI_ext", r"$\ESGSIext$", 3),
             ("Z_SEN", r"$\Z(\SEN)$", 3), ("Z_SUS", r"$\Z(\susdens)$", 3),
             ("Z_QUANT", r"$\Z(\QUANT)$", 3), ("Z_HEDGE", r"$\Z(\HEDGE)$", 3)]
    rows = [[lab] + [tnum(v, dd) for v in yr[c]] for c, lab, dd in names]
    rows.append(["Documents"] + [str(int(n)) for n in d.groupby("year").size()])
    write_tabular(tex / "yearly_index.tex", "l" + "r" * len(years), [""] + [str(y) for y in years],
                  rows, midrules=[4, 8])
    drv = [("tokens", "Extracted ESG text per report (lemmatised tokens)", 0),
           ("mentions", "ESG mentions per report", 1),
           ("distinct", "Distinct vocabulary entries present per report", 1),
           ("SUS_density", r"$\susdens$ (mentions per 100 words)", 2),
           ("Breadth", r"$\BREADTH$ (effective distinct terms)", 2),
           ("SEN_Score", r"$\SEN$", 3),
           ("QUANT_Score", r"$\QUANT$", 4), ("HEDGE_Score", r"$\HEDGE$", 4)]
    rows = []
    for c, lab, dd in drv:
        a, b = yr[c].iloc[0], yr[c].iloc[-1]
        # SEN esta cerca de cero: un cambio porcentual no significa nada, se da la diferencia
        chg = tnum(b - a, dd, sign=True) if c == "SEN_Score" else tpct(100 * (b / a - 1), 1, sign=True)
        rows.append([lab, tnum(a, dd) if dd else tint(round(a)), tnum(b, dd) if dd else tint(round(b)), chg])
    write_tabular(tex / "yearly_drivers.tex", "lrrr", ["Quantity", str(years[0]), str(years[-1]), "Change"], rows)

    rows = []
    lab = {"ESGSI": r"$\ESGSI$", "ESGSI_ext": r"$\ESGSIext$", "SEN_Score": r"$\SEN$",
           "SUS_density": r"$\susdens$", "ESGSI_lagasio": r"$\ESGSI(\suslag)$",
           "ESGSI_Alemania": r"$\ESGSI$, Germany only", "ESGSI_ext_Alemania": r"$\ESGSIext$, Germany only"}
    dd = {"SEN_Score": 4}
    for k in lab:
        t, q = trends[k], dd.get(k, 3)
        rows.append([lab[k], f"{t['n']}/{t['firmas']}",
                     tnum(t["mco_convencional"]["pendiente"], q), tp(t["mco_convencional"]["p"]),
                     tnum(t["mco_cluster"]["se"], q), tp(t["mco_cluster"]["p"]),
                     tnum(t["ef_cluster"]["pendiente"], q), tnum(t["ef_cluster"]["se"], q),
                     tp(t["ef_cluster"]["p"]), tpw(t["ef_wild_bootstrap"]["p_wild"], n_boot)])
    write_tabular(tex / "trend_regressions.tex", "lrrrrrrrrr",
                  ["", "Docs/firms", "Slope", r"$p_{\mathrm{OLS}}$", r"SE$_{\mathrm{cl}}$",
                   r"$p_{\mathrm{cl}}$", r"Slope$_{\mathrm{FE}}$", r"SE$_{\mathrm{cl}}$",
                   r"$p_{\mathrm{cl}}$", r"$p_{\mathrm{wild}}$"], rows, midrules=[2, 5],
                  note="MCO agrupada (convencional y cluster por empresa); EF de empresa con cluster; wild cluster bootstrap Rademacher WCR")
    return d


def variance_block(d: pd.DataFrame, fm: pd.DataFrame, y: str) -> tuple[dict, pd.DataFrame, list]:
    """Descomposicion de la varianza de un score (documento y medias de empresa), ICC y
    comparacion intra-pais."""
    seqs = {"industria_primero": ["C(year)", "C(industry)", "C(country)", "C(doctype)", "C(firm)"],
            "pais_primero": ["C(year)", "C(country)", "C(industry)", "C(doctype)", "C(firm)"]}
    dec_rows = []
    for name, terms in seqs.items():
        prev = 0.0
        for k in range(1, len(terms) + 1):
            f = f"{y} ~ " + " + ".join(terms[:k])
            # los efectos de empresa anidan pais, industria y tipo: el diseno es
            # deficiente en rango a proposito; el R2 no depende de ello.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = smf.ols(f, data=d).fit()
            dec_rows.append({"variable": y, "secuencia": name, "paso": terms[k - 1], "r2": m.rsquared,
                             "r2_adj": m.rsquared_adj, "delta_r2": m.rsquared - prev,
                             "k_params": int(m.df_model)})
            prev = m.rsquared
    dec = pd.DataFrame(dec_rows)
    m2 = smf.ols(f"{y} ~ C(year) + C(industry) + C(country)", data=d).fit()
    an = sm.stats.anova_lm(m2, typ=2)
    ssr = an.loc["Residual", "sum_sq"]
    eta = {i.replace("C(", "").replace(")", ""): float(an.loc[i, "sum_sq"] / (an.loc[i, "sum_sq"] + ssr))
           for i in an.index if i != "Residual"}
    anova_p = {i.replace("C(", "").replace(")", ""): float(an.loc[i, "PR(>F)"])
               for i in an.index if i != "Residual"}

    # Entre empresas: medias de empresa (panel equilibrado: el efecto ano se cancela).
    between = {}
    for lab, f in (("industria", "C(industry)"), ("pais", "C(country)"),
                   ("supersector", "C(supersector)"), ("tipo_documento", "C(doctype)"),
                   ("industria+pais", "C(industry) + C(country)")):
        m = smf.ols(f"{y} ~ {f}", data=fm).fit()
        between[lab] = {"r2": m.rsquared, "r2_adj": m.rsquared_adj, "F_p": float(m.f_pvalue),
                        "k": int(m.df_model)}
    between["industria_dado_pais_delta_r2"] = between["industria+pais"]["r2"] - between["pais"]["r2"]
    between["pais_dado_industria_delta_r2"] = between["industria+pais"]["r2"] - between["industria"]["r2"]

    # ICC: parte de la varianza (sin efecto ano) que esta entre empresas.
    yd = d[y] - d.groupby("year")[y].transform("mean")
    k = d.groupby("firm").size().iloc[0]
    G = d["firm"].nunique()
    fmean = yd.groupby(d["firm"]).transform("mean")
    msb = k * ((yd.groupby(d["firm"]).mean() - yd.mean()) ** 2).sum() / (G - 1)
    msw = ((yd - fmean) ** 2).sum() / (len(d) - G)
    icc1 = (msb - msw) / (msb + (k - 1) * msw)
    ss_between_share = float(((fmean - yd.mean()) ** 2).sum() / ((yd - yd.mean()) ** 2).sum())
    yraw = d[y]
    ss_between_raw = float(((yraw.groupby(d["firm"]).transform("mean") - yraw.mean()) ** 2).sum()
                           / ((yraw - yraw.mean()) ** 2).sum())
    ss_year_raw = float(((yraw.groupby(d["year"]).transform("mean") - yraw.mean()) ** 2).sum()
                        / ((yraw - yraw.mean()) ** 2).sum())
    var = {"secuencial": dec.to_dict(orient="records"),
           "eta2_parcial_ano_industria_pais": eta, "anova_tipo2_p": anova_p,
           "entre_empresas_medias": between, "icc1_sin_efecto_ano": float(icc1),
           "cuota_ss_entre_empresas_sin_ano": ss_between_share,
           "cuota_ss_entre_empresas_bruta": ss_between_raw,
           "cuota_ss_entre_anos_bruta": ss_year_raw}

    # Comparacion intra-pais: industrias dentro de Francia y de Alemania.
    intra, intra_rows = {}, []
    for ctry in ("France", "Germany"):
        sub = fm[fm["country"] == ctry]
        m = smf.ols(f"{y} ~ C(industry)", data=sub).fit()
        g = sub.groupby("industry")[y].agg(["mean", "size"]).reset_index()
        g.insert(0, "country", ctry)
        g.insert(0, "variable", y)
        intra_rows.append(g)
        intra[ctry] = {"firmas": len(sub), "industrias": int(sub["industry"].nunique()),
                       "r2": m.rsquared, "r2_adj": m.rsquared_adj, "F_p": float(m.f_pvalue),
                       "medias": g.set_index("industry")["mean"].to_dict(),
                       "firmas_por_industria": g.set_index("industry")["size"].to_dict(),
                       "media_pais": float(sub[y].mean())}
    return {"varianza": var, "intra_pais": intra}, dec, intra_rows


def block_d(d, out, tex, S):
    res: dict = {}
    fm = d.groupby("firm").agg(ESGSI=("ESGSI", "mean"), ESGSI_ext=("ESGSI_ext", "mean"),
                               industry=("industry", "first"), country=("country", "first"),
                               supersector=("supersector", "first"),
                               doctype=("doctype", lambda s: s.mode().iloc[0])).reset_index()
    slopes_all, decs, intra_all = [], [], []
    for y in SCORES:
        r_y: dict = {}
        # el tipo de documento no es constante dentro de la empresa en todos los casos: sin pendientes
        for grp in ("industry", "country", "supersector"):
            sl, eq = group_slopes(d, y, grp)
            sl.insert(0, "dimension", grp)
            sl.insert(0, "variable", y)
            sl = sl.rename(columns={grp: "group"})
            slopes_all.append(sl)
            r_y[f"pendientes_{grp}"] = {"grupos": sl.drop(columns=["dimension", "variable"]).to_dict(orient="records"),
                                        "igualdad_de_pendientes": eq}
        vb, dec, intra_rows = variance_block(d, fm, y)
        r_y.update(vb)
        decs.append(dec)
        intra_all += intra_rows
        res[y] = r_y
    slopes = pd.concat(slopes_all, ignore_index=True)
    save_csv(slopes, out / "group_slopes.csv")
    dec = pd.concat(decs, ignore_index=True)
    save_csv(dec, out / "variance_sequential.csv")
    save_csv(pd.concat(intra_all), out / "within_country_industry.csv")

    # Tipo de documento: constancia dentro de la empresa y confusion con el pais.
    dt_const = d.groupby("firm_label")["doctype"].nunique()
    ct = pd.crosstab(d["country"], d["doctype"])
    save_csv(ct.reset_index(), out / "country_x_doctype.csv")
    ci = pd.crosstab(d.drop_duplicates("firm")["industry"], d.drop_duplicates("firm")["country"])
    save_csv(ci.reset_index(), out / "industry_x_country_firms.csv")
    res["confusion_pais_tipo"] = {"tabla": ct.to_dict(),
                                  "empresas_con_mas_de_un_tipo": dt_const[dt_const > 1].index.tolist()}
    res["empresas_industria_x_pais"] = ci.to_dict()
    S["d_transversal"] = res

    # --- tablas ---
    gm = pd.read_csv(out / "group_means.csv", sep=";")
    for grp, fname in (("industry", "cross_industry.tex"), ("country", "cross_country.tex")):
        sub = gm[gm.dimension == grp]
        for y in SCORES:
            s = slopes[(slopes.dimension == grp) & (slopes.variable == y)][["group", "slope", "ci_lo", "ci_hi"]]
            sub = sub.merge(s.rename(columns={c: f"{c}_{y}" for c in ("slope", "ci_lo", "ci_hi")}), on="group")
        sub = sub.sort_values("mean_ESGSI", ascending=False)
        # dagger: menos de 5 empresas (clusters); el IC agrupado es poco fiable ahi
        rows = []
        for r in sub.to_dict(orient="records"):
            row = [tex_escape(r["group"]) + (r"$^\dagger$" if r["n_firms"] < 5 else ""),
                   tint(r["n_firms"]), tint(r["n_docs"])]
            for y in SCORES:
                row += [tnum(r[f"mean_{y}"], 3), tnum(r[f"slope_{y}"], 3),
                        f"[{tnum(r[f'ci_lo_{y}'], 3)}, {tnum(r[f'ci_hi_{y}'], 3)}]"]
            rows.append(row)
        write_tabular(tex / fname, "lrrrrrrrr",
                      ["", "Firms", "Docs", r"$\ESGSI$", "Slope", r"95\% CI (cluster)",
                       r"$\ESGSIext$", "Slope", r"95\% CI (cluster)"], rows,
                      note="medias de los informes; pendiente con EF de empresa, cluster por empresa (G-1 gl); dagger = menos de 5 empresas")
    step = {"C(year)": "Year", "C(industry)": "+ Industry", "C(country)": "+ Country",
            "C(doctype)": "+ Document type", "C(firm)": "+ Firm"}
    a_ = dec[(dec.secuencia == "industria_primero") & (dec.variable == "ESGSI")].reset_index(drop=True)
    b_ = dec[(dec.secuencia == "industria_primero") & (dec.variable == "ESGSI_ext")].reset_index(drop=True)
    rows = [[step[ra.paso], str(ra.k_params), tnum(ra.r2, 3), tnum(ra.r2_adj, 3), tnum(ra.delta_r2, 3, sign=True),
             tnum(rb.r2, 3), tnum(rb.r2_adj, 3), tnum(rb.delta_r2, 3, sign=True)]
            for ra, rb in zip(a_.itertuples(), b_.itertuples())]
    write_tabular(tex / "variance_sequential.tex", "lrrrrrrr",
                  ["Model", "Params", r"$R^2$", r"Adj.\ $R^2$", r"$\Delta R^2$",
                   r"$R^2$", r"Adj.\ $R^2$", r"$\Delta R^2$"], rows,
                  note="columnas 3-5: ESGSI; 6-8: ESGSI_ext; nivel documento, efectos anadidos en orden")
    rows = []
    for k, lab in (("industria", "Industry"), ("supersector", "Supersector"), ("pais", "Country"),
                   ("tipo_documento", "Document type"), ("industria+pais", "Industry + country")):
        row = [lab, str(res["ESGSI"]["varianza"]["entre_empresas_medias"][k]["k"])]
        for y in SCORES:
            b = res[y]["varianza"]["entre_empresas_medias"][k]
            row += [tnum(b["r2"], 3), tnum(b["r2_adj"], 3), tp(b["F_p"])]
        rows.append(row)
    write_tabular(tex / "variance_between_firms.tex", "lrrrrrrr",
                  ["Firm means on", "Params", r"$R^2$", r"Adj.\ $R^2$", r"$p$ ($F$)",
                   r"$R^2$", r"Adj.\ $R^2$", r"$p$ ($F$)"], rows,
                  note="49 medias de empresa; columnas 3-5: ESGSI; 6-8: ESGSI_ext")


def block_e(d, out, tex, S):
    agg = {"firm_label": ("firm_label", "first"), "country": ("country", "first"),
           "industry": ("industry", "first"), "supersector": ("supersector", "first"),
           "SUS_density": ("SUS_density", "mean"), "SEN": ("SEN_Score", "mean"),
           "QUANT": ("QUANT_Score", "mean"), "HEDGE": ("HEDGE_Score", "mean")}
    for y in SCORES:
        agg[y] = (y, "mean")
        agg[f"sd_{y}"] = (y, "std")
        agg[f"min_{y}"] = (y, "min")
        agg[f"max_{y}"] = (y, "max")
    fm = d.groupby("firm").agg(**agg).reset_index()
    for y in SCORES:
        fm = fm.merge(firm_slopes(d, y).rename(columns={"slope": f"slope_{y}"}), on="firm")
    fm = fm.merge(firm_slopes(d, "SUS_density").rename(columns={"slope": "slope_SUS"}), on="firm")
    fm = fm.merge(firm_slopes(d, "SEN_Score").rename(columns={"slope": "slope_SEN"}), on="firm")
    fm = fm.sort_values("ESGSI", ascending=False).reset_index(drop=True)
    fm["rank_ESGSI"] = np.arange(1, len(fm) + 1)
    fm["rank_ESGSI_ext"] = fm["ESGSI_ext"].rank(ascending=False, method="first").astype(int)
    save_csv(fm[["firm", "firm_label"]].rename(columns={"firm_label": "firm_id"}),
             out / "firm_ids_internal.csv")
    save_csv(fm.drop(columns="firm").rename(columns={"firm_label": "firm_id"}), out / "firm_level.csv")

    res: dict = {"identificador": "'<Industria>-NN': NN = posicion por ESGSI medio 2018-2024 dentro de la industria ICB"}
    for y in SCORES:
        wide = d.pivot(index="firm", columns="year", values=y)
        years = sorted(wide.columns)
        pers = {f"{a}-{b}": float(stats.spearmanr(wide[a], wide[b])[0]) for a, b in zip(years[:-1], years[1:])}
        pers[f"{years[0]}-{years[-1]}"] = float(stats.spearmanr(wide[years[0]], wide[years[-1]])[0])
        res[y] = {
            "persistencia_spearman": pers,
            "persistencia_consecutiva_media": float(np.mean([v for k, v in pers.items()
                                                              if k != f"{years[0]}-{years[-1]}"])),
            "empresas_pendiente_negativa": int((fm[f"slope_{y}"] < 0).sum()), "empresas": len(fm),
            "pendiente_empresa_mediana": float(fm[f"slope_{y}"].median()),
            "dispersion": {"sd_medias_empresa": float(fm[y].std(ddof=1)),
                           "min_media_empresa": float(fm[y].min()), "max_media_empresa": float(fm[y].max()),
                           "sd_intra_empresa_media": float(fm[f"sd_{y}"].mean()),
                           "sd_pendientes_empresa": float(fm[f"slope_{y}"].std(ddof=1)),
                           "rango_anual_medio": float((fm[f"max_{y}"] - fm[f"min_{y}"]).mean())},
            "por_industria": {ind: {"firmas": len(g), "min": float(g[y].min()), "mediana": float(g[y].median()),
                                    "max": float(g[y].max()),
                                    "pendiente_negativa": int((g[f"slope_{y}"] < 0).sum())}
                              for ind, g in fm.groupby("industry")},
        }
    res["medias_empresa_ESGSI_vs_ext"] = compare_scores(fm["ESGSI"], fm["ESGSI_ext"])
    res["pendientes_empresa_ESGSI_vs_ext"] = {
        "r": float(np.corrcoef(fm["slope_ESGSI"], fm["slope_ESGSI_ext"])[0, 1]),
        "rho": float(stats.spearmanr(fm["slope_ESGSI"], fm["slope_ESGSI_ext"])[0])}
    keep = ["firm_label", "industry", "ESGSI", "ESGSI_ext", "slope_ESGSI", "slope_ESGSI_ext"]
    res["top10"] = fm.head(10)[keep].rename(columns={"firm_label": "firm_id"}).to_dict(orient="records")
    res["bottom10"] = fm.tail(10)[keep].rename(columns={"firm_label": "firm_id"}).to_dict(orient="records")
    S["e_empresas"] = res

    rows = []
    for part in (fm.head(10), fm.tail(10)):
        for r in part.itertuples():
            rows.append([str(r.rank_ESGSI), tex_escape(r.firm_label),
                         tnum(r.ESGSI, 3), tnum(r.ESGSI_ext, 3), tnum(r.SEN, 3), tnum(r.SUS_density, 2),
                         tnum(r.slope_ESGSI, 3), tnum(r.slope_ESGSI_ext, 3)])
    write_tabular(tex / "firms_top_bottom.tex", "rlrrrrrr",
                  ["Rank", "Firm (anonymous)", r"$\ESGSI$", r"$\ESGSIext$", r"$\SEN$", r"$\susdens$",
                   r"Slope $\ESGSI$", r"Slope $\ESGSIext$"], rows, midrules=[10],
                  note="medias 2018-2024 por empresa; pendiente MCO por empresa; identificador <Industria>-NN")
    rows = []
    order = fm.groupby("industry")["ESGSI"].mean().sort_values(ascending=False).index
    for ind in order:
        g = fm[fm.industry == ind]
        row = [tex_escape(ind), tint(len(g))]
        for y in SCORES:
            row += [tnum(g[y].min(), 2), tnum(g[y].median(), 2), tnum(g[y].max(), 2),
                    tfrac((g[f"slope_{y}"] < 0).sum(), len(g))]
        rows.append(row)
    write_tabular(tex / "firm_dispersion.tex", "lrrrrrrrrr",
                  ["Industry", "Firms", "Min", "Median", "Max", r"Slope $<0$",
                   "Min", "Median", "Max", r"Slope $<0$"], rows,
                  note="medias de empresa 2018-2024; columnas 3-6: ESGSI; 7-10: ESGSI_ext; pendiente MCO por empresa")
    return fm


def block_f(d, C, cols, pillar, sectoral, regime_lemmas, out, tex, S, n_boot):
    col = np.array(cols)
    P = np.array([pillar.get(t, "?") for t in cols])
    is_sec = np.array([t in sectoral for t in cols])
    tokens = d["tokens"].to_numpy(float)
    total = C.sum(axis=1)

    # recuentos por pilar y documento
    pdoc = pd.DataFrame({f"m_{p}": C[:, P == p].sum(axis=1) for p in ("E", "S", "G", "TRANS")})
    pdoc["m_sectoral"] = C[:, is_sec].sum(axis=1)
    pdoc["m_total"] = total
    for p in ("E", "S", "G", "TRANS", "sectoral"):
        pdoc[f"dens_{p}"] = pdoc[f"m_{p}"] / tokens * 100
    pdoc = pd.concat([d[["firm_label", "year", "country", "industry", "supersector"]]
                      .rename(columns={"firm_label": "firm_id"}).reset_index(drop=True), pdoc], axis=1)
    save_csv(pdoc, out / "pillar_by_doc.csv")

    def shares(by):
        g = pdoc.groupby(by)[[f"m_{p}" for p in ("E", "S", "G", "TRANS", "sectoral")] + ["m_total"]].sum()
        sh = g.div(g["m_total"], axis=0).drop(columns="m_total")
        sh.columns = [c.replace("m_", "share_") for c in sh.columns]
        dn = pdoc.groupby(by)[[f"dens_{p}" for p in ("E", "S", "G", "TRANS", "sectoral")]].mean()
        return sh.join(dn).reset_index()

    sh_all = {}
    for by in ("year", "industry", "country"):
        t = shares(by)
        save_csv(t, out / f"pillar_by_{by}.csv")
        sh_all[by] = t.set_index(by).to_dict(orient="index")
    tot_p = {p: float(C[:, P == p].sum() / C.sum()) for p in ("E", "S", "G", "TRANS")}
    tot_p["sectoral"] = float(C[:, is_sec].sum() / C.sum())

    # terminos por masa
    mass = C.sum(axis=0)
    firm_codes = d["firm"].to_numpy()
    ndocs = (C > 0).sum(axis=0)
    nfirms = np.array([len(set(firm_codes[C[:, j] > 0])) for j in range(C.shape[1])])
    terms = pd.DataFrame({"term": col, "pillar": P, "sectoral": is_sec, "mass": mass.astype(int),
                          "share": mass / mass.sum(), "docs": ndocs, "firms": nfirms})
    terms = terms.sort_values("mass", ascending=False).reset_index(drop=True)
    terms["cum_share"] = terms["share"].cumsum()
    save_csv(terms, out / "term_mass.csv")
    zero_terms = terms.loc[terms["mass"] == 0, "term"].tolist()

    # terminos distintivos por industria: cuota en la industria / cuota global,
    # con soporte minimo: >= 50 menciones en la industria y presentes en >= 2
    # empresas de la industria (todas las industrias tienen al menos 2).
    dist_rows = []
    share_all = mass / mass.sum()
    for ind, idx in d.groupby("industry").groups.items():
        rows_i = d.index.get_indexer(idx)
        mi = C[rows_i].sum(axis=0)
        fi = np.array([len(set(firm_codes[rows_i][C[rows_i, j] > 0])) for j in range(C.shape[1])])
        with np.errstate(divide="ignore", invalid="ignore"):
            lift = (mi / mi.sum()) / share_all
        ok = (mi >= 50) & (fi >= 2)
        order = np.argsort(-np.where(ok, lift, -1))[:5]
        for j in order:
            if ok[j]:
                dist_rows.append({"industry": ind, "term": col[j], "pillar": P[j], "lift": lift[j],
                                  "mass_in_industry": int(mi[j]), "firms_in_industry": int(fi[j]),
                                  "n_firms_industry": int(d.loc[idx, "firm"].nunique()),
                                  "share_in_industry": mi[j] / mi.sum(), "share_overall": share_all[j]})
    dist = pd.DataFrame(dist_rows)
    save_csv(dist, out / "distinctive_terms_by_industry.csv")

    # Tendencia sin las entradas de marcos/regimenes (ESGSI y ESGSI_ext).
    zsen = z(d["SEN_Score"])
    ext_part = -EXT_DEFAULT[0] * z(d["QUANT_Score"]) + EXT_DEFAULT[1] * z(d["HEDGE_Score"])
    full_idx = zsen - z(total / tokens * 100)
    gap = float(np.abs(full_idx - d["ESGSI"].to_numpy()).max())
    reg = {v for v in regime_lemmas.values() if v}
    variants = {
        "vocabulario completo": np.ones(len(cols), bool),
        "sin regimenes (lista explicita)": ~np.isin(col, list(reg)),
        "sin pilar TRANS": P != "TRANS",
        "sin TRANS ni regimenes": (P != "TRANS") & ~np.isin(col, list(reg)),
    }
    fw_rows = []
    fw_json = {}
    for name, keep in variants.items():
        dens = C[:, keep].sum(axis=1) / tokens * 100
        idx_v = zsen - z(dens)
        dv = d.assign(_idx=idx_v, _ext=idx_v + ext_part, _dens=dens)
        tb = trend_block(dv, "_idx", n_boot, SEED)
        tbe = trend_block(dv, "_ext", n_boot, SEED)
        ym = dv.groupby("year")["_dens"].mean()
        cmp_ = compare_scores(idx_v, full_idx)
        r = {"variante": name, "entradas": int(keep.sum()),
             "masa_quitada": float(1 - C[:, keep].sum() / C.sum()),
             "densidad_2018": float(ym.iloc[0]), "densidad_2024": float(ym.iloc[-1]),
             "densidad_cambio_pct": float(100 * (ym.iloc[-1] / ym.iloc[0] - 1)),
             "r": cmp_["r"], "rho": cmp_["rho"], "solape_top": cmp_["solape_top"],
             "solape_bottom": cmp_["solape_bottom"],
             "pendiente_mco_cluster": tb["mco_cluster"]["pendiente"], "p_mco_cluster": tb["mco_cluster"]["p"],
             "pendiente_ef": tb["ef_cluster"]["pendiente"], "se_ef": tb["ef_cluster"]["se"],
             "p_ef": tb["ef_cluster"]["p"], "ci_ef": tb["ef_cluster"]["ci95"],
             "p_wild": tb["ef_wild_bootstrap"]["p_wild"],
             "ext_pendiente_ef": tbe["ef_cluster"]["pendiente"], "ext_se_ef": tbe["ef_cluster"]["se"],
             "ext_p_ef": tbe["ef_cluster"]["p"], "ext_p_wild": tbe["ef_wild_bootstrap"]["p_wild"]}
        fw_rows.append(r)
        fw_json[name] = r
    fw = pd.DataFrame(fw_rows)
    save_csv(fw, out / "frameworks_excluded_trend.csv")

    S["f_pilares"] = {
        "cuota_menciones_total": tot_p, "cuotas_y_densidades": sh_all,
        "menciones_totales": int(C.sum()),
        "top15_terminos": terms.head(15).to_dict(orient="records"),
        "top15_cuota_acumulada": float(terms["cum_share"].iloc[14]),
        "entradas_con_masa_cero": zero_terms,
        "distintivos_por_industria_regla": "lift = cuota en la industria / cuota global; masa >= 50 y >= 2 empresas",
        "recalculo_ESGSI_dif_max": gap,
        "sin_marcos": fw_json,
    }

    # --- tablas ---
    rows = []
    for by in ("year", "industry"):
        t = pd.read_csv(out / f"pillar_by_{by}.csv", sep=";")
        for r in t.itertuples():
            rows.append([str(getattr(r, by)) if by == "year" else tex_escape(getattr(r, by))] +
                        [tnum(100 * getattr(r, f"share_{p}"), 1) for p in ("E", "S", "G", "TRANS")] +
                        [tnum(getattr(r, f"dens_{p}"), 2) for p in ("E", "S", "G", "TRANS")])
    rows.append(["All"] + [tnum(100 * tot_p[p], 1) for p in ("E", "S", "G", "TRANS")] +
                [tnum(pdoc[f"dens_{p}"].mean(), 2) for p in ("E", "S", "G", "TRANS")])
    nyear = pdoc["year"].nunique()
    write_tabular(tex / "pillar_shares.tex", "lrrrrrrrr",
                  ["", r"E \%", r"S \%", r"G \%", r"TRANS \%", "E dens.", "S dens.", "G dens.", "TRANS dens."],
                  rows, midrules=[nyear, len(rows) - 1],
                  note="cuota de menciones (sobre la masa del grupo) y densidad media por 100 palabras")
    rows = [[str(i + 1), tex_escape(r.term), r.pillar, tint(r.mass), tnum(100 * r.share, 2),
             tnum(100 * r.cum_share, 1), f"{r.docs}/{len(d)}", f"{r.firms}/49"]
            for i, r in enumerate(terms.head(15).itertuples())]
    write_tabular(tex / "top_terms.tex", "rllrrrrr",
                  ["", "Entry (lemmatised)", "Pillar", "Mentions", r"Share \%", r"Cum.\ \%", "Docs", "Firms"], rows)
    rows = [[tex_escape(r.industry), tex_escape(r.term), r.pillar, tnum(r.lift, 1), tint(r.mass_in_industry),
             f"{r.firms_in_industry}/{r.n_firms_industry}"] for r in dist.groupby("industry").head(3).itertuples()]
    write_tabular(tex / "distinctive_terms.tex", "lllrrr",
                  ["Industry", "Entry", "Pillar", "Lift", "Mentions", "Firms"], rows,
                  note="tres primeros por industria; lift = cuota en la industria / cuota global")
    rows = [[r.variante.replace("vocabulario completo", "Full vocabulary")
             .replace("sin regimenes (lista explicita)", "Without regime/framework names")
             .replace("sin pilar TRANS", "Without TRANS pillar")
             .replace("sin TRANS ni regimenes", "Without TRANS and regime names"),
             str(r.entradas), tpct(100 * r.masa_quitada, 1),
             f"{tnum(r.densidad_2018, 2)} $\\to$ {tnum(r.densidad_2024, 2)}",
             tnum(r.rho, 3), tnum(r.pendiente_ef, 3), tp(r.p_ef), tpw(r.p_wild, n_boot),
             tnum(r.ext_pendiente_ef, 3), tp(r.ext_p_ef)]
            for r in fw.itertuples()]
    write_tabular(tex / "frameworks_excluded.tex", "lrrrrrrrrr",
                  ["Vocabulary", "Entries", "Mass removed", r"$\susdens$ 2018 $\to$ 2024", r"$\rho$",
                   r"Slope$_{\mathrm{FE}}$", r"$p_{\mathrm{cl}}$", r"$p_{\mathrm{wild}}$",
                   r"Slope$_{\mathrm{FE}}$ $\ESGSIext$", r"$p_{\mathrm{cl}}$"], rows,
                  note="rho de Spearman frente al ESGSI con el vocabulario completo; pendientes con EF de empresa y cluster")
    return terms


def pdf_wordcounts(pdf_dir: Path, cache: Path) -> pd.DataFrame:
    """Palabras del texto completo de cada PDF (fitz, page.get_text()). Cacheado."""
    import fitz
    done = pd.read_csv(cache, sep=";") if cache.exists() else pd.DataFrame(columns=["rel_path", "pages", "words"])
    have = set(done["rel_path"])
    pdfs = sorted(p for p in pdf_dir.rglob("*") if p.suffix.lower() == ".pdf")
    new = []
    t0 = time.time()
    for i, p in enumerate(pdfs, 1):
        rel = nfc(p.relative_to(pdf_dir).as_posix())
        if rel in have:
            continue
        with fitz.open(p) as doc:
            words = sum(len(page.get_text().split()) for page in doc)
            new.append({"rel_path": rel, "pages": doc.page_count, "words": words})
        if len(new) % 20 == 0:
            print(f"  pdf {i}/{len(pdfs)} ({time.time() - t0:.0f}s)", flush=True)
            pd.concat([done, pd.DataFrame(new)]).to_csv(cache, sep=";", index=False)
    if new:
        done = pd.concat([done, pd.DataFrame(new)], ignore_index=True)
        done.to_csv(cache, sep=";", index=False)
    return done


def block_g(d, dedup_info, chunks_dir: Path, pdf_dir: Path, pdf_cache: Path, out, tex, S, skip_pdf: bool):
    rows = []
    for p in sorted(chunks_dir.rglob("*.json")):
        with open(p, encoding="utf-8") as fh:
            js = json.load(fh)
        zones = js.get("zones", [])
        zw = [z_.get("word_count", len(z_.get("text", "").split())) for z_ in zones]
        rows.append({"rel_json": nfc(p.relative_to(chunks_dir).as_posix()),
                     "País": nfc(p.parent.parent.name), "Compañía": nfc(p.parent.name),
                     "Documento": nfc(p.name), "zones": len(zones), "zone_words": int(sum(zw)),
                     "extracted_words": len(js.get("relevant_text", "").split()),
                     "pdf_name": js.get("file", p.stem + ".pdf")})
    ex = pd.DataFrame(rows)
    ex["rel_pdf"] = ex["País"] + "/" + ex["Compañía"] + "/" + ex["pdf_name"].map(nfc)
    dups = {r["descartado"] for r in dedup_info[1:]}
    ex["duplicate"] = ex["Documento"].isin(dups)
    key3 = ["Documento", "País", "Compañía"]
    ex = ex.merge(d[key3 + ["year", "firm"]], on=key3, how="left")
    n_json = len(ex)
    unmatched = ex.loc[ex["year"].isna() & ~ex["duplicate"], "rel_json"].tolist()
    if not skip_pdf:
        wc = pdf_wordcounts(pdf_dir, pdf_cache)
        wc["rel_path"] = wc["rel_path"].map(nfc)
        ex = ex.merge(wc.rename(columns={"rel_path": "rel_pdf", "words": "pdf_words", "pages": "pdf_pages"}),
                      on="rel_pdf", how="left")
        ex["yield"] = ex["extracted_words"] / ex["pdf_words"]
    # con rutas y carpetas de empresa: solo interno (lo lee scripts/extraction_stats_v3.py)
    save_csv(ex, out / "extraction_by_doc_internal.csv")
    u = ex[~ex["duplicate"] & ex["year"].notna()].copy()
    u["year"] = u["year"].astype(int)
    u["words_per_zone"] = u["extracted_words"] / u["zones"].clip(lower=1)
    agg_cols = ["zones", "extracted_words", "words_per_zone"] + (["pdf_words", "pdf_pages", "yield"] if not skip_pdf else [])
    by_year = u.groupby("year")[agg_cols].agg(["mean", "median"])
    by_year.columns = [f"{a}_{b}" for a, b in by_year.columns]
    by_year = by_year.reset_index()
    save_csv(by_year, out / "extraction_by_year.csv")
    internal = {"json_duplicados_excluidos": ex.loc[ex["duplicate"], "rel_json"].tolist(),
                "json_sin_cruce_con_results": unmatched}
    g = {"json_encontrados": n_json, "pdf_encontrados": int(sum(1 for p in pdf_dir.rglob("*") if p.suffix.lower() == ".pdf")),
         "json_duplicados_excluidos": len(internal["json_duplicados_excluidos"]),
         "json_sin_cruce_con_results": len(unmatched),
         "documentos_analizados": len(u), "nota_duplicado": KNOWN_DUPLICATE,
         "deduplicacion": anon_dedup(dedup_info, d),
         "zonas_total": int(u["zones"].sum()), "palabras_extraidas_total": int(u["extracted_words"].sum()),
         "palabras_extraidas_total_344": int(ex["extracted_words"].sum()),
         "por_documento": {c: describe(u[c]) for c in agg_cols},
         "palabras_por_zona_global": float(u["extracted_words"].sum() / u["zones"].sum()),
         "por_ano": by_year.set_index("year").to_dict(orient="index")}
    if not skip_pdf:
        internal["pdf_sin_recuento"] = ex.loc[ex["pdf_words"].isna(), "rel_pdf"].tolist()
        g["pdf_sin_recuento"] = len(internal["pdf_sin_recuento"])
        g["rendimiento_agregado"] = float(u["extracted_words"].sum() / u["pdf_words"].sum())
        g["palabras_pdf_total"] = int(u["pdf_words"].sum())
    S["g_extraccion"] = g
    (out / "extraction_internal.json").write_text(
        json.dumps(jsonable({**internal, "deduplicacion": dedup_info}), ensure_ascii=False, indent=1),
        encoding="utf-8")

    rows = []
    for r in by_year.itertuples():
        row = [str(r.year), tnum(r.zones_mean, 1), tint(round(r.extracted_words_mean)),
               tnum(r.words_per_zone_mean, 1)]
        if not skip_pdf:
            row += [tint(round(r.pdf_words_mean)), tpct(100 * r.yield_mean, 1), tpct(100 * r.yield_median, 1)]
        rows.append(row)
    allrow = ["All", tnum(u["zones"].mean(), 1), tint(round(u["extracted_words"].mean())),
              tnum(u["words_per_zone"].mean(), 1)]
    if not skip_pdf:
        allrow += [tint(round(u["pdf_words"].mean())), tpct(100 * u["yield"].mean(), 1),
                   tpct(100 * u["yield"].median(), 1)]
    rows.append(allrow)
    hdr = ["Year", "Zones", "Extracted words", "Words/zone"] + (
        ["Full-text words", "Yield (mean)", "Yield (median)"] if not skip_pdf else [])
    write_tabular(tex / "extraction_by_year.tex", "l" + "r" * (len(hdr) - 1), hdr, rows,
                  midrules=[len(rows) - 1], note=f"medias por informe; n = {len(u)} (sin el duplicado)")


def anon_dedup(dedup_info: list[dict], df: pd.DataFrame) -> list[dict]:
    """La deduplicacion sin nombres: recuentos y, por informe descartado, el
    identificador anonimo de la empresa y el ano (detalle en *_internal.json)."""
    ids = df.drop_duplicates("firm").set_index("firm")["firm_label"].to_dict()
    out = [dedup_info[0]]
    for r in dedup_info[1:]:
        out.append({"empresa": ids.get(nfc(r["empresa"]), "?"), "año": r["año"],
                    "motivo": "clean_text identico (MD5) a otro informe de la misma empresa y ano"})
    return out


def quant_hedge_full(raw_texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """QUANT y HEDGE a precision completa con el analizador real (results.csv los
    redondea a 4 decimales). Solo lectura: no toca data/ ni results.csv."""
    import config
    from esgsi_analyzer import ESGSIAnalyzer
    from loguru import logger
    logger.remove()
    an = ESGSIAnalyzer(keywords=list(config.ESG_KEYWORDS), hedge_words=config.HEDGE_KEYWORDS,
                       quant_patterns=config.QUANT_PATTERNS, ext_weights=config.ESGSI_EXT_WEIGHTS,
                       sus_mode=config.SUS_MODE)
    return an.calculate_quant_scores(raw_texts), an.calculate_hedge_scores(raw_texts)


def block_w(d, out, tex, S):
    """
    Rejilla de pesos del ESGSI extendido: ESGSI_ext(w_q, w_h) = ESGSI - w_q Z(QUANT)
    + w_h Z(HEDGE), con z poblacional sobre QUANT y HEDGE a precision completa.
    Para cada par: Spearman con el ESGSI y con el extendido adoptado (0,5; 0,5),
    solapamiento de deciles extremos con ambos y pendiente con EF de empresa (SE
    agrupados por empresa). (0, 0) es el ESGSI.
    """
    zq, zh = z(d["QUANT_Score"]), z(d["HEDGE_Score"])
    base = d["ESGSI"].to_numpy(float)

    def ext(wq, wh):
        return base - wq * zq + wh * zh

    default = ext(*EXT_DEFAULT)
    gap = float(np.abs(default - d["ESGSI_ext"].to_numpy(float)).max())
    rows = []
    for wq in EXT_WEIGHT_GRID:
        for wh in EXT_WEIGHT_GRID:
            v = ext(wq, wh)
            ce, cd = compare_scores(v, base), compare_scores(v, default)
            fe = fe_slope(d.assign(_v=v), "_v")
            rows.append({"w_q": wq, "w_h": wh,
                         "r_ESGSI": ce["r"], "rho_ESGSI": ce["rho"],
                         "solape_top_ESGSI": ce["solape_top"], "solape_bottom_ESGSI": ce["solape_bottom"],
                         "r_default": cd["r"], "rho_default": cd["rho"],
                         "solape_top_default": cd["solape_top"], "solape_bottom_default": cd["solape_bottom"],
                         "pendiente_ef": fe["pendiente"], "se_ef": fe["se"], "p_ef": fe["p"],
                         "ci_lo": fe["ci95"][0], "ci_hi": fe["ci95"][1]})
    g = pd.DataFrame(rows)
    save_csv(g, out / "ext_weights_grid.csv")
    other = g[~((g.w_q == EXT_DEFAULT[0]) & (g.w_h == EXT_DEFAULT[1]))]
    near = g[(np.abs(g.w_q - EXT_DEFAULT[0]) <= 0.25) & (np.abs(g.w_h - EXT_DEFAULT[1]) <= 0.25)]
    k = compare_scores(base, base)["k_decil"]
    S["w_pesos_ext"] = {
        "definicion": "ESGSI_ext(w_q, w_h) = ESGSI - w_q*Z(QUANT) + w_h*Z(HEDGE); z poblacional",
        "rejilla": list(EXT_WEIGHT_GRID), "adoptado": list(EXT_DEFAULT), "k_decil": k,
        "recalculo_ESGSI_ext_dif_max": gap,
        "filas": g.to_dict(orient="records"),
        "resumen": {
            "rho_default_min": float(other.rho_default.min()),
            "rho_default_min_vecinos_025": float(near.rho_default.min()),
            "rho_ESGSI_min": float(g.rho_ESGSI.min()), "rho_ESGSI_max_sin_00": float(g.rho_ESGSI[g.rho_ESGSI < 1 - 1e-12].max()),
            "rho_ESGSI_adoptado": float(g.loc[(g.w_q == 0.5) & (g.w_h == 0.5), "rho_ESGSI"].iloc[0]),
            "solape_top_default_min": float(other.solape_top_default.min()),
            "solape_bottom_default_min": float(other.solape_bottom_default.min()),
            "pendiente_ef_min": float(g.pendiente_ef.min()), "pendiente_ef_max": float(g.pendiente_ef.max()),
            "pares_p_menor_005": int((g.p_ef < 0.05).sum()), "pares": len(g),
            "pares_pendiente_negativa": int((g.pendiente_ef < 0).sum()),
        },
    }
    trows = [[tnum(r.w_q, 2), tnum(r.w_h, 2), tnum(r.rho_ESGSI, 3), tnum(r.rho_default, 3),
              tfrac(round(r.solape_top_default * k), k), tfrac(round(r.solape_bottom_default * k), k),
              tnum(r.pendiente_ef, 3), tnum(r.se_ef, 3), tp(r.p_ef)] for r in g.itertuples()]
    write_tabular(tex / "ext_weights_grid.tex", "rrrrrrrrr",
                  [r"$w_q$", r"$w_h$", r"$\rho$ ($\ESGSI$)", r"$\rho$ (0.5, 0.5)", r"Top 10\,\%",
                   r"Bottom 10\,\%", r"Slope$_{\mathrm{FE}}$", r"SE$_{\mathrm{cl}}$", r"$p_{\mathrm{cl}}$"],
                  trows, midrules=[5, 10, 15, 20],
                  note="solapamiento de deciles extremos y rho (0.5,0.5) frente al ESGSI_ext adoptado; (0,0) = ESGSI; EF de empresa, cluster por empresa")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=RESULTS_DIR / "metrics_v3_sec" / "results.csv")
    ap.add_argument("--processed", type=Path, default=DATA_DIR / "clean_v3_sec" / "processed_texts.csv")
    ap.add_argument("--chunks", type=Path, default=DATA_DIR / "chunks_lexical_v3_sec")
    ap.add_argument("--pdf-dir", type=Path, default=DATA_DIR / "pdf")
    ap.add_argument("--sectors", type=Path, default=METADATA_DIR / "empresas_supersector.csv")
    ap.add_argument("--terms-v3", type=Path, default=METADATA_DIR / "ESG_terms_v3.csv")
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "analysis_v3")
    ap.add_argument("--pdf-cache", type=Path, default=None,
                    help="cache de palabras por PDF (por defecto <out>/pdf_wordcounts.csv)")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--skip-pdf", action="store_true", help="no contar palabras del PDF completo")
    ap.add_argument("--only-pdf-wordcounts", action="store_true",
                    help="solo rellenar la cache pdf_wordcounts.csv y salir")
    a = ap.parse_args()

    out, tex = a.out, a.out / "tex"
    pdf_cache = a.pdf_cache or out / "pdf_wordcounts.csv"
    tex.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if a.only_pdf_wordcounts:
        out.mkdir(parents=True, exist_ok=True)
        pdf_wordcounts(a.pdf_dir, pdf_cache)
        print(f"cache de palabras PDF lista ({time.time() - t0:.0f}s)")
        return

    S: dict = {"generado": time.strftime("%Y-%m-%d %H:%M:%S"),
               "entradas": {k: str(v) for k, v in vars(a).items()},
               "convenciones": {"zscore": "desviacion tipica poblacional (ddof=0)",
                                "score": "continuo y relativo al corpus; sin umbral ni etiquetas",
                                "comparacion_scores": "Pearson r, Spearman rho, solapamiento del 10 % superior e inferior",
                                "empresas": "identificadores anonimos <Industria>-NN",
                                "cluster": "empresa (49)",
                                "wild_bootstrap": f"Rademacher, WCR, {a.n_boot} reps, semilla {SEED}",
                                "inferencia_cluster": "t con G-1 gl; correccion G/(G-1)*(N-1)/(N-K)"}}

    df = load_results(a.results, a.sectors)
    corpus, dedup_info = load_corpus(a.processed, with_raw=True)
    df = df.merge(corpus, on=KEY, how="left", validate="one_to_one")
    if df["clean_text"].isna().any():
        raise ValueError(f"{df['clean_text'].isna().sum()} filas de results sin clean_text")
    # QUANT y HEDGE a precision completa (results.csv los guarda con 4 decimales).
    q_full, h_full = quant_hedge_full(df["raw_text"].astype(str).tolist())
    chk = {"QUANT_dif_max_vs_redondeado": float(np.abs(np.round(q_full, 4) - df["QUANT_Score"]).max()),
           "HEDGE_dif_max_vs_redondeado": float(np.abs(np.round(h_full, 4) - df["HEDGE_Score"]).max())}
    if max(chk.values()) > 1e-4 + 1e-9:
        raise ValueError(f"QUANT/HEDGE recalculados no reproducen results.csv: {chk}")
    df["QUANT_Score"], df["HEDGE_Score"] = q_full, h_full
    df = df.drop(columns="raw_text")
    S["precision_componentes"] = {**chk, "nota": "QUANT y HEDGE recalculados con ESGSIAnalyzer sobre raw_text"}
    S["corpus"] = {"documentos": len(df), "empresas": int(df["firm"].nunique()),
                   "anos": sorted(df["year"].unique().tolist()),
                   "por_ano": df.groupby("year").size().to_dict(),
                   "por_pais": df["country"].value_counts().to_dict(),
                   "por_tipo": df["doctype"].value_counts().to_dict(),
                   "empresas_por_industria": df.drop_duplicates("firm")["industry"].value_counts().to_dict(),
                   "empresas_por_supersector": df.drop_duplicates("firm")["supersector"].value_counts().to_dict(),
                   "empresas_por_pais": df.drop_duplicates("firm")["country"].value_counts().to_dict(),
                   "deduplicacion": anon_dedup(dedup_info, df)}
    print(f"datos: {len(df)} documentos ({time.time() - t0:.0f}s)", flush=True)

    vocab, pillar, sectoral, lem, vinfo, _ = load_vocab_and_pillars(a.terms_v3)
    save_csv(lem, out / "term_lemma_pillar_map.csv")
    S["vocabulario"] = vinfo
    texts = df["clean_text"].astype(str).tolist()
    C, cols = count_matrix(texts, vocab)
    tokens = np.array([max(len(t.split()), 1) for t in texts], dtype=float)
    dens_check = float(np.abs(C.sum(axis=1) / tokens * 100 - df["SUS_density"].to_numpy()).max())
    S["vocabulario"]["recalculo_SUS_density_dif_max"] = dens_check
    if dens_check > 1e-4:
        print(f"AVISO: el recuento no reproduce SUS_density (dif. max {dens_check:.3g})")
    extra = {"tokens": tokens, "mentions": C.sum(axis=1), "distinct": (C > 0).sum(axis=1)}
    print(f"recuentos {C.shape} ({time.time() - t0:.0f}s)", flush=True)

    block_a(df, out, tex, S)
    idx = block_b(df, out, tex, S)
    d = block_c(df, idx, extra, out, tex, S, a.n_boot)
    print(f"a-c hechos ({time.time() - t0:.0f}s)", flush=True)
    block_d(d, out, tex, S)
    block_e(d, out, tex, S)
    block_f(d, C, cols, pillar, sectoral, vinfo["regimenes_lema"], out, tex, S, a.n_boot)
    block_w(d, out, tex, S)
    print(f"d-f, w hechos ({time.time() - t0:.0f}s)", flush=True)
    block_g(d, dedup_info, a.chunks, a.pdf_dir, pdf_cache, out, tex, S, a.skip_pdf)
    S["segundos"] = round(time.time() - t0)
    (out / "summary.json").write_text(json.dumps(jsonable(S), ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print(f"listo: {out} ({S['segundos']}s)")


if __name__ == "__main__":
    main()
