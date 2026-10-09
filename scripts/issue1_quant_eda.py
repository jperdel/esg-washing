"""
issue1_quant_eda.py
-------------------
EDA de la tendencia temporal del QUANT (issue #1: "Analizar la fuerte
tendencia de QUANT"). La tabla de tendencias del paper da para QUANT una
pendiente de 4,7 (en unidades de 1e-4 por ano), frente a 0,366 del SUS por
densidad o -0,4 del HEDGE. Este script:

  1  reproduce el QUANT de results.csv a precision completa sobre raw_text;
  2  pone las pendientes de todos los componentes en unidades comparables
     (desviaciones tipicas por ano, % de la media por ano) y su aportacion a
     la pendiente del ESGSI extendido;
  3  descompone EXACTAMENTE la pendiente del QUANT por familia de patrones y
     por subtipo (la pendiente con efectos fijos es lineal en y, y el QUANT
     es una suma de densidades: las pendientes de las partes suman la total);
  4  mira si la tendencia se concentra en pocas empresas o informes, si va
     con la longitud del texto extraido, el tipo de documento, el sector o el
     pais;
  5  prototipa un QUANT de "solo cifras" (cada numero contado una vez:
     porcentajes, cantidades con unidad fisica y cifras >= 1.000 que no son
     anos, rangos de anos, codigos de normas ni numeros de actos legales) y
     da su pendiente y la del ESGSI extendido resultante, con Spearman y
     solapamiento de deciles extremos frente a la especificacion actual.

Directrices (paper/_research/directrices.md): el indice es un score continuo
(sin umbrales ni etiquetas); robustez como Spearman, solapamiento de deciles
extremos y pendiente temporal; ninguna salida lleva nombres de empresa (las
empresas son los identificadores anonimos '<Industria>-NN' de analysis_v3).

Entradas (solo lectura; data/ y results/ no estan versionados, por eso
--root permite leerlos de otro clon):
    <root>/data/clean_v3_sec/processed_texts.csv
    <root>/results/analysis_v3/doc_level_internal.csv
    <root>/results/analysis_v3/extraction_by_doc_internal.csv
Salidas: results/issue1_quant/ de ESTE arbol (csv, json, png).

    PYTHONIOENCODING=utf-8 python scripts/issue1_quant_eda.py [--root RUTA]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

KEY = ["Documento", "País", "Compañía", "Año"]
W_QUANT, W_HEDGE = 0.5, 0.5
DECILE_Q = 0.10

# --- Patrones del QUANT actual (copia literal de src/config.py) -------------
QUANT_PATTERNS = {
    "percentages":   r'\b\d+(?:[.,]\d+)?\s*%',
    "large_numbers": r'\b(?!(?:19|20)\d{2}\b)\d{4,}\b',
    "units":         r'\b(?:tonne|ton|mt|ktco2|co2e?|ghg|kwh|mwh|gwh|twh|mw|gw|litre|liter|m3|cubic meter)\b',
    "frameworks":    r'\b(?:gri|tcfd|sasb|issb|sdg|ungc|sfdr|csrd|un global compact|paris agreement|taxonomy)\b',
}
FAMILIES = list(QUANT_PATTERNS)
RX = {k: re.compile(v, re.IGNORECASE) for k, v in QUANT_PATTERNS.items()}

# Subtipos (diagnostico) --------------------------------------------------
TAXO_CTX = re.compile(r"taxonom|eligib", re.IGNORECASE)        # entorno de tablas de taxonomia
CODE_PREFIX = re.compile(r"(?:ISO|OHSAS|FSSC|SA|IATF|EN|IEC|EMAS|BS|NF)\s?[-:]?\s?$", re.IGNORECASE)
LEGAL_PREFIX = re.compile(r"(?:\bL\.|\bR\.|\bArt(?:icle|\.)?|\bNo\.?|\bn[o°]\.?)\s?$", re.IGNORECASE)
YEAR_RANGE = re.compile(r"^(?:19|20)\d{2}(?:19|20)\d{2}$")   # "2021-2023" sin guion (la extraccion quita el '–')
TOPIC_UNITS = {"co2", "co2e", "ghg", "ktco2"}

# --- QUANT de "solo cifras" (prototipo) ---------------------------------------
# Un numero = cifra con separadores de miles o decimales opcionales, no pegado
# a letras ni a '/' (excluye 'FY2023', 'Scope3', '2021/2139').
NUM_RE = re.compile(r"(?<![\w/.,])\d+(?:[.,]\d+)*(?![\w/]|[.,]\d)")
UNIT_AFTER = re.compile(
    r"^\s?(?:%|"
    r"(?:k|m|g)?t\s?co2e?q?\b|(?:k|m|g)?tco2e?q?\b|"
    r"(?:tonnes?|tons?|kt|mt|gt|t|kwh|mwh|gwh|twh|kw|mw|gw|gj|tj|pj|m3|m2|litres?|liters?|hl|kg|km|ha)\b)",
    re.IGNORECASE)


def nfc(x) -> str:
    return unicodedata.normalize("NFC", str(x)).strip()


def z(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    s = np.std(x)
    return np.zeros_like(x) if s == 0 else (x - x.mean()) / s


def decile_overlap(a, b, q=DECILE_Q) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    k = max(int(round(q * len(a))), 1)
    oa, ob = np.argsort(a, kind="stable"), np.argsort(b, kind="stable")
    return {"top": len(set(oa[-k:]) & set(ob[-k:])) / k, "bottom": len(set(oa[:k]) & set(ob[:k])) / k}


def within(v: pd.Series, g: pd.Series) -> np.ndarray:
    return (v - v.groupby(g).transform("mean")).to_numpy(float)


def fe_slope(df: pd.DataFrame, col: str, controls: list[str] | None = None) -> dict:
    """Pendiente en ano con efectos fijos de empresa, SE agrupados por empresa
    (igual que analysis_v3.fe_slope; opcionalmente con controles)."""
    yd = within(df[col], df["firm"])
    X = [within(df["year"].astype(float), df["firm"])]
    for c in controls or []:
        X.append(within(df[c], df["firm"]))
    X = np.column_stack(X)
    m = sm.OLS(yd, X).fit(cov_type="cluster", cov_kwds={"groups": df["firm"].to_numpy()}, use_t=True)
    ci = m.conf_int()[0]
    return {"slope": float(m.params[0]), "se": float(m.bse[0]), "p": float(m.pvalues[0]),
            "ci_lo": float(ci[0]), "ci_hi": float(ci[1]), "r2_within": float(m.rsquared)}


def firm_slopes(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("firm").apply(lambda d: stats.linregress(d["year"], d[col]).slope)


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------

def load(root: Path) -> pd.DataFrame:
    proc = pd.read_csv(root / "data" / "clean_v3_sec" / "processed_texts.csv", sep=";",
                       usecols=KEY + ["clean_text", "raw_text"], encoding="utf-8")
    for c in KEY[:3]:
        proc[c] = proc[c].map(nfc)
    proc["Año"] = proc["Año"].astype(str).str.strip()
    dig = proc["clean_text"].astype(str).map(lambda t: hashlib.md5(t.encode("utf-8")).hexdigest())
    proc = proc[~dig.duplicated()].reset_index(drop=True)          # misma deduplicacion que main.py

    dl = pd.read_csv(root / "results" / "analysis_v3" / "doc_level_internal.csv", sep=";", encoding="utf-8")
    for c in KEY[:3]:
        dl[c] = dl[c].map(nfc)
    dl["Año"] = dl["Año"].astype(str).str.strip()
    df = dl.merge(proc[KEY + ["raw_text"]], on=KEY, how="left", validate="one_to_one")
    if df["raw_text"].isna().any():
        raise ValueError("documentos sin raw_text")

    ex = pd.read_csv(root / "results" / "analysis_v3" / "extraction_by_doc_internal.csv", sep=";", encoding="utf-8")
    for c in ("País", "Compañía", "Documento"):
        ex[c] = ex[c].map(nfc)
    ex = ex[~ex["duplicate"]][["País", "Compañía", "Documento", "pdf_pages", "pdf_words", "zones", "yield"]]
    df = df.merge(ex, on=["País", "Compañía", "Documento"], how="left", validate="one_to_one")
    df["raw_text"] = df["raw_text"].astype(str)
    return df


# ---------------------------------------------------------------------------
# Recuentos por documento
# ---------------------------------------------------------------------------

def classify_large_number(text: str, m: re.Match) -> str:
    s = m.group(0)
    before = text[max(0, m.start() - 12):m.start()]
    after = text[m.end():m.end() + 1]
    if YEAR_RANGE.match(s):
        return "ln_rango_anos"
    if CODE_PREFIX.search(before) or after == ":":
        return "ln_codigo_norma"
    if text[max(0, m.start() - 1):m.start()] == "/" or after == "/" or LEGAL_PREFIX.search(before) \
            or text[m.end():m.end() + 1] == "-" and text[max(0, m.start() - 3):m.start()].strip().endswith("."):
        return "ln_acto_legal"
    if s in {"2100", "3000"} or re.fullmatch(r"(?:19|20)\d{2}", s):
        return "ln_ano_lejano"
    return "ln_cifra"


def doc_counts(text: str) -> dict:
    n_tok = max(len(text.split()), 1)
    out: dict[str, float] = {"n_tok": n_tok, "n_alpha": max(len(re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)*", text)), 1)}
    terms: Counter = Counter()
    # Familias actuales y subtipos (sumando los subtipos se recupera cada familia)
    for fam in FAMILIES:
        hits = list(RX[fam].finditer(text))
        out[fam] = len(hits)
        for h in hits:
            s = h.group(0)
            if fam == "percentages":
                val = float(re.sub(r"[^\d.,]", "", s).replace(",", "."))
                if val == 0:
                    sub = "pct_cero"
                elif TAXO_CTX.search(text[max(0, h.start() - 300):h.end() + 300]):
                    sub = "pct_entorno_taxonomia"
                else:
                    sub = "pct_resto"
            elif fam == "large_numbers":
                sub = classify_large_number(text, h)
            elif fam == "units":
                w = s.lower()
                num_before = bool(re.search(r"\d\s?$", text[max(0, h.start() - 3):h.start()]))
                sub = ("u_tema_" + w) if w in TOPIC_UNITS else ("u_fisica_tras_cifra" if num_before else "u_fisica_suelta")
            else:
                sub = "fw_" + re.sub(r"\s+", "_", s.lower())
            terms[sub] += 1
    out.update(terms)

    # Prototipo: solo cifras, cada numero una vez
    n_pct = n_unit = n_big = n_pct0 = 0
    for m in NUM_RE.finditer(text):
        s = m.group(0)
        tail = text[m.end():m.end() + 14]
        u = UNIT_AFTER.match(tail)
        if u and u.group(0).strip().startswith("%"):
            n_pct += 1
            if re.fullmatch(r"0+(?:[.,]0+)?", s):
                n_pct0 += 1
            continue
        if u:
            n_unit += 1
            continue
        digits = s.replace(",", "") if re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", s) else s
        if not re.fullmatch(r"\d+(?:\.\d+)?", digits):
            continue
        if float(digits) < 1000:
            continue
        if re.fullmatch(r"(?:19|20)\d{2}", digits) or YEAR_RANGE.match(digits) or digits in {"2100", "3000"}:
            continue
        before = text[max(0, m.start() - 12):m.start()]
        if CODE_PREFIX.search(before) or LEGAL_PREFIX.search(before) or tail[:1] in {":", "-"}:
            continue
        n_big += 1
    out.update({"c_pct": n_pct, "c_pct_cero": n_pct0, "c_unidad": n_unit, "c_grande": n_big})
    return out


# ---------------------------------------------------------------------------
# Figuras (estilo de scripts/figures_v3.py)
# ---------------------------------------------------------------------------

C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def setup_style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "axes.titlelocation": "left", "axes.titlecolor": INK, "lines.linewidth": 1.6,
        "legend.frameon": False, "figure.facecolor": "white", "savefig.facecolor": "white",
    })


def fig_families(ym: pd.DataFrame, path: Path):
    """Medias anuales de la densidad por familia (barras apiladas, x1e-4)."""
    fig, ax = plt.subplots(figsize=(5.6, 3.2), constrained_layout=True)
    cols = {"percentages": (C1, "Porcentajes"), "large_numbers": (C4, "Cifras >= 4 digitos"),
            "units": (C3, "Unidades (incl. CO2/GHG)"), "frameworks": (C2, "Marcos (taxonomy, GRI...)")}
    bottom = np.zeros(len(ym))
    for fam, (col, lab) in cols.items():
        v = ym[f"d_{fam}"].to_numpy() * 1e4
        ax.bar(ym["year"], v, bottom=bottom, color=col, width=0.72, edgecolor="white", linewidth=1.5, label=lab)
        bottom += v
    for x, b in zip(ym["year"], bottom):
        ax.text(x, b + 0.6, f"{b:.0f}", ha="center", va="bottom", fontsize=7, color=INK2)
    ax.set_ylabel("QUANT medio (aciertos por 10.000 tokens)")
    ax.set_title("QUANT actual por familia de patrones, media anual (49 informes/ano)")
    ax.legend(ncol=2, loc="upper left", bbox_to_anchor=(0, -0.12))
    ax.set_ylim(0, bottom.max() * 1.12)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_contrib(contrib: pd.DataFrame, path: Path):
    """Aportacion de cada subtipo a la pendiente del QUANT (x1e-4/ano)."""
    c = contrib.sort_values("slope_x1e4")
    fig, ax = plt.subplots(figsize=(5.6, 0.22 * len(c) + 0.9), constrained_layout=True)
    colors = [C2 if v > 0 else C1 for v in c["slope_x1e4"]]
    ax.barh(c["label"], c["slope_x1e4"], color=colors, height=0.68)
    for y, v in enumerate(c["slope_x1e4"]):
        ax.text(v + (0.03 if v >= 0 else -0.03), y, f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=6.8, color=INK2)
    ax.axvline(0, color=AXIS, lw=0.8)
    ax.set_xlabel("Pendiente EF (unidades de 1e-4 por ano); la suma es la pendiente total")
    ax.set_title(f"De donde sale la pendiente del QUANT ({c['slope_x1e4'].sum():.2f})")
    ax.grid(axis="y", visible=False)
    lo, hi = c["slope_x1e4"].min(), c["slope_x1e4"].max()
    ax.set_xlim(min(lo * 1.7, -0.3), hi * 1.25)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_variants(ym: pd.DataFrame, path: Path, variants: dict):
    """Media anual de Z(variante): mismas unidades (DT del corpus)."""
    fig, ax = plt.subplots(figsize=(5.6, 3.1), constrained_layout=True)
    styles = [(INK, "o", "-"), (C2, "s", "-"), (C1, "D", "-"), (MUTED, "^", (0, (3, 2)))]
    for (col, lab), (c, mk, ls) in zip(variants.items(), styles):
        ax.plot(ym["year"], ym[f"z_{col}"], color=c, marker=mk, ms=4.5, mec="white", mew=1, ls=ls, label=lab)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_ylabel("Media anual de Z (DT del corpus)")
    ax.set_title("QUANT actual frente a variantes, media anual estandarizada")
    ax.legend(loc="upper left")
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=BASE_DIR,
                    help="raiz con data/ y results/analysis_v3 (solo lectura)")
    ap.add_argument("--out", type=Path, default=BASE_DIR / "results" / "issue1_quant")
    a = ap.parse_args()
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    S: dict = {}

    df = load(a.root)
    cnt = pd.DataFrame([doc_counts(t) for t in df["raw_text"]]).fillna(0)
    df = pd.concat([df.drop(columns=["raw_text"]), cnt], axis=1)
    sub_cols = [c for c in cnt.columns if c.startswith(("pct_", "ln_", "u_", "fw_"))]

    # 1. Reproduccion ------------------------------------------------------
    df["QUANT_full"] = df[FAMILIES].sum(axis=1) / df["n_tok"]
    S["reproduccion"] = {
        "dif_max_vs_doc_level": float(np.abs(df["QUANT_full"] - df["QUANT_Score"]).max()),
        "subtipos_suman_familias": bool(np.allclose(df[sub_cols].sum(axis=1), df[FAMILIES].sum(axis=1))),
        "n_docs": len(df), "n_firmas": int(df["firm"].nunique()),
    }
    df["QUANT_Score"] = df["QUANT_full"]
    for fam in FAMILIES:
        df[f"d_{fam}"] = df[fam] / df["n_tok"]
    for c in sub_cols:
        df[f"d_{c}"] = df[c] / df["n_tok"]
    df["log_tok"] = np.log(df["n_tok"])

    # 2. Pendientes en unidades comparables --------------------------------
    comp = {"SUS_density": "SUS (densidad)", "SEN_Score": "SEN", "QUANT_Score": "QUANT", "HEDGE_Score": "HEDGE"}
    weights = {"SUS_density": -1.0, "SEN_Score": 1.0, "QUANT_Score": -W_QUANT, "HEDGE_Score": W_HEDGE}
    rows = []
    for col, lab in comp.items():
        f = fe_slope(df, col)
        sd = df[col].std(ddof=0)
        m18 = df.loc[df.year == 2018, col].mean()
        rows.append({"componente": lab, "pendiente_unidades_propias": f["slope"], "se": f["se"], "p": f["p"],
                     "r2_within": f["r2_within"], "media_corpus": df[col].mean(), "dt_corpus": sd,
                     "pendiente_en_DT_por_ano": f["slope"] / sd,
                     "pendiente_pct_media_2018_por_ano": 100 * f["slope"] / m18 if m18 else np.nan,
                     "cambio_pct_media_2018_2024": 100 * (df.loc[df.year == 2024, col].mean() / m18 - 1) if m18 else np.nan,
                     "aporte_a_pendiente_ESGSI_ext": weights[col] * f["slope"] / sd})
    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(out / "pendientes_componentes.csv", sep=";", index=False)
    S["pendiente_ESGSI_ext"] = fe_slope(df, "ESGSI_ext")
    S["pendiente_ESGSI"] = fe_slope(df, "ESGSI")

    # 3. Descomposicion exacta de la pendiente ------------------------------
    fam_rows, sub_rows = [], []
    q_slope = fe_slope(df, "QUANT_Score")["slope"]
    for fam in FAMILIES:
        f = fe_slope(df, f"d_{fam}")
        fam_rows.append({"familia": fam, "slope_x1e4": f["slope"] * 1e4, "p": f["p"],
                         "cuota_de_la_pendiente": f["slope"] / q_slope,
                         "aciertos_2018": int(df.loc[df.year == 2018, fam].sum()),
                         "aciertos_2024": int(df.loc[df.year == 2024, fam].sum()),
                         "cuota_aciertos_2018": df.loc[df.year == 2018, fam].sum() / df.loc[df.year == 2018, FAMILIES].sum().sum(),
                         "cuota_aciertos_2024": df.loc[df.year == 2024, fam].sum() / df.loc[df.year == 2024, FAMILIES].sum().sum()})
    for c in sub_cols:
        f = fe_slope(df, f"d_{c}")
        sub_rows.append({"subtipo": c, "aciertos_total": int(df[c].sum()),
                         "aciertos_2018": int(df.loc[df.year == 2018, c].sum()),
                         "aciertos_2024": int(df.loc[df.year == 2024, c].sum()),
                         "slope_x1e4": f["slope"] * 1e4, "cuota_de_la_pendiente": f["slope"] / q_slope})
    fam_df = pd.DataFrame(fam_rows)
    sub_df = pd.DataFrame(sub_rows).sort_values("slope_x1e4", ascending=False)
    fam_df.to_csv(out / "pendiente_por_familia.csv", sep=";", index=False)
    sub_df.to_csv(out / "pendiente_por_subtipo.csv", sep=";", index=False)
    S["pendiente_QUANT_x1e4"] = q_slope * 1e4
    S["suma_familias_x1e4"] = float(fam_df["slope_x1e4"].sum())

    # Agrupacion legible de subtipos para la figura
    groups = {
        "Marco: 'taxonomy'": ["fw_taxonomy"],
        "Marco: CSRD/SFDR/ISSB": ["fw_csrd", "fw_sfdr", "fw_issb"],
        "Marco: GRI/SASB/TCFD": ["fw_gri", "fw_sasb", "fw_tcfd"],
        "Marco: SDG/UNGC/Paris": ["fw_sdg", "fw_ungc", "fw_un_global_compact", "fw_paris_agreement"],
        "Unidad-tema: CO2/CO2e/GHG": [c for c in sub_cols if c.startswith("u_tema_")],
        "Unidad fisica tras cifra": ["u_fisica_tras_cifra"],
        "Unidad fisica suelta": ["u_fisica_suelta"],
        "% = 0 (celdas vacias)": ["pct_cero"],
        "% en entorno taxonomia": ["pct_entorno_taxonomia"],
        "% resto": ["pct_resto"],
        "Num.: cifra": ["ln_cifra"],
        "Num.: codigo de norma (ISO...)": ["ln_codigo_norma"],
        "Num.: acto legal (2021/2139...)": ["ln_acto_legal"],
        "Num.: rango de anos / 2100": ["ln_rango_anos", "ln_ano_lejano"],
    }
    assigned = sum(groups.values(), [])
    missing = [c for c in sub_cols if c not in assigned]
    if missing:
        groups["Otros"] = missing
    sub_idx = sub_df.set_index("subtipo")
    contrib = pd.DataFrame([{"label": k, "slope_x1e4": float(sub_idx.loc[[c for c in v if c in sub_idx.index], "slope_x1e4"].sum()),
                             "aciertos_2018": int(sub_idx.loc[[c for c in v if c in sub_idx.index], "aciertos_2018"].sum()),
                             "aciertos_2024": int(sub_idx.loc[[c for c in v if c in sub_idx.index], "aciertos_2024"].sum())}
                            for k, v in groups.items()])
    contrib["cuota_de_la_pendiente"] = contrib["slope_x1e4"] / (q_slope * 1e4)
    contrib.to_csv(out / "pendiente_por_grupo_de_subtipos.csv", sep=";", index=False)

    # Numerador frente a denominador
    df["aciertos"] = df[FAMILIES].sum(axis=1)
    tot = df.groupby("year").agg(aciertos=("aciertos", "sum"),
                                 tokens=("n_tok", "sum"), paginas_pdf=("pdf_pages", "mean"),
                                 palabras_pdf=("pdf_words", "mean"), rendimiento=("yield", "mean"))
    tot["aciertos_por_10k_tokens_agregado"] = 1e4 * tot["aciertos"] / tot["tokens"]
    tot.to_csv(out / "numerador_denominador_por_ano.csv", sep=";")
    S["crecimiento_2018_2024"] = {
        "aciertos_pct": 100 * (tot.loc[2024, "aciertos"] / tot.loc[2018, "aciertos"] - 1),
        "tokens_extraidos_pct": 100 * (tot.loc[2024, "tokens"] / tot.loc[2018, "tokens"] - 1),
        "paginas_pdf_pct": 100 * (tot.loc[2024, "paginas_pdf"] / tot.loc[2018, "paginas_pdf"] - 1),
        "palabras_pdf_pct": 100 * (tot.loc[2024, "palabras_pdf"] / tot.loc[2018, "palabras_pdf"] - 1),
        "rendimiento_extraccion_2018_2024": [tot.loc[2018, "rendimiento"], tot.loc[2024, "rendimiento"]],
    }

    # 4. Variantes ----------------------------------------------------------
    df["V0_actual"] = df["QUANT_Score"]
    df["V1_sin_marcos"] = (df["percentages"] + df["large_numbers"] + df["units"]) / df["n_tok"]
    df["V2_sin_marcos_ni_temas"] = (df["percentages"] + df["large_numbers"] + df["u_fisica_tras_cifra"]
                                    + df["u_fisica_suelta"]) / df["n_tok"]
    df["V2b_sin_marcos_temas_ni_codigos"] = (df["pct_cero"] + df["pct_entorno_taxonomia"] + df["pct_resto"]
                                             + df["ln_cifra"] + df["u_fisica_tras_cifra"]
                                             + df["u_fisica_suelta"]) / df["n_tok"]
    df["V3_solo_cifras"] = (df["c_pct"] + df["c_unidad"] + df["c_grande"]) / df["n_tok"]
    df["V3b_solo_cifras_sin_pct_cero"] = (df["c_pct"] - df["c_pct_cero"] + df["c_unidad"] + df["c_grande"]) / df["n_tok"]
    df["V3c_solo_cifras_por_palabra"] = (df["c_pct"] + df["c_unidad"] + df["c_grande"]) / df["n_alpha"]
    df["V_solo_porcentajes"] = df["percentages"] / df["n_tok"]
    variants = ["V0_actual", "V1_sin_marcos", "V2_sin_marcos_ni_temas", "V2b_sin_marcos_temas_ni_codigos", "V3_solo_cifras",
                "V3b_solo_cifras_sin_pct_cero", "V3c_solo_cifras_por_palabra", "V_solo_porcentajes"]
    zsus, zsen, zhed = z(df["SUS_density"]), z(df["SEN_Score"]), z(df["HEDGE_Score"])
    ext_ref = zsen - zsus - W_QUANT * z(df["V0_actual"]) + W_HEDGE * zhed
    S["ESGSI_ext_reproducido_dif_max"] = float(np.abs(ext_ref - df["ESGSI_ext"]).max())
    vrows = []
    for v in variants:
        f = fe_slope(df, v)
        sd = df[v].std(ddof=0)
        df[f"ext_{v}"] = zsen - zsus - W_QUANT * z(df[v]) + W_HEDGE * zhed
        fe_ext = fe_slope(df, f"ext_{v}")
        ov_q, ov_e = decile_overlap(df["V0_actual"], df[v]), decile_overlap(ext_ref, df[f"ext_{v}"])
        fs = firm_slopes(df, v)
        vrows.append({
            "variante": v, "media_x1e4": df[v].mean() * 1e4,
            "pendiente_x1e4": f["slope"] * 1e4, "p": f["p"], "r2_within": f["r2_within"],
            "pendiente_DT_por_ano": f["slope"] / sd,
            "pendiente_pct_media2018_por_ano": 100 * f["slope"] / df.loc[df.year == 2018, v].mean(),
            "firmas_pendiente_positiva": int((fs > 0).sum()),
            "spearman_vs_QUANT_actual": stats.spearmanr(df["V0_actual"], df[v])[0],
            "solape_top10_vs_QUANT": ov_q["top"], "solape_bottom10_vs_QUANT": ov_q["bottom"],
            "ESGSI_ext_pendiente": fe_ext["slope"], "ESGSI_ext_ci_lo": fe_ext["ci_lo"], "ESGSI_ext_ci_hi": fe_ext["ci_hi"],
            "ESGSI_ext_spearman_vs_actual": stats.spearmanr(ext_ref, df[f"ext_{v}"])[0],
            "ESGSI_ext_solape_top10": ov_e["top"], "ESGSI_ext_solape_bottom10": ov_e["bottom"],
            "corr_spearman_con_SUS": stats.spearmanr(df[v], df["SUS_density"])[0],
            "corr_within_con_SUS": float(np.corrcoef(within(df[v], df["firm"]), within(df["SUS_density"], df["firm"]))[0, 1]),
        })
    var_df = pd.DataFrame(vrows)
    # Partes del prototipo (suman su pendiente)
    S["V3_partes"] = {}
    for part in ("c_pct", "c_pct_cero", "c_unidad", "c_grande"):
        f = fe_slope(df.assign(_p=df[part] / df["n_tok"]), "_p")
        S["V3_partes"][part] = {"media_x1e4": float((df[part] / df["n_tok"]).mean() * 1e4),
                                "pendiente_x1e4": f["slope"] * 1e4, "p": f["p"],
                                "total_2018": int(df.loc[df.year == 2018, part].sum()),
                                "total_2024": int(df.loc[df.year == 2024, part].sum())}
    var_df.to_csv(out / "variantes_quant.csv", sep=";", index=False)

    # 5. Concentracion y robustez de la pendiente ----------------------------
    rob = {}
    for v in ("V0_actual", "V2b_sin_marcos_temas_ni_codigos", "V3_solo_cifras"):
        fs = firm_slopes(df, v)
        contrib_f = (fs / len(fs)).sort_values(ascending=False)   # panel equilibrado: EF = media de pendientes de empresa
        loo = [fe_slope(df[df.firm != f], v)["slope"] * 1e4 for f in fs.index]
        cut = df[v].quantile(0.99)
        med = df.groupby("year")[v].median()
        trimmed = df[df[v] <= cut]
        wins = df.assign(w=df[v].clip(df[v].quantile(0.01), cut))
        rho_firm = df.groupby("firm").apply(lambda d: stats.spearmanr(d["year"], d[v])[0])
        rob[v] = {
            "pendiente_x1e4": fe_slope(df, v)["slope"] * 1e4,
            "firmas_pendiente_positiva": int((fs > 0).sum()), "firmas": len(fs),
            "cuota_pendiente_top5_firmas": float(contrib_f.head(5).sum() / fs.mean()),
            "cuota_pendiente_top10_firmas": float(contrib_f.head(10).sum() / fs.mean()),
            "dejando_una_firma_fuera_min_max_x1e4": [min(loo), max(loo)],
            "sin_1pct_superior_x1e4": fe_slope(trimmed, v)["slope"] * 1e4 if trimmed.firm.value_counts().min() > 1 else None,
            "winsorizada_1_99_x1e4": fe_slope(wins, "w")["slope"] * 1e4,
            "log_pendiente_pct_por_ano": 100 * fe_slope(df.assign(l=np.log(df[v].clip(lower=1e-6))), "l")["slope"],
            "mediana_anual_x1e4": {int(k): float(x * 1e4) for k, x in med.items()},
            "spearman_ano_dentro_de_firma_media": float(rho_firm.mean()),
            "control_log_tokens_x1e4": fe_slope(df, v, ["log_tok"])["slope"] * 1e4,
            "corr_con_log_tokens_pooled": float(stats.spearmanr(df[v], df["log_tok"])[0]),
            "corr_con_log_tokens_within": float(np.corrcoef(within(df[v], df["firm"]), within(df["log_tok"], df["firm"]))[0, 1]),
        }
        lab = df.drop_duplicates("firm").set_index("firm")["firm_label"]
        top = pd.DataFrame({"firm_id": lab.reindex(contrib_f.index).values,
                            "pendiente_firma_x1e4": fs.reindex(contrib_f.index).values * 1e4,
                            "cuota_de_la_pendiente": contrib_f.values / fs.mean()})
        top.to_csv(out / f"pendiente_por_firma_{v}.csv", sep=";", index=False)
    S["robustez"] = rob

    # 6. Por grupo -----------------------------------------------------------
    grows = []
    for dim in ("industry", "country", "doctype"):
        for g, d in df.groupby(dim):
            for v in ("V0_actual", "V2b_sin_marcos_temas_ni_codigos", "V3_solo_cifras"):
                fs = firm_slopes(d, v)
                grows.append({"dimension": dim, "grupo": g, "variante": v, "firmas": len(fs), "docs": len(d),
                              "media_x1e4": d[v].mean() * 1e4,
                              "media_2018_x1e4": d.loc[d.year == 2018, v].mean() * 1e4,
                              "media_2024_x1e4": d.loc[d.year == 2024, v].mean() * 1e4,
                              "pendiente_media_firmas_x1e4": fs.mean() * 1e4,
                              "firmas_pendiente_positiva": int((fs > 0).sum())})
    pd.DataFrame(grows).to_csv(out / "por_grupo.csv", sep=";", index=False)
    S["tipo_documento_por_ano"] = pd.crosstab(df["year"], df["doctype"]).to_dict()

    # Medias anuales -----------------------------------------------------
    ycols = ["QUANT_Score"] + [f"d_{f}" for f in FAMILIES] + variants
    ym = df.groupby("year")[ycols + ["n_tok"]].mean()
    for v in variants:
        ym[f"z_{v}"] = df.assign(zz=z(df[v])).groupby("year")["zz"].mean()
    ym = ym.reset_index()
    ym.to_csv(out / "medias_anuales.csv", sep=";", index=False)

    # Figuras ------------------------------------------------------------
    setup_style()
    fig_families(ym, out / "fig_quant_familias_por_ano.png")
    fig_contrib(contrib, out / "fig_aporte_a_la_pendiente.png")
    fig_variants(ym, out / "fig_variantes_z_por_ano.png",
                 {"V0_actual": "QUANT actual", "V2b_sin_marcos_temas_ni_codigos": "Sin marcos, CO2/GHG ni codigos (propuesta)",
                  "V3_solo_cifras": "Solo cifras (alternativa)", "V_solo_porcentajes": "Solo porcentajes"})

    S["componentes"] = comp_df.to_dict(orient="records")
    S["familias"] = fam_df.to_dict(orient="records")
    S["grupos_de_subtipos"] = contrib.to_dict(orient="records")
    S["variantes"] = var_df.to_dict(orient="records")

    def js(o):
        if isinstance(o, dict):
            return {str(k): js(x) for k, x in o.items()}
        if isinstance(o, (list, tuple)):
            return [js(x) for x in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, float) and not np.isfinite(o):
            return None
        return o
    (out / "summary.json").write_text(json.dumps(js(S), indent=1, ensure_ascii=False), encoding="utf-8")

    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(json.dumps(js(S["reproduccion"]), indent=1))
    print(comp_df.round(4).to_string())
    print(fam_df.round(4).to_string())
    print(contrib.round(3).to_string())
    print(var_df.round(4).T.to_string())
    print(json.dumps(js(S["V3_partes"]), indent=1))
    print(json.dumps(js(rob), indent=1))
    print(tot.round(3).to_string())


if __name__ == "__main__":
    main()
