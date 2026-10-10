"""
revision_v3.py
--------------
Diagnosticos pedidos en la revision simulada (issue #3, seccion 1 de la
sintesis). Lee las salidas de analysis_v3.py y escribe
results/analysis_v3/revision.json (y tablas csv) con:

  r1  Correlaciones entre Z(SEN), Z(SUS), Z(QUANT) y Z(HEDGE): agrupada, dentro
      de empresa (desviaciones de la media de empresa) y entre empresas (medias
      de empresa, n = 49); IC 95 % por bootstrap de empresas.
  r2  Varianza de los indices (ESGSI y ESGSI_ext no tienen varianza 1: son
      combinaciones de z) y pendientes intra-empresa re-estandarizadas, en DT
      del propio indice; pendiente de SUS en unidades naturales y en DT
      intra-empresa.
  r3  Descomposicion de la varianza de cada componente y de los indices: parte
      entre empresas, parte entre anos y residuo.
  r4  Pesos efectivos del ESGSI_ext: contribucion de cada componente a su
      varianza (peso por covarianza con el indice).
  r5  Mandatos: diferencias en diferencias con efectos fijos de empresa y ano
      (errores agrupados por empresa) del alineamiento de la Taxonomia y de la
      CSRD sobre Z(SUS), Z(SEN), ESGSI y ESGSI_ext, con su efecto minimo
      detectable (2,8 x EE: 80 % de potencia, 5 % bilateral).
  r6  Adopcion voluntaria de las ESRS en 2024: informes que remiten a normas
      tematicas ESRS (E1-E5, S1-S4, G1, ESRS 2), por obligacion legal.

Directrices (paper/_research/directrices.md): scores continuos, sin umbrales
ni etiquetas; ninguna salida con nombres de empresa (identificadores
'<Industria>-NN' de analysis_v3).

    PYTHONIOENCODING=utf-8 python scripts/revision_v3.py
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

import analysis_v3 as A  # noqa: E402

OUT = BASE_DIR / "results" / "analysis_v3"
MANDATES = BASE_DIR / "metadata" / "mandates.csv"
PROCESSED = BASE_DIR / "data" / "clean_v3_sec" / "processed_texts.csv"

SEED = 20261010
B = 2000
COMPONENTS = ["Z_SEN", "Z_SUS", "Z_QUANT", "Z_HEDGE"]
INDICES = ["ESGSI", "ESGSI_ext"]
W_QUANT, W_HEDGE = 0.5, 0.5
MDE_FACTOR = 2.8           # z_{0,975} + z_{0,80}
ESRS_RE = re.compile(r"\bESRS\s?(?:E[1-5]|S[1-4]|G1|2)\b")
ESRS_MIN = 10              # remisiones a normas tematicas para contar como estado ESRS


def within(df: pd.DataFrame, col: str) -> np.ndarray:
    return (df[col] - df.groupby("firm")[col].transform("mean")).to_numpy(float)


def corr_views(df: pd.DataFrame, a: str, b: str) -> dict:
    fm = df.groupby("firm")[[a, b]].mean()
    return {"agrupada": float(np.corrcoef(df[a], df[b])[0, 1]),
            "intra_empresa": float(np.corrcoef(within(df, a), within(df, b))[0, 1]),
            "entre_empresas": float(np.corrcoef(fm[a], fm[b])[0, 1])}


def boot_corr(df: pd.DataFrame, a: str, b: str, rng: np.random.Generator) -> dict:
    firms = df["firm"].unique()
    groups = {f: g for f, g in df.groupby("firm")}
    vals = {"agrupada": [], "intra_empresa": [], "entre_empresas": []}
    for _ in range(B):
        draw = rng.choice(firms, size=len(firms), replace=True)
        bd = pd.concat([groups[f].assign(firm=f"{f}#{i}") for i, f in enumerate(draw)], ignore_index=True)
        for k, v in corr_views(bd, a, b).items():
            vals[k].append(v)
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in vals.items()}


def var_decomp(df: pd.DataFrame, col: str) -> dict:
    y = df[col].to_numpy(float)
    tot = y.var()
    firm_part = df.groupby("firm")[col].transform("mean").to_numpy(float)
    year_part = df.groupby("year")[col].transform("mean").to_numpy(float)
    f = firm_part.var() / tot
    t = year_part.var() / tot
    return {"entre_empresas": float(f), "entre_anos": float(t), "residuo": float(1 - f - t)}


def did(df: pd.DataFrame, outcome: str, treat: str) -> dict:
    """y_it = a_i + g_t + b * D_it + e_it, EE agrupados por empresa (t con G-1 gl)."""
    X = pd.get_dummies(df[["firm", "year"]].astype(str), drop_first=True, dtype=float)
    X.insert(0, treat, df[treat].astype(float).to_numpy())
    X = sm.add_constant(X)
    fit = sm.OLS(df[outcome].to_numpy(float), X).fit(
        cov_type="cluster", cov_kwds={"groups": df["firm"].to_numpy()}, use_t=True)
    b, se = float(fit.params[treat]), float(fit.bse[treat])
    lo, hi = fit.conf_int().loc[treat]
    return {"coef": b, "ee": se, "p": float(fit.pvalues[treat]), "ic95": [float(lo), float(hi)],
            "mde_80": MDE_FACTOR * se, "tratados_empresa_ano": int(df[treat].sum())}


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    dl = pd.read_csv(OUT / "doc_level_internal.csv", sep=";")
    dl["firm"] = dl["firm"].map(A.nfc)
    S: dict = {"generado": time.strftime("%Y-%m-%d %H:%M:%S"), "documentos": len(dl),
               "empresas": int(dl["firm"].nunique()), "bootstrap_empresas": B, "semilla": SEED}

    # r1 ----------------------------------------------------------------------
    pairs = [(a, b) for i, a in enumerate(COMPONENTS) for b in COMPONENTS[i + 1:]]
    r1 = {}
    for a, b in pairs:
        r1[f"{a}~{b}"] = {**corr_views(dl, a, b), "ic95": boot_corr(dl, a, b, rng)}
    S["r1_correlaciones_componentes"] = r1
    print(f"r1 ({time.time() - t0:.0f}s)", flush=True)

    # r2 ----------------------------------------------------------------------
    r2 = {}
    for ix in INDICES:
        sd = float(dl[ix].std(ddof=0))
        slope = A.fe_slope(dl, ix)
        r2[ix] = {"varianza": sd ** 2, "dt": sd, "pendiente_ef": slope["pendiente"],
                  "pendiente_ef_en_dt_del_indice": slope["pendiente"] / sd,
                  "ic95_en_dt_del_indice": [v / sd for v in slope["ci95"]]}
    sus_slope = A.fe_slope(dl, "SUS_density")
    r2["SUS_density_unidades"] = {
        "pendiente_menciones_100_tokens_ano": sus_slope["pendiente"], "ic95": sus_slope["ci95"],
        "dt_intra_empresa": float(np.std(within(dl, "SUS_density"))),
        "pendiente_en_dt_intra": sus_slope["pendiente"] / float(np.std(within(dl, "SUS_density"))),
        "media_2018": float(dl.loc[dl["year"] == 2018, "SUS_density"].mean()),
        "media_2024": float(dl.loc[dl["year"] == 2024, "SUS_density"].mean())}
    S["r2_varianza_y_pendientes"] = r2

    # r3 ----------------------------------------------------------------------
    S["r3_descomposicion_varianza"] = {c: var_decomp(dl, c) for c in COMPONENTS + INDICES}

    # r4 ----------------------------------------------------------------------
    w = {"Z_SEN": 1.0, "Z_SUS": -1.0, "Z_QUANT": -W_QUANT, "Z_HEDGE": W_HEDGE}
    ext = sum(wt * dl[c] for c, wt in w.items())
    vext = float(ext.var(ddof=0))
    S["r4_pesos_efectivos_ext"] = {
        c: float(wt * np.cov(dl[c], ext, ddof=0)[0, 1] / vext) for c, wt in w.items()}
    S["r4_pesos_efectivos_ext"]["comprobacion_reconstruye_ESGSI_ext"] = float(np.abs(ext - dl["ESGSI_ext"]).max())

    # r5 ----------------------------------------------------------------------
    m = pd.read_csv(MANDATES, sep=";")
    m["Compañía"] = m["Compañía"].map(A.nfc)
    d5 = dl.merge(m[["Compañía", "year", "taxonomy_alignment", "csrd", "plantilla_art8"]],
                  left_on=["firm", "year"], right_on=["Compañía", "year"], how="left", validate="one_to_one")
    if d5["taxonomy_alignment"].isna().any():
        raise ValueError("informes sin fila en metadata/mandates.csv")
    r5 = {}
    for treat in ("taxonomy_alignment", "csrd"):
        r5[treat] = {y: did(d5, y, treat) for y in ["Z_SUS", "Z_SEN", "ESGSI", "ESGSI_ext"]}
    r5["firmas_con_csrd_2024"] = int(d5.loc[d5["year"] == 2024, "csrd"].sum())
    r5["alineamiento_primer_ano"] = (d5[d5["taxonomy_alignment"] == 1].groupby("firm")["year"].min()
                                     .value_counts().sort_index().to_dict())
    r5["subida_total_Z_SUS_2018_2024"] = float(dl.loc[dl["year"] == 2024, "Z_SUS"].mean()
                                               - dl.loc[dl["year"] == 2018, "Z_SUS"].mean())
    S["r5_mandatos_did"] = r5
    print(f"r5 ({time.time() - t0:.0f}s)", flush=True)

    # r6 ----------------------------------------------------------------------
    proc = pd.read_csv(PROCESSED, sep=";", usecols=["País", "Compañía", "Año", "Documento", "raw_text"])
    proc = A.norm_keys(proc)
    p24 = proc[proc["Año"].astype(str) == "2024"].copy()
    p24["esrs_refs"] = p24["raw_text"].astype(str).map(lambda t: len(ESRS_RE.findall(t)))
    p24 = p24.groupby("Compañía", as_index=False)["esrs_refs"].max()
    p24 = p24.merge(m[m["year"] == 2024][["Compañía", "csrd"]], on="Compañía", how="left")
    p24["estado_esrs"] = p24["esrs_refs"] >= ESRS_MIN
    S["r6_esrs_2024"] = {
        "umbral_remisiones": ESRS_MIN,
        "con_csrd_legal": {"empresas": int((p24["csrd"] == 1).sum()),
                           "con_estado_esrs": int(p24.loc[p24["csrd"] == 1, "estado_esrs"].sum())},
        "sin_csrd_legal": {"empresas": int((p24["csrd"] == 0).sum()),
                           "con_estado_esrs": int(p24.loc[p24["csrd"] == 0, "estado_esrs"].sum())},
        "remisiones_mediana": float(p24["esrs_refs"].median())}

    (OUT / "revision.json").write_text(json.dumps(A.jsonable(S), ensure_ascii=False, indent=1), encoding="utf-8")
    rows = [{"par": k, **{f"r_{v}": r[v] for v in ("agrupada", "intra_empresa", "entre_empresas")},
             **{f"ic95_{v}": r["ic95"][v] for v in ("agrupada", "intra_empresa", "entre_empresas")}}
            for k, r in r1.items()]
    A.save_csv(pd.DataFrame(rows), OUT / "revision_correlaciones.csv")
    print(json.dumps(A.jsonable({k: S[k] for k in ("r2_varianza_y_pendientes", "r4_pesos_efectivos_ext",
                                                    "r6_esrs_2024")}), indent=1, ensure_ascii=False))
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
