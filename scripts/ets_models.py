"""
ets_models.py
-------------
Issue #2, criterio externo gratuito: "lo que se dice" frente a "lo que se
hace". Contrasta los scores de los informes (ESGSI, ESGSI extendido y sus
componentes, siempre como variables continuas) con las emisiones verificadas
del EU ETS de las instalaciones fijas de cada grupo (panel de ets_match.py).

    python -I scripts/ets_models.py --results-dir <ruta a results/ del clon principal> [--n-boot 9999]

Preguntas
  A. Prospectiva: el score del informe del ejercicio t, ¿se asocia con la
     variacion de las emisiones verificadas entre t y t+1 (y t+2)?
     y = log E(t+h) - log E(t), h = 1, 2; variante like-for-like (solo
     instalaciones con emisiones en los dos anos, h = 1).
     Especificaciones: EF de ano; EF de ano + industria; EF de empresa + ano.
     SE agrupados por empresa (CR1, t con G-1 gl) y wild cluster bootstrap
     restringido (Rademacher). Regresor estandarizado en la muestra: beta =
     cambio en puntos logaritmicos por 1 DT del score.
  B. Contemporanea: ¿la densidad de sustancia (SUS) o el ESGSI siguen el nivel
     o la intensidad de emisiones (t CO2e por millon de euros de ingresos)?
     Spearman entre medias de empresa y correlacion parcial tras EF de ano,
     ano + industria o empresa + ano (p agrupado y wild).

Potencia: MDE al 80 % (alfa = 0,05, bilateral) = (t_{0,975;G-1} + t_{0,80;G-1}) * SE.

Entradas
  results/issue2_ets/panel_ets.csv                     (ets_match.py, anonimo)
  <results-dir>/analysis_v3/doc_level.csv              scores por informe (anonimo)
  data/issue2_ets/revenue/*.csv + ets_firms_internal.csv + firm_ids_internal.csv
                                                       (ingresos para la intensidad; interno)
Salidas (anonimas)
  results/issue2_ets/modelos_ets.csv        A: beta, SE, p agrupado, p wild, MDE, G, N
  results/issue2_ets/correlaciones_ets.csv  B
  results/issue2_ets/muestra_ets.csv        descriptivos por firm_id de la muestra
  results/issue2_ets/modelos_ets_resumen.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from fin_econ import dummies, ols_cl, twoway_demean, wild_cluster_p  # noqa: E402

DATA = ROOT / "data" / "issue2_ets"
OUT = ROOT / "results" / "issue2_ets"
SCORES = ["ESGSI", "ESGSI_ext", "Z_SEN", "Z_SUS", "Z_QUANT", "Z_HEDGE"]
MATERIAL_KT = 25.0   # muestra principal: media 2018-2024 >= 25 kt CO2e (excluye
                     # centros de datos y plantas moviles cuyo log es ruido)


def load_revenue(results_dir: Path) -> pd.DataFrame:
    wd = pd.read_csv(DATA / "revenue" / "wikidata_revenue_internal.csv")
    fx = pd.read_csv(DATA / "revenue" / "ecb_usd_eur_annual.csv")
    fx = fx.set_index(fx.TIME_PERIOD.astype(int))["OBS_VALUE"]          # USD por EUR
    wd["year"] = pd.to_datetime(wd.date, errors="coerce").dt.year
    wd["unit"] = wd.unit.str.rsplit("/", n=1).str[-1]
    wd = wd.dropna(subset=["year"])
    wd["year"] = wd.year.astype(int)
    eur = wd[wd.unit == "Q4916"].assign(rev_eur=lambda d: d.rev)
    usd = wd[wd.unit == "Q4917"].assign(rev_eur=lambda d: d.rev / d.year.map(fx))
    r = pd.concat([eur, usd]).dropna(subset=["rev_eur"])
    # Un valor por ISIN-ano: se prefiere el euro; si hay varios, la mediana
    r["pref"] = (r.unit != "Q4916").astype(int)
    r = r.sort_values("pref").groupby(["isin", "year"]).apply(
        lambda g: g[g.pref == g.pref.min()].rev_eur.median()).rename("rev_eur").reset_index()
    firms = pd.read_csv(DATA / "ets_firms_internal.csv", sep=";", encoding="utf-8", keep_default_na=False)
    ids = pd.read_csv(results_dir / "analysis_v3" / "firm_ids_internal.csv", sep=";", encoding="utf-8")
    m = firms[["firm", "isin"]].merge(ids, on="firm")
    return r.merge(m[["isin", "firm_id"]], on="isin")[["firm_id", "year", "rev_eur"]]


def build(results_dir: Path) -> pd.DataFrame:
    pan = pd.read_csv(OUT / "panel_ets.csv", sep=";")
    doc = pd.read_csv(results_dir / "analysis_v3" / "doc_level.csv", sep=";")
    E = pan.pivot_table(index="firm_id", columns="year", values="em_fixed")
    L = pan.pivot_table(index="firm_id", columns="year", values="dlog_lfl").reindex(E.index)
    rows = []
    for r in doc.itertuples():
        if r.firm_id not in E.index:
            continue
        e = E.loc[r.firm_id]
        def lg(y):
            v = e.get(y, np.nan)
            return np.log(v) if v and v > 0 else np.nan
        rows.append({"firm_id": r.firm_id, "year": r.year, "E_t": e.get(r.year, np.nan),
                     "logE_t": lg(r.year), "dlog1": lg(r.year + 1) - lg(r.year),
                     "dlog2": lg(r.year + 2) - lg(r.year),
                     "lfl1": L.loc[r.firm_id].get(r.year + 1, np.nan)})
    d = pd.DataFrame(rows).merge(doc, on=["firm_id", "year"])
    mean_kt = (E.loc[:, 2018:2024].mean(axis=1) / 1e3).rename("mean_kt_2018_2024")
    d = d.merge(mean_kt, left_on="firm_id", right_index=True)
    d = d[d.mean_kt_2018_2024 > 0]                    # cubiertas: emisiones fijas positivas
    rev = load_revenue(results_dir)
    d = d.merge(rev, on=["firm_id", "year"], how="left")
    d["log_int"] = np.log(d.E_t / (d.rev_eur / 1e6))          # t CO2e por M EUR
    return d


def zs(x: pd.Series) -> np.ndarray:
    return ((x - x.mean()) / x.std(ddof=0)).to_numpy()


def fit(d: pd.DataFrame, y: str, x: str, spec: str, n_boot: int) -> dict:
    s = d.dropna(subset=[y, x]).copy()
    codes = pd.factorize(s.firm_id)[0]
    G = codes.max() + 1
    xv = zs(s[x])
    yv = s[y].to_numpy(float)
    if spec == "firm_year":
        yc = pd.factorize(s.year)[0]
        yd = twoway_demean(yv, codes, yc)
        Xd = twoway_demean(xv, codes, yc)[:, None]
        b, se, p = ols_cl(yd, Xd, codes, 0, df_absorbed=G + yc.max())
        pw = wild_cluster_p(yd, Xd, codes, 0, n_boot=n_boot)
    else:
        parts = [xv[:, None], np.ones((len(s), 1)), dummies(s.year)]
        if spec == "year_industry":
            parts.append(dummies(s.industry))
        X = np.hstack(parts)
        b, se, p = ols_cl(yv, X, codes, 0)
        pw = wild_cluster_p(yv, X, codes, 0, n_boot=n_boot)
    tcrit = stats.t.ppf(0.975, G - 1) + stats.t.ppf(0.80, G - 1)
    sdy = float(np.std(yv, ddof=1))
    return {"outcome": y, "score": x, "spec": spec, "beta": b, "se": se, "p_cluster": p, "p_wild": pw,
            "mde80": tcrit * se, "mde80_sd_y": tcrit * se / sdy, "sd_y": sdy, "G": int(G), "N": int(len(s))}


def resid(v: np.ndarray, s: pd.DataFrame, spec: str) -> np.ndarray:
    """Residuo de v tras los efectos fijos de la especificacion."""
    if spec == "firm_year":
        return twoway_demean(v, pd.factorize(s.firm_id)[0], pd.factorize(s.year)[0])
    parts = [np.ones((len(s), 1)), dummies(s.year)]
    if spec == "year_industry":
        parts.append(dummies(s.industry))
    X = np.hstack(parts)
    return v - X @ np.linalg.lstsq(X, v, rcond=None)[0]


def corr_block(d: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    """Correlacion parcial (tras EF) entre cada score y el nivel o la intensidad
    de emisiones del mismo ano; p agrupado por empresa y wild sobre la
    regresion de los residuos estandarizados."""
    out = []
    for lev in ["logE_t", "log_int"]:
        for x in ["SUS_density", "ESGSI", "ESGSI_ext", "Z_SEN", "Z_QUANT", "Z_HEDGE"]:
            s = d.dropna(subset=[lev, x])
            fm = s.groupby("firm_id")[[lev, x]].mean()
            rho, prho = stats.spearmanr(fm[x], fm[lev])
            rec = {"nivel": lev, "score": x, "spearman_medias_empresa": rho, "p_spearman": prho,
                   "n_firmas": int(len(fm)), "N": int(len(s))}
            codes = pd.factorize(s.firm_id)[0]
            G = codes.max() + 1
            for spec in ["year", "year_industry", "firm_year"]:
                rx = resid(s[lev].to_numpy(float), s, spec)
                ry = resid(s[x].to_numpy(float), s, spec)
                zx, zy = (rx - rx.mean()) / rx.std(), (ry - ry.mean()) / ry.std()
                k_abs = (G + s.year.nunique() - 1) if spec == "firm_year" else (
                    s.year.nunique() + (s.industry.nunique() - 1 if spec == "year_industry" else 0))
                _, _, pc = ols_cl(zy, zx[:, None], codes, 0, df_absorbed=k_abs)
                rec[f"r_parcial_{spec}"] = float(np.corrcoef(zx, zy)[0, 1])
                rec[f"p_{spec}"] = pc
                rec[f"pwild_{spec}"] = wild_cluster_p(zy, zx[:, None], codes, 0, n_boot=n_boot)
            out.append(rec)
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True, type=Path)
    ap.add_argument("--n-boot", type=int, default=9999)
    a = ap.parse_args()
    d = build(a.results_dir)
    mat = d[d.mean_kt_2018_2024 >= MATERIAL_KT]
    rows = []
    for name, s in [("material", mat), ("todas_cubiertas", d)]:
        for y in ["dlog1", "dlog2", "lfl1"]:
            for x in SCORES:
                for spec in ["year", "year_industry", "firm_year"]:
                    r = fit(s, y, x, spec, a.n_boot)
                    r["muestra"] = name
                    rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "modelos_ets.csv", sep=";", index=False, float_format="%.4f")
    cor = corr_block(mat, a.n_boot)
    cor.to_csv(OUT / "correlaciones_ets.csv", sep=";", index=False, float_format="%.4f")

    desc = (d.groupby("firm_id").agg(industry=("industry", "first"), country=("country", "first"),
                                     mean_kt_2018_2024=("mean_kt_2018_2024", "first"),
                                     dlog1_media=("dlog1", "mean"), dlog1_dt=("dlog1", "std"),
                                     n_dlog1=("dlog1", "count"), n_ingresos=("rev_eur", "count"),
                                     ESGSI_media=("ESGSI", "mean"))
            .reset_index())
    desc["material"] = desc.mean_kt_2018_2024 >= MATERIAL_KT
    desc.round(4).to_csv(OUT / "muestra_ets.csv", sep=";", index=False)

    # Varianza intra-empresa del regresor en la muestra principal
    s = mat.dropna(subset=["ESGSI", "dlog1"])
    fc, yc = pd.factorize(s.firm_id)[0], pd.factorize(s.year)[0]
    share_within = {x: float(np.var(twoway_demean(s[x].to_numpy(float), fc, yc)) / np.var(s[x]))
                    for x in ["ESGSI", "ESGSI_ext"]}
    key = res[(res.muestra == "material") & (res.score.isin(["ESGSI", "ESGSI_ext"]))]
    summ = {"n_firmas_cubiertas": int(d.firm_id.nunique()),
            "n_firmas_material": int(mat.firm_id.nunique()),
            "umbral_material_kt": MATERIAL_KT,
            "N_dlog1_material": int(mat.dlog1.notna().sum()),
            "N_dlog2_material": int(mat.dlog2.notna().sum()),
            "firm_years_con_ingresos_material": int(mat.rev_eur.notna().sum()),
            "share_var_intra_empresa_ano": share_within,
            "min_p_cluster_ESGSI_y_ext": float(key.p_cluster.min()),
            "min_p_wild_ESGSI_y_ext": float(key.p_wild.min()),
            "mde80_dlog1_rango": [float(key[key.outcome == "dlog1"].mde80.min()),
                                  float(key[key.outcome == "dlog1"].mde80.max())],
            "n_boot": a.n_boot}
    (OUT / "modelos_ets_resumen.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    pd.set_option("display.width", 220)
    print(res[res.muestra == "material"].round(3).to_string(index=False))
    print(cor.round(3).to_string(index=False))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
