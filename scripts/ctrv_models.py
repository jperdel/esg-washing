"""
ctrv_models.py
--------------
Issue #2 (H2), modelos exploratorios de recuento: controversias en t+1 y t+2
sobre el ESGSI, el ESGSI extendido y sus componentes (Z(SEN), Z(SUS)) del
informe del ejercicio t. Placebo: controversias en t-1 (que el informe de t
no puede anticipar).

    python -I scripts/ctrv_models.py [--outcomes n_bhrrc,m_bhrrc,...]

Muestra: informe del ejercicio t (2018-2024) emparejado con el recuento del
ano natural t+h; solo anos completos (<= 2025), asi que h = +1 usa 343
informes, h = +2 usa 294 y h = -1 usa 343 (recuentos 2017-2023).

Especificaciones (todas con SE agrupados por empresa, CR1, t con G-1 g.l.):
  A  Poisson (cuasi-verosimilitud), EF de ano + industria + pais,
     controles: log tamano (capitalizacion aproximada a cierre de t) y
     log(1 + recuento en t) (controversias previas).
  NB Binomial negativa (NB2), mismas variables que A.
  B  Poisson con EF de empresa y de ano (solo variacion intra-empresa),
     control log tamano. Las empresas sin ningun evento en la muestra no
     aportan informacion y se excluyen.
Regresores: el score estandarizado (DT = 1 en la muestra de estimacion) o
Z(SEN) y Z(SUS) juntos. Se publica la razon de tasas (IRR) por DT.

Directrices: el score se usa como variable continua, sin umbrales ni
etiquetas; sin nombres de empresa (solo identificadores anonimos).
Salida: results/issue2_controversies/modelos_h2.csv y modelos_h2_resumen.json
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ctrv_common import DATA, RESULTS, firm_panel  # noqa: E402
from fin_econ import dummies, poisson_cl  # noqa: E402

warnings.filterwarnings("ignore")
DEFAULT_OUT = ["n_bhrrc", "m_bhrrc", "n_bhrrcf", "n_bhrrc_E", "n_bhrrc_sev2", "n_pool", "n_poolf",
               "gd_neg", "gd_E", "gd_GW"]
REGS = {"ESGSI": ["ESGSI"], "ESGSI_ext": ["ESGSI_ext"], "componentes": ["Z_SEN", "Z_SUS"]}
LEADS = [1, 2, -1]


def build(outcome: str, h: int) -> pd.DataFrame:
    sc = firm_panel()[["firm_id", "year", "industry", "country", "ESGSI", "ESGSI_ext", "Z_SEN", "Z_SUS"]]
    p = pd.read_csv(RESULTS / "panel_controversias.csv", sep=";")
    if outcome not in p.columns:
        return pd.DataFrame()
    y = p[["firm_id", "year", outcome]].rename(columns={outcome: "y"})
    y["year"] = y["year"] - h                     # recuento de t+h alineado con el informe de t
    pre = p[["firm_id", "year", outcome]].rename(columns={outcome: "y_pre"})
    d = sc.merge(y, on=["firm_id", "year"]).merge(pre, on=["firm_id", "year"])
    if "gd_all" in p.columns and outcome.startswith("gd_"):
        att = p[["firm_id", "year", "gd_all"]].rename(columns={"gd_all": "att_pre"})
        d = d.merge(att, on=["firm_id", "year"])
    mk = pd.read_csv(DATA / "panel_mercado_internal.csv", sep=";")[["firm_id", "year", "log_size"]]
    d = d.merge(mk, on=["firm_id", "year"], how="left")
    d = d[(d["year"] + h).between(2017, 2025)].dropna(subset=["y", "log_size"])
    return d.reset_index(drop=True)


def zs(v: pd.Series) -> np.ndarray:
    return ((v - v.mean()) / v.std(ddof=0)).to_numpy()


def nb_cl(y, X, codes, ks):
    """NB2 por maxima verosimilitud (statsmodels) con SE agrupados por empresa."""
    import statsmodels.api as sm
    try:
        m = sm.NegativeBinomial(y, X, loglike_method="nb2").fit(
            disp=0, maxiter=200, cov_type="cluster", cov_kwds={"groups": codes})
    except Exception:  # noqa: BLE001
        return [(np.nan, np.nan, np.nan)] * len(ks), np.nan
    G = codes.max() + 1
    out = []
    for k in ks:
        b, se = float(m.params[k]), float(m.bse[k])
        out.append((b, se, float(2 * stats.t.sf(abs(b / se), G - 1))))
    return out, float(m.params[-1])


def drop_separated(d: pd.DataFrame, fes: list[str]) -> pd.DataFrame:
    """Quita observaciones de celdas de efectos fijos sin ningun evento (separacion:
    su coeficiente tenderia a -inf y no aportan informacion), como ppmlhdfe."""
    while True:
        bad = np.zeros(len(d), bool)
        for fe in fes:
            bad |= (d.groupby(fe)["y"].transform("sum") == 0).to_numpy()
        if not bad.any():
            return d.reset_index(drop=True)
        d = d[~bad]


def fit(d: pd.DataFrame, regs: list[str], spec: str):
    d = drop_separated(d, ["year", "industry", "country"] if spec in ("A", "NB") else ["year", "firm_id"])
    y = d["y"].to_numpy(float)
    fc = pd.factorize(d["firm_id"])[0]
    xs = [zs(d[r]) for r in regs]
    ctrl = [zs(d["log_size"])]
    if spec in ("A", "NB"):
        ctrl.append(np.log1p(d["y_pre"].to_numpy(float)))
        if "att_pre" in d:
            ctrl.append(np.log1p(d["att_pre"].to_numpy(float)))
        X = np.column_stack(xs + ctrl + [np.ones(len(d)), dummies(d["year"]), dummies(d["industry"]),
                                          dummies(d["country"])])
    else:  # B: EF de empresa y ano
        X = np.column_stack(xs + ctrl + [np.ones(len(d)), dummies(d["year"]), dummies(d["firm_id"])])
    # quita columnas constantes o colineales (p. ej. pais = industria en celdas pequenas)
    keep = [j for j in range(X.shape[1]) if j < len(xs) + len(ctrl) + 1 or X[:, j].std() > 0]
    X = X[:, keep]
    if spec == "NB":
        res, alpha = nb_cl(y, X, fc, list(range(len(xs))))
        return res, alpha, len(d)
    return [poisson_cl(y, X, fc, j=k, max_iter=300) for k in range(len(xs))], np.nan, len(d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcomes", default=",".join(DEFAULT_OUT))
    a = ap.parse_args()
    rows = []
    for oc in a.outcomes.split(","):
        for h in LEADS:
            d0 = build(oc, h)
            if d0.empty:
                continue
            for spec in ["A", "NB", "B"]:
                d = d0
                if spec == "B":
                    pos = d.groupby("firm_id")["y"].transform("sum") > 0
                    d = d[pos].reset_index(drop=True)
                for rname, regs in REGS.items():
                    res, alpha, n_eff = fit(d, regs, spec)
                    for r, (b, se, p) in zip(regs, res):
                        rows.append({"resultado": oc, "h": h, "especificacion": spec, "regresor": rname,
                                     "variable": r, "n_obs": n_eff, "n_empresas": d["firm_id"].nunique(),
                                     "media_y": d["y"].mean(), "beta": b, "se": se, "p": p,
                                     "irr_por_dt": np.exp(b) if np.isfinite(b) else np.nan,
                                     "irr_ic95_inf": np.exp(b - 1.96 * se) if np.isfinite(se) else np.nan,
                                     "irr_ic95_sup": np.exp(b + 1.96 * se) if np.isfinite(se) else np.nan,
                                     "nb_alpha": alpha})
    out = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS / "modelos_h2.csv", sep=";", index=False)
    pd.set_option("display.width", 250)
    show = out[out["regresor"] != "componentes"].pivot_table(
        index=["resultado", "regresor", "h"], columns="especificacion", values=["irr_por_dt", "p"])
    print(show.round(3).to_string())
    comp = out[out["regresor"] == "componentes"].pivot_table(
        index=["resultado", "variable", "h"], columns="especificacion", values=["irr_por_dt", "p"])
    print(comp.round(3).to_string())
    summ = {"n_modelos": len(out), "n_p_menor_005": int((out["p"] < 0.05).sum()),
            "n_p_menor_010": int((out["p"] < 0.10).sum())}
    (RESULTS / "modelos_h2_resumen.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(summ)


if __name__ == "__main__":
    main()
