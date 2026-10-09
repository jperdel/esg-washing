"""
ctrv_power.py
-------------
Issue #2 (H2): potencia y efecto minimo detectable (MDE) de los modelos de
recuento con las tasas base OBSERVADAS en el panel gratuito de controversias.
Reutiliza la logica de fin_power.py (regresor real: ESGSI de los 343
informes; solo se simula el resultado), pero calibra el resultado a cada
recuento observado:

    y_it ~ Poisson(lambda * exp(u_i - s_u^2/2) * g_it * exp(beta * x_it))
    u_i ~ N(0, s_u)                       heterogeneidad entre empresas
    g_it ~ Gamma(1/a_w, a_w)              sobredispersion intra-empresa (NB2)

lambda = media observada 2018-2025; s_u y a_w por momentos a partir de la
varianza entre medias de empresa y de la varianza intra-empresa.

Estimadores: (A) Poisson con EF de ano; (B) Poisson con EF de empresa y ano.
SE agrupados por empresa, t con G-1 g.l., bilateral al 5 %, potencia 80 %.

    python -I scripts/ctrv_power.py [--reps 300] [--outcomes n_bhrrc,...]
Salidas: results/issue2_controversies/potencia_recuento_observado.csv y
potencia_resumen.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ctrv_common import RESULTS, firm_panel  # noqa: E402
from fin_econ import dummies, poisson_cl  # noqa: E402
from fin_power import ALPHA, SEED, mde  # noqa: E402

DEFAULT_OUT = ["n_bhrrc", "m_bhrrc", "n_bhrrc_E", "n_bhrrc_sev2", "n_pool", "n_oecd", "n_ads", "gd_neg"]


def calibrate(p: pd.DataFrame, col: str) -> dict:
    w = p[p["year"].between(2018, 2025)][["firm_id", col]].dropna()
    lam = w[col].mean()
    g = w.groupby("firm_id")[col]
    m, s2, T = g.mean(), g.var(ddof=1), g.size().mean()
    vb = max(m.var(ddof=1) - s2.mean() / T, 1e-9)
    s_u = float(np.sqrt(np.log1p(vb / lam ** 2))) if lam > 0 else 0.0
    a_w = float(max((s2 - m).sum() / max((m ** 2).sum(), 1e-9), 0.0))
    return {"lambda": float(lam), "sigma_u": s_u, "alpha_intra": a_w,
            "pct_con_evento": float(100 * (w[col] > 0).mean())}


def simulate(d: pd.DataFrame, cal: dict, reps: int, grid: np.ndarray, rng) -> pd.DataFrame:
    fc = pd.factorize(d["firm_id"])[0]
    G = fc.max() + 1
    x = d["x"].to_numpy()
    XA = np.column_stack([x, np.ones(len(d)), dummies(d["year"])])
    XB = np.column_stack([x, np.ones(len(d)), dummies(d["year"]), dummies(d["firm_id"])])
    lam, su, aw = cal["lambda"], cal["sigma_u"], cal["alpha_intra"]
    rows = []
    for beta in grid:
        rA = rB = nA = nB = 0
        for _ in range(reps):
            u = rng.normal(0, su, G)[fc] - su ** 2 / 2
            gmult = rng.gamma(1 / aw, aw, len(d)) if aw > 1e-6 else 1.0
            y = rng.poisson(lam * np.exp(u + beta * x) * gmult).astype(float)
            _, _, pA = poisson_cl(y, XA, fc)
            if np.isfinite(pA):
                nA += 1
                rA += pA < ALPHA
            pos = np.bincount(fc, weights=y, minlength=G)[fc] > 0
            if pos.sum() > 20:
                fcb = pd.factorize(fc[pos])[0]
                Xb = XB[pos]
                Xb = Xb[:, [j for j in range(Xb.shape[1]) if j < 2 or Xb[:, j].std() > 0]]
                _, _, pB = poisson_cl(y[pos], Xb, fcb)
                if np.isfinite(pB):
                    nB += 1
                    rB += pB < ALPHA
        rows.append({"beta_log": beta, "irr": float(np.exp(beta)),
                     "potencia_A_ef_ano": rA / max(nA, 1), "potencia_B_ef_empresa": rB / max(nB, 1),
                     "convergidos_B": nB / reps})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--outcomes", default=",".join(DEFAULT_OUT))
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    p = pd.read_csv(RESULTS / "panel_controversias.csv", sep=";")
    sc = firm_panel()[["firm_id", "year", "ESGSI"]]
    grid = np.round(np.arange(0, 0.81, 0.05), 2)
    allrows, summ = [], {"generado": time.strftime("%Y-%m-%d %H:%M:%S"), "reps": a.reps, "resultados": []}
    for oc in a.outcomes.split(","):
        if oc not in p.columns:
            continue
        cal = calibrate(p, oc)
        for h, yrs in [(1, range(2018, 2025)), (2, range(2018, 2024))]:
            d = sc[sc["year"].isin(list(yrs))].reset_index(drop=True)
            d["x"] = (d["ESGSI"] - d["ESGSI"].mean()) / d["ESGSI"].std(ddof=0)
            r = simulate(d, cal, a.reps, grid, rng)
            r.insert(0, "h", h)
            r.insert(0, "n_obs", len(d))
            r.insert(0, "resultado", oc)
            allrows.append(r)
            summ["resultados"].append({"resultado": oc, "h": h, "n_obs": len(d), **cal,
                                       "mde_irr_A": float(np.exp(mde(r.beta_log.to_numpy(), r.potencia_A_ef_ano.to_numpy()))),
                                       "mde_irr_B": float(np.exp(mde(r.beta_log.to_numpy(), r.potencia_B_ef_empresa.to_numpy())))})
            print(summ["resultados"][-1], flush=True)
    pd.concat(allrows).to_csv(RESULTS / "potencia_recuento_observado.csv", sep=";", index=False)
    (RESULTS / "potencia_resumen.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False),
                                                    encoding="utf-8")


if __name__ == "__main__":
    main()
