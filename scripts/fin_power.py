"""
fin_power.py
------------
Issue #2: potencia estadistica y efecto minimo detectable (MDE) para H1 y H2
con la estructura REAL del panel (49 empresas, 2018-2024, 343 informes) y los
valores REALES del ESGSI y del ESGSI extendido como regresor; solo el
resultado (outcome) se simula.

    python scripts/fin_power.py --results-dir <ruta a results/> [--reps 500]

H1 (resultado continuo, p. ej. log de la dispersion de previsiones):
    y_it = a_i + d_t + beta * x_it + e_it,   Var(a_i + e_it) = 1,
    a_i ~ N(0, rho), e_it AR(1) con phi = 0.5 dentro de empresa.
    beta en desviaciones tipicas de y por desviacion tipica de x.
    Especificaciones: (A) MCO agrupada con EF de ano, industria y pais;
                      (B) EF de empresa y ano (solo variacion intra-empresa).

H2 (controversia en t+1 o t+2):
    logit P(c_it = 1) = alpha + u_i + d_t + beta * x_it,  u_i ~ N(0, sigma_u).
    Tasas base 5/10/20/40 %; sigma_u en {0, 1} (ICC latente 0 y 0,23).
    Especificaciones: (A) logit agrupado con EF de ano; (B) modelo lineal de
    probabilidad con EF de empresa y ano. Para recuentos: Poisson con EF de ano.
    Muestras: 294 informes (t+1 con controversias hasta 2024 o t+2 hasta 2025)
    y 245 (t+2 con controversias hasta 2024).

Todas las pruebas: SE agrupados por empresa (CR1), t con G-1 = 48 g.l.,
bilateral al 5 %. MDE = menor beta con potencia >= 80 % (interpolacion lineal
en la rejilla). Salidas en results/issue2_finance/: potencia_h1.csv,
potencia_h2.csv, potencia_resumen.json.
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
from fin_econ import dummies, logit_cl, ols_cl, poisson_cl, twoway_demean  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261009
ALPHA = 0.05
TARGET = 0.80


def load_panel(results_dir: Path) -> pd.DataFrame:
    d = pd.read_csv(results_dir / "analysis_v3" / "doc_level.csv", sep=";")
    d = d.sort_values(["firm_id", "year"]).reset_index(drop=True)
    for c in ["ESGSI", "ESGSI_ext"]:
        d[c + "_std"] = (d[c] - d[c].mean()) / d[c].std(ddof=0)
    return d


def mde(grid: np.ndarray, power: np.ndarray) -> float:
    ok = np.where(power >= TARGET)[0]
    if not len(ok):
        return np.nan
    i = ok[0]
    if i == 0:
        return float(grid[0])
    g0, g1, p0, p1 = grid[i - 1], grid[i], power[i - 1], power[i]
    return float(g0 + (TARGET - p0) * (g1 - g0) / (p1 - p0))


def ar1_errors(rng, n_firms: int, T: int, phi: float, var: float) -> np.ndarray:
    e = np.zeros((n_firms, T))
    sd = np.sqrt(var)
    e[:, 0] = rng.normal(0, sd, n_firms)
    innov_sd = sd * np.sqrt(1 - phi ** 2)
    for t in range(1, T):
        e[:, t] = phi * e[:, t - 1] + rng.normal(0, innov_sd, n_firms)
    return e.ravel()


def sim_h1(d: pd.DataFrame, xcol: str, reps: int, rng) -> pd.DataFrame:
    fc = pd.factorize(d["firm_id"])[0]
    yc = pd.factorize(d["year"])[0]
    G, T = fc.max() + 1, yc.max() + 1
    x = d[xcol].to_numpy()
    XA = np.column_stack([x, np.ones(len(d)), dummies(d["year"]), dummies(d["industry"]),
                          dummies(d["country"])])
    xd = twoway_demean(x, fc, yc)
    df_abs_B = G + T - 1
    grid = np.round(np.arange(0, 0.501, 0.025), 3)
    rows = []
    for rho in [0.3, 0.6]:
        for beta in grid:
            rejA = rejB = 0
            seA = []
            seB = []
            for _ in range(reps):
                a = np.repeat(rng.normal(0, np.sqrt(rho), G), T)
                e = ar1_errors(rng, G, T, 0.5, 1 - rho)
                dt = np.tile(rng.normal(0, 0.3, T), G)
                y = a + dt + beta * x + e
                bA, sA, pA = ols_cl(y, XA, fc)
                yd = twoway_demean(y, fc, yc)
                bB, sB, pB = ols_cl(yd, xd[:, None], fc, df_absorbed=df_abs_B)
                rejA += pA < ALPHA
                rejB += pB < ALPHA
                seA.append(sA)
                seB.append(sB)
            rows.append(dict(regresor=xcol.replace("_std", ""), rho_empresa=rho, beta=beta,
                             potencia_A_pooled=rejA / reps, potencia_B_ef_empresa=rejB / reps,
                             se_medio_A=np.mean(seA), se_medio_B=np.mean(seB)))
    return pd.DataFrame(rows)


def calibrate_alpha(base: float, sigma_u: float, rng) -> float:
    u = rng.normal(0, sigma_u, 200_000) if sigma_u > 0 else np.zeros(1)
    lo, hi = -15.0, 15.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(mid + u)))).mean() < base:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def sim_h2(d: pd.DataFrame, xcol: str, years: list[int], reps: int, rng) -> pd.DataFrame:
    s = d[d["year"].isin(years)].reset_index(drop=True)
    fc = pd.factorize(s["firm_id"])[0]
    yc = pd.factorize(s["year"])[0]
    G, T = fc.max() + 1, yc.max() + 1
    x = s[xcol].to_numpy()
    x = (x - x.mean()) / x.std()
    XA = np.column_stack([x, np.ones(len(s)), dummies(s["year"])])
    xd = twoway_demean(x, fc, yc)
    grid = np.round(np.arange(0, 1.01, 0.1), 2)
    rows = []
    for base in [0.05, 0.10, 0.20, 0.40]:
        for sigma_u in [0.0, 1.0]:
            alpha = calibrate_alpha(base, sigma_u, rng)
            for beta in grid:
                rejA = rejB = nA = 0
                dpp = []
                for _ in range(reps):
                    u = np.repeat(rng.normal(0, sigma_u, G), T) if sigma_u > 0 else 0.0
                    dt = np.tile(rng.normal(0, 0.2, T), G)
                    eta = alpha + u + dt + beta * x
                    p = 1 / (1 + np.exp(-eta))
                    y = (rng.random(len(s)) < p).astype(float)
                    dpp.append(np.mean(p * (1 - p)) * beta)
                    bA, sA, pA = logit_cl(y, XA, fc)
                    if np.isfinite(pA):
                        nA += 1
                        rejA += pA < ALPHA
                    yd = twoway_demean(y, fc, yc)
                    _, _, pB = ols_cl(yd, xd[:, None], fc, df_absorbed=G + T - 1)
                    rejB += (pB < ALPHA) if np.isfinite(pB) else 0
                rows.append(dict(regresor=xcol.replace("_std", ""), n_obs=len(s), anos_x=f"{min(years)}-{max(years)}",
                                 tasa_base=base, sigma_u=sigma_u, beta_logodds=beta,
                                 odds_ratio=float(np.exp(beta)), efecto_marginal_pp=100 * float(np.mean(dpp)),
                                 potencia_A_logit=rejA / max(nA, 1), convergidos_A=nA / reps,
                                 potencia_B_lpm_ef=rejB / reps))
    return pd.DataFrame(rows)


def sim_count(d: pd.DataFrame, xcol: str, years: list[int], reps: int, rng) -> pd.DataFrame:
    s = d[d["year"].isin(years)].reset_index(drop=True)
    fc = pd.factorize(s["firm_id"])[0]
    yc = pd.factorize(s["year"])[0]
    G, T = fc.max() + 1, yc.max() + 1
    x = s[xcol].to_numpy()
    x = (x - x.mean()) / x.std()
    XA = np.column_stack([x, np.ones(len(s)), dummies(s["year"])])
    grid = np.round(np.arange(0, 0.61, 0.05), 2)
    rows = []
    for lam in [0.5, 2.0]:
        for sigma_u in [0.0, 1.0]:
            for beta in grid:
                rej = n = 0
                for _ in range(reps):
                    u = np.repeat(rng.normal(0, sigma_u, G), T) if sigma_u > 0 else 0.0
                    mu = lam * np.exp(u - sigma_u ** 2 / 2 + beta * x)
                    y = rng.poisson(mu).astype(float)
                    _, _, p = poisson_cl(y, XA, fc)
                    if np.isfinite(p):
                        n += 1
                        rej += p < ALPHA
                rows.append(dict(regresor=xcol.replace("_std", ""), n_obs=len(s), media_recuento=lam,
                                 sigma_u=sigma_u, beta_log=beta, irr=float(np.exp(beta)),
                                 potencia_poisson=rej / max(n, 1)))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results")
    ap.add_argument("--reps", type=int, default=500)
    a = ap.parse_args()
    out = ROOT / "results" / "issue2_finance"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    d = load_panel(a.results_dir)

    # descomposicion de la varianza del regresor: cuanta queda tras EF empresa+ano
    fc = pd.factorize(d["firm_id"])[0]
    yc = pd.factorize(d["year"])[0]
    var_info = {}
    for c in ["ESGSI_std", "ESGSI_ext_std"]:
        x = d[c].to_numpy()
        var_info[c.replace("_std", "")] = {
            "cuota_varianza_intra_empresa_tras_EF_ano": float(np.var(twoway_demean(x, fc, yc)) / np.var(x)),
            "cuota_entre_empresas": float(np.var(d.groupby("firm_id")[c].transform("mean")) / np.var(x)),
        }

    t0 = time.time()
    h1 = pd.concat([sim_h1(d, c, a.reps, rng) for c in ["ESGSI_std", "ESGSI_ext_std"]])
    h1.to_csv(out / "potencia_h1.csv", sep=";", index=False)
    h2 = pd.concat([sim_h2(d, c, yrs, a.reps, rng)
                    for c in ["ESGSI_std", "ESGSI_ext_std"]
                    for yrs in [list(range(2018, 2024)), list(range(2018, 2023))]])
    h2.to_csv(out / "potencia_h2.csv", sep=";", index=False)
    cnt = sim_count(d, "ESGSI_std", list(range(2018, 2024)), a.reps, rng)
    cnt.to_csv(out / "potencia_h2_recuento.csv", sep=";", index=False)

    summ = {"generado": time.strftime("%Y-%m-%d %H:%M:%S"), "semilla": SEED, "reps": a.reps,
            "alfa": ALPHA, "potencia_objetivo": TARGET, "segundos": round(time.time() - t0, 1),
            "varianza_regresor": var_info, "h1": [], "h2": [], "h2_recuento": []}
    for (reg, rho), g in h1.groupby(["regresor", "rho_empresa"]):
        summ["h1"].append({"regresor": reg, "rho_empresa": rho,
                           "mde_A_pooled_sd": mde(g.beta.to_numpy(), g.potencia_A_pooled.to_numpy()),
                           "mde_B_ef_empresa_sd": mde(g.beta.to_numpy(), g.potencia_B_ef_empresa.to_numpy())})
    for (reg, n, base, su), g in h2.groupby(["regresor", "n_obs", "tasa_base", "sigma_u"]):
        bA = mde(g.beta_logodds.to_numpy(), g.potencia_A_logit.to_numpy())
        bB = mde(g.beta_logodds.to_numpy(), g.potencia_B_lpm_ef.to_numpy())
        summ["h2"].append({"regresor": reg, "n_obs": int(n), "tasa_base": base, "sigma_u": su,
                           "mde_A_logit_OR": float(np.exp(bA)) if np.isfinite(bA) else None,
                           "mde_B_lpm_ef_OR": float(np.exp(bB)) if np.isfinite(bB) else None})
    for (lam, su), g in cnt.groupby(["media_recuento", "sigma_u"]):
        b = mde(g.beta_log.to_numpy(), g.potencia_poisson.to_numpy())
        summ["h2_recuento"].append({"media_recuento": lam, "sigma_u": su,
                                    "mde_IRR": float(np.exp(b)) if np.isfinite(b) else None})
    (out / "potencia_resumen.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summ, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
