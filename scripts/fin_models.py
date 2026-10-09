"""
fin_models.py
-------------
Issue #2: primer modelo exploratorio para H1 y H2 con el ESGSI y el ESGSI
extendido, y validacion del pipeline sobre datos simulados.

    python scripts/fin_models.py --results-dir <ruta a results/> [--outcomes <csv>]
                                 [--sim-reps 200] [--n-boot 9999]

Tres bloques:

1. PROXIES DE MERCADO (datos reales y gratuitos; requiere haber ejecutado
   fin_fetch_market.py). Para el informe del ejercicio t (publicado hacia
   febrero-abril de t+1) se mide en la ventana [1-abr-(t+1), 31-mar-(t+2)],
   con rendimientos semanales y un modelo de mercado con el indice EURO STOXX 50
   (retardo, contemporaneo y adelanto):
     - log_ivol  volatilidad idiosincratica anualizada (log): proxy de
                 incertidumbre informativa. NO es la dispersion de analistas de
                 H1; solo un sustituto de la misma familia.
     - ncskew, duvol, crash  asimetria negativa de los rendimientos
                 especificos y semana de caida extrema (< media - 3,09 sd):
                 proxies de acumulacion y revelacion de malas noticias. NO son
                 controversias ESG (H2); son su analogo de mercado.
   Controles medidos en el ano natural t: rendimiento, volatilidad y asimetria
   previas y tamano aproximado (log de acciones actuales x precio a cierre de t).

2. DATOS EXTERNOS (I/B/E/S, RepRisk, ...). Si --outcomes apunta a un CSV con
   el esquema de metadata/issue2_esquema_outcomes.csv se estiman H1 y H2 con
   esos datos. Si no existe, se omite.

3. VALIDACION SIMULADA. Se generan resultados con el esquema anterior sobre el
   panel real de scores, con efectos conocidos y un choque de mandato que
   afecta al resultado, y se comprueba sesgo y cobertura de cada
   especificacion. Valida el pipeline; no dice nada sobre los datos reales.

Especificaciones (todas con SE agrupados por empresa, t con G-1 g.l.):
    A  agrupada: EF de ano, industria y pais + controles
    B  EF de empresa y de ano + controles (solo variacion intra-empresa)
    C  EF de empresa y de regimen x ano (regimen = empresa financiera a efectos
       del art. 8 de la Taxonomia x pais con la CSRD en vigor para el
       ejercicio 2024): absorbe cualquier choque de mandato comun a cada regimen
    D  B + score x mandato prescrito (Taxonomia: alineamiento desde 2022 las no
       financieras y desde 2023 las financieras; CSRD en 2024 donde estaba
       transpuesta): cambia la asociacion cuando el contenido es obligatorio?
    E  B con Z(SEN) y Z(SUS) por separado: tono frente a sustancia
Resultados binarios: A por logit; B-E por modelo lineal de probabilidad.

Directrices: el score es continuo y se usa como tal (sin umbrales ni
etiquetas); sin nombres de empresa; el panel con resultados de mercado queda
en data/issue2_finance/ (interno) y en results/ solo van coeficientes.
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
from fin_econ import dummies, logit_cl, ols_cl, twoway_demean, wild_cluster_p  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INDEX = "^STOXX50E"
SEED = 20261009
# Paises cuya transposicion de la CSRD aplicaba ya al ejercicio 2024 de las
# grandes EIP. A VERIFICAR por el autor (Belgica transpuso en dic-2024;
# Alemania, Espana y Paises Bajos no habian transpuesto a cierre de 2024).
CSRD_FY2024 = {"France", "Italy", "Finland", "Belgium"}
SCORES = ["ESGSI", "ESGSI_ext"]


# ---------------------------------------------------------------------------
# Panel de scores y exposicion a mandatos
# ---------------------------------------------------------------------------

def load_scores(results_dir: Path) -> pd.DataFrame:
    d = pd.read_csv(results_dir / "analysis_v3" / "doc_level.csv", sep=";")
    fmap_p = ROOT / "data" / "issue2_finance" / "firm_map_internal.csv"
    if fmap_p.exists():
        fm = pd.read_csv(fmap_p, sep=";")[["firm_id", "sector_financiero", "primer_ejercicio_ticker"]]
        d = d.merge(fm, on="firm_id", how="left")
    else:
        d["sector_financiero"] = (d["industry"] == "Financials").astype(int)
        d["primer_ejercicio_ticker"] = 2018
    return add_mandates(d.sort_values(["firm_id", "year"]).reset_index(drop=True))


def add_mandates(d: pd.DataFrame) -> pd.DataFrame:
    fin = d["sector_financiero"].astype(int)
    csrd_pais = d["country"].isin(CSRD_FY2024)
    d["taxo_alineamiento"] = (((fin == 0) & (d["year"] >= 2022)) | ((fin == 1) & (d["year"] >= 2023))).astype(int)
    d["csrd_vigente"] = ((d["year"] == 2024) & csrd_pais).astype(int)
    d["mandato_prescrito"] = ((d["taxo_alineamiento"] == 1) | (d["csrd_vigente"] == 1)).astype(int)
    d["regimen"] = "fin" + fin.astype(str) + "_csrd" + csrd_pais.astype(int).astype(str)
    d["regimen_ano"] = d["regimen"] + "_" + d["year"].astype(str)
    return d


# ---------------------------------------------------------------------------
# Proxies de mercado
# ---------------------------------------------------------------------------

def _mm_resid(ri: pd.Series, rm: pd.Series) -> np.ndarray | None:
    df = pd.DataFrame({"ri": ri, "m0": rm, "m_1": rm.shift(1), "m1": rm.shift(-1)}).dropna()
    if len(df) < 40:
        return None
    X = np.column_stack([np.ones(len(df)), df[["m_1", "m0", "m1"]].to_numpy()])
    b = np.linalg.lstsq(X, df["ri"].to_numpy(), rcond=None)[0]
    return df["ri"].to_numpy() - X @ b


def _crash_stats(w: np.ndarray) -> dict:
    n = len(w)
    w = w - w.mean()
    s2, s3 = (w ** 2).sum(), (w ** 3).sum()
    ncskew = -(n * (n - 1) ** 1.5 * s3) / ((n - 1) * (n - 2) * s2 ** 1.5)
    down, up = w[w < 0], w[w >= 0]
    duvol = np.log(((len(up) - 1) * (down ** 2).sum()) / ((len(down) - 1) * (up ** 2).sum()))
    sd = w.std(ddof=1)
    return {"ivol": sd * np.sqrt(52), "ncskew": ncskew, "duvol": duvol,
            "crash": int((w < -3.09 * sd).any())}


def market_panel(d: pd.DataFrame) -> pd.DataFrame:
    px = pd.read_csv(ROOT / "data" / "issue2_finance" / "prices_daily.csv", sep=";",
                     index_col=0, parse_dates=True)
    fm = pd.read_csv(ROOT / "data" / "issue2_finance" / "firm_map_internal.csv", sep=";")
    wk = px.resample("W-FRI").last()
    r = np.log(wk).diff()
    rm = r[INDEX]
    rows = []
    for _, f in fm.iterrows():
        tk = f["ticker_yahoo"]
        if tk not in r:
            continue
        for t in range(2018, 2025):
            rec = {"firm_id": f["firm_id"], "year": t}
            if t < f["primer_ejercicio_ticker"]:
                rows.append(rec)
                continue
            # ventana de resultado: tras la publicacion del informe de t
            m = (r.index > pd.Timestamp(f"{t + 1}-04-01")) & (r.index <= pd.Timestamp(f"{t + 2}-03-31"))
            e = _mm_resid(r.loc[m, tk], rm[m])
            if e is not None:
                cs = _crash_stats(e)
                rec.update(log_ivol=np.log(cs["ivol"]), ncskew=cs["ncskew"], duvol=cs["duvol"],
                           crash=cs["crash"])
            # controles: ano natural t
            mc = (r.index > pd.Timestamp(f"{t}-01-01")) & (r.index <= pd.Timestamp(f"{t}-12-31"))
            ec = _mm_resid(r.loc[mc, tk], rm[mc])
            if ec is not None:
                cc = _crash_stats(ec)
                rec.update(log_ivol_prev=np.log(cc["ivol"]), ncskew_prev=cc["ncskew"],
                           ret_prev=float(r.loc[mc, tk].sum()))
            p_end = px.loc[:f"{t}-12-31", tk].dropna()
            if len(p_end) and pd.notna(f.get("acciones_en_circulacion")):
                rec["log_size"] = float(np.log(p_end.iloc[-1] * f["acciones_en_circulacion"]))
            rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Especificaciones
# ---------------------------------------------------------------------------

def _std(v: pd.Series) -> pd.Series:
    return (v - v.mean()) / v.std(ddof=0)


def estimate(d: pd.DataFrame, y: str, score: str, controls: list[str], binary: bool,
             n_boot: int, specs: str = "ABCDE") -> list[dict]:
    cols = list(dict.fromkeys([y, score, "Z_SEN", "Z_SUS"] + controls))
    s = d.dropna(subset=cols).reset_index(drop=True).copy()
    s["S"] = _std(s[score])
    s["SEN"] = _std(s["Z_SEN"])
    s["SUS"] = _std(s["Z_SUS"])
    for c in controls:
        s[c] = _std(s[c])
    fc = pd.factorize(s["firm_id"])[0]
    yc = pd.factorize(s["year"])[0]
    rc = pd.factorize(s["regimen_ano"])[0]
    G = fc.max() + 1
    Y = s[y].to_numpy(float)
    sd_y = float(Y.std())
    C = s[controls].to_numpy(float) if controls else np.empty((len(s), 0))
    base = dict(resultado=y, score=score, n=len(s), empresas=int(G), sd_resultado=sd_y)
    out = []

    def rec(spec, term, b, se, p, p_wild=None, modelo="MCO"):
        out.append({**base, "especificacion": spec, "termino": term, "modelo": modelo,
                    "beta": b, "se_cluster": se, "p_cluster": p, "p_wild": p_wild,
                    "beta_sd_resultado": b / sd_y if (not binary and sd_y > 0) else None})

    if "A" in specs:
        XA = np.column_stack([s["S"], np.ones(len(s)), C, dummies(s["year"]),
                              dummies(s["industry"]), dummies(s["country"])])
        if binary:
            b, se, p = logit_cl(Y, XA, fc)
            if not np.isfinite(b):   # separacion: solo EF de ano
                b, se, p = logit_cl(Y, np.column_stack([s["S"], np.ones(len(s)), C, dummies(s["year"])]), fc)
            rec("A", "score", b, se, p, modelo="logit (log-odds)")
        else:
            rec("A", "score", *ols_cl(Y, XA, fc))

    def fe_block(spec, regs, names, codes2, df_abs):
        X = np.column_stack(regs)
        Xd = twoway_demean(X, fc, codes2)
        Yd = twoway_demean(Y, fc, codes2)
        for j, nm in enumerate(names):
            b, se, p = ols_cl(Yd, Xd, fc, j=j, df_absorbed=df_abs)
            pw = (wild_cluster_p(Yd, Xd, fc, j=j, n_boot=n_boot)
                  if (n_boot and nm in ("score", "score_x_mandato", "Z_SEN", "Z_SUS")) else None)
            rec(spec, nm, b, se, p, pw, modelo="LPM" if binary else "MCO")

    T, R = yc.max() + 1, rc.max() + 1
    ctrl = [C[:, k] for k in range(C.shape[1])]
    if "B" in specs:
        fe_block("B", [s["S"].to_numpy()] + ctrl, ["score"] + controls, yc, G + T - 1)
    if "C" in specs:
        fe_block("C", [s["S"].to_numpy()] + ctrl, ["score"] + controls, rc, G + R - 1)
    if "D" in specs:
        inter = s["S"].to_numpy() * s["mandato_prescrito"].to_numpy()
        fe_block("D", [s["S"].to_numpy(), inter, s["mandato_prescrito"].to_numpy(float)] + ctrl,
                 ["score", "score_x_mandato", "mandato"] + controls, yc, G + T - 1)
    if "E" in specs:
        fe_block("E", [s["SEN"].to_numpy(), s["SUS"].to_numpy()] + ctrl, ["Z_SEN", "Z_SUS"] + controls,
                 yc, G + T - 1)
    return out


# ---------------------------------------------------------------------------
# Validacion simulada (esquema de datos externos)
# ---------------------------------------------------------------------------

def simulate_external(d: pd.DataFrame, rng, beta_h1: float = 0.3, or_h2: float = 1.8) -> pd.DataFrame:
    """Genera un fichero con el esquema de metadata/issue2_esquema_outcomes.csv.
    El mandato afecta directamente al resultado (choque por regimen x ano)."""
    s = d[["firm_id", "year", "country", "industry", "ESGSI", "ESGSI_ext", "Z_SEN", "Z_SUS",
           "regimen_ano", "mandato_prescrito", "csrd_vigente", "taxo_alineamiento"]].copy()
    G = s["firm_id"].nunique()
    fid = pd.factorize(s["firm_id"])[0]
    S = _std(s["ESGSI"]).to_numpy()
    a = rng.normal(0, np.sqrt(0.5), G)[fid]
    shock = 0.6 * s["taxo_alineamiento"].to_numpy() + 0.8 * s["csrd_vigente"].to_numpy()
    s["log_mcap"] = 11 + rng.normal(0, 1, G)[fid] + rng.normal(0, 0.15, len(s))
    s["n_analistas"] = np.clip(np.round(np.exp(2.9 + 0.2 * (s["log_mcap"] - 11) + rng.normal(0, 0.2, len(s)))), 3, 45)
    s["rating_esg_lag"] = rng.normal(0, 1, G)[fid] + rng.normal(0, 0.3, len(s))
    e = rng.normal(0, np.sqrt(0.5), len(s))
    s["disp_prev"] = beta_h1 * S + shock + a - 0.2 * (s["log_mcap"] - 11) + e
    u = rng.normal(0, 1, G)[fid]
    eta = -1.6 + np.log(or_h2) * S + 0.5 * shock + u + 0.3 * (s["log_mcap"] - 11)
    s["controv_t1"] = (rng.random(len(s)) < 1 / (1 + np.exp(-eta))).astype(int)
    # t+2: solo informes 2018-2022 si las controversias acaban en 2024
    s["controv_t2"] = np.where(s["year"] <= 2022, s.groupby("firm_id")["controv_t1"].shift(-1), np.nan)
    s.loc[s["year"] == 2024, "controv_t1"] = np.nan    # controversias hasta 2024
    return s


def validate(d: pd.DataFrame, reps: int, rng) -> pd.DataFrame:
    ctrl = ["log_mcap", "n_analistas", "rating_esg_lag"]
    rows = []
    for r in range(reps):
        dd = simulate_external(d, rng)
        dd["n_analistas"] = np.log(dd["n_analistas"])
        for spec_rows in (estimate(dd, "disp_prev", "ESGSI", ctrl, False, 0, "ABC"),
                          estimate(dd, "controv_t1", "ESGSI", ctrl, True, 0, "ABC")):
            for x in spec_rows:
                if x["termino"] == "score":
                    rows.append({"rep": r, "resultado": x["resultado"], "especificacion": x["especificacion"],
                                 "modelo": x["modelo"], "beta": x["beta"], "se": x["se_cluster"],
                                 "rechaza_5": x["p_cluster"] < 0.05})
    v = pd.DataFrame(rows)
    # verdad: H1 0,3 en unidades del resultado por sd(ESGSI); H2 log(1,8) en log-odds (solo logit)
    truth = {("disp_prev", "A"): 0.3, ("disp_prev", "B"): 0.3, ("disp_prev", "C"): 0.3,
             ("controv_t1", "A"): np.log(1.8)}
    agg = []
    for (res, sp, mod), g in v.groupby(["resultado", "especificacion", "modelo"]):
        tr = truth.get((res, sp))
        cover = None
        if tr is not None:
            from scipy import stats
            tcrit = stats.t.ppf(0.975, 48)
            cover = float(((g.beta - tcrit * g.se <= tr) & (tr <= g.beta + tcrit * g.se)).mean())
        agg.append({"resultado": res, "especificacion": sp, "modelo": mod, "reps": len(g),
                    "verdad": tr, "beta_medio": g.beta.mean(), "sd_beta": g.beta.std(),
                    "cobertura_ic95": cover, "potencia_5": g.rechaza_5.mean()})
    return pd.DataFrame(agg)


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results")
    ap.add_argument("--outcomes", type=Path, default=ROOT / "data" / "issue2_finance" / "outcomes_externos.csv")
    ap.add_argument("--sim-reps", type=int, default=200)
    ap.add_argument("--n-boot", type=int, default=9999)
    a = ap.parse_args()
    out = ROOT / "results" / "issue2_finance"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    d = load_scores(a.results_dir)
    summary: dict = {"generado": time.strftime("%Y-%m-%d %H:%M:%S"), "semilla": SEED,
                     "csrd_fy2024_paises": sorted(CSRD_FY2024),
                     "regimenes": d.groupby("regimen")["firm_id"].nunique().to_dict()}

    # 1. proxies de mercado
    if (ROOT / "data" / "issue2_finance" / "prices_daily.csv").exists():
        mk = market_panel(d)
        mk.to_csv(ROOT / "data" / "issue2_finance" / "panel_mercado_internal.csv", sep=";", index=False)
        dm = d.merge(mk, on=["firm_id", "year"], how="left")
        res = []
        cr = ["ncskew_prev", "log_ivol_prev", "ret_prev", "log_size"]
        for sc in SCORES:
            sp = "ABCDE" if sc == "ESGSI" else "ABCD"   # E (componentes) no depende del score
            res += estimate(dm, "log_ivol", sc, ["log_ivol_prev", "ret_prev", "log_size"], False, a.n_boot, sp)
            for yv in ["ncskew", "duvol"]:
                res += estimate(dm, yv, sc, cr, False, a.n_boot, sp)
            res += estimate(dm, "crash", sc, cr, True, a.n_boot, sp)
        rt = pd.DataFrame(res)
        rt.to_csv(out / "modelos_proxy_mercado.csv", sep=";", index=False)
        desc = dm[["log_ivol", "ncskew", "duvol", "crash"]].describe().T
        desc.to_csv(out / "descriptivos_proxy_mercado.csv", sep=";")
        summary["proxy_mercado"] = {
            "obs_con_resultado": int(dm["log_ivol"].notna().sum()),
            "tasa_crash": float(dm["crash"].mean()),
            "corr_score_resultado": {sc: {y: float(dm[[sc, y]].corr(method="spearman").iloc[0, 1])
                                          for y in ["log_ivol", "ncskew", "duvol"]} for sc in SCORES},
            "clave": rt[rt["termino"].isin(["score", "score_x_mandato", "Z_SEN", "Z_SUS"])][
                ["resultado", "score", "especificacion", "termino", "modelo", "n", "beta", "se_cluster",
                 "p_cluster", "p_wild", "beta_sd_resultado"]].round(4).astype(object)
                .where(lambda z: z.notna(), None).to_dict(orient="records"),
        }

    # 2. datos externos reales, si existen
    if a.outcomes.exists():
        ext = pd.read_csv(a.outcomes, sep=";")
        de = d.merge(ext, on=["firm_id", "year"], how="inner")
        res = []
        ctrl = [c for c in ["log_mcap", "log_n_analistas", "rating_esg_lag"] if c in de]
        for sc in SCORES:
            if "disp_prev" in de:
                res += estimate(de, "disp_prev", sc, ctrl, False, a.n_boot)
            for yv in ["controv_t1", "controv_t2"]:
                if yv in de:
                    res += estimate(de, yv, sc, ctrl, True, a.n_boot)
        pd.DataFrame(res).to_csv(out / "modelos_datos_externos.csv", sep=";", index=False)
        summary["datos_externos"] = {"fichero": a.outcomes.name, "obs": len(de)}
    else:
        summary["datos_externos"] = "no disponibles: ver metadata/issue2_esquema_outcomes.csv"

    # 3. validacion simulada del pipeline
    v = validate(d, a.sim_reps, rng)
    v.to_csv(out / "validacion_simulada.csv", sep=";", index=False)
    summary["validacion_simulada"] = v.round(4).astype(object).where(v.notna(), None).to_dict(orient="records")
    (out / "modelos_resumen.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=str),
                                             encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
