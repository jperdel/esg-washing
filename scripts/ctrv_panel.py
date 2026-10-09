"""
ctrv_panel.py
-------------
Issue #2 (H2): construye el panel empresa-ano de controversias (2017-2026) a
partir de los eventos internos de cada fuente y de las series diarias de
GDELT, y calcula frecuencias por fuente, sector y pais.

    python -I scripts/ctrv_panel.py

Entradas (internas, data/issue2_controversies/):
    events_bhrrc_internal.csv, events_oecdwatch_internal.csv,
    events_adrulings_internal.csv, gdelt_raw/<firm_id>__<tema>.json
Salidas (anonimas, results/issue2_controversies/):
    panel_controversias.csv      firm_id x ano con todos los recuentos
    frecuencias_fuente.csv       media por empresa-ano, % con >=1 evento, DT,
                                 ratio varianza/media, cobertura de empresas
    frecuencias_sector.csv       medias por industria (2018-2025)
    frecuencias_pais.csv         medias por pais (2018-2025)
    frecuencias_ano.csv          medias por ano (2017-2026)

Recuentos (n_ = eventos; sufijos: _sev2 = severidad alta; _E, _S, _G, _GW =
categoria; una alegacion puede tener varias categorias):
    n_bhrrc        noticias/demandas con alegaciones (BHRRC)
    m_bhrrc        meses distintos con >=1 alegacion BHRRC (robusto a
                   coberturas repetidas del mismo incidente)
    n_oecd         quejas ante PNC de la OCDE (OECD Watch)
    n_ads          resoluciones publicitarias (JDP + RCC)
    n_pool         n_bhrrc + n_oecd + n_ads
    gd_all, gd_neg, gd_E, gd_S, gd_G, gd_GW   articulos GDELT por ano
    gd_negdays     dias con >=1 articulo NEG
2026 es un ano parcial (eventos hasta la fecha de descarga; GDELT hasta el
30-sep-2026): `meses_obs` recoge los meses observados para usarlo como
exposicion.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ctrv_common import DATA, RESULTS, firm_panel  # noqa: E402

YEARS = list(range(2017, 2027))
CUTOFF = pd.Timestamp("2026-09-30")


def events(source: str) -> pd.DataFrame:
    f = DATA / f"events_{source}_internal.csv"
    if not f.exists():
        return pd.DataFrame(columns=["firm_id", "date", "year", "category", "severity"])
    e = pd.read_csv(f, sep=";")
    e["date"] = pd.to_datetime(e["date"], errors="coerce")
    e = e[e["date"].notna() & (e["date"] <= CUTOFF)]
    e["year"] = e["date"].dt.year
    e["category"] = e["category"].fillna("")
    return e


def count(e: pd.DataFrame, name: str, idx: pd.MultiIndex) -> pd.DataFrame:
    out = pd.DataFrame(index=idx)
    g = e.groupby(["firm_id", "year"])
    out[f"n_{name}"] = g.size()
    out[f"n_{name}_sev2"] = e[e["severity"] >= 2].groupby(["firm_id", "year"]).size()
    for c in ["E", "S", "G", "GW"]:
        if c == "E":
            m = e["category"].str.contains(r"\bE\b|GW")
        else:
            m = e["category"].str.contains(rf"\b{c}\b")
        out[f"n_{name}_{c}"] = e[m].groupby(["firm_id", "year"]).size()
    return out.fillna(0).astype(int)


def gdelt(idx: pd.MultiIndex) -> pd.DataFrame:
    out = pd.DataFrame(index=idx)
    raw = DATA / "gdelt_raw"
    for topic in ["ALL", "NEG", "E", "S", "G", "GW"]:
        col = "gd_all" if topic == "ALL" else f"gd_{topic if topic != 'NEG' else 'neg'}"
        vals = {}
        days = {}
        for fid in idx.get_level_values(0).unique():
            f = raw / f"{fid}__{topic}.json"
            if not f.exists():
                continue
            d = json.loads(f.read_text(encoding="utf-8"))
            if not d.get("timeline"):
                s = pd.Series(dtype=float)
            else:
                s = pd.DataFrame(d["timeline"][0]["data"])
                s["date"] = pd.to_datetime(s["date"].str[:8])
                s = s.set_index("date")["value"]
            for y in YEARS:
                sy = s[s.index.year == y]
                vals[(fid, y)] = float(sy.sum())
                days[(fid, y)] = int((sy > 0).sum())
        if vals:
            out[col] = pd.Series(vals)
            if topic == "NEG":
                out["gd_negdays"] = pd.Series(days)
    return out


def main() -> None:
    sc = firm_panel()
    firms = sc.groupby("firm_id")[["industry", "country"]].first()
    idx = pd.MultiIndex.from_product([firms.index, YEARS], names=["firm_id", "year"])
    b, o, a = events("bhrrc"), events("oecdwatch"), events("adrulings")
    p = pd.concat([count(b, "bhrrc", idx), count(o, "oecd", idx), count(a, "ads", idx)], axis=1)
    b["ym"] = b["date"].dt.to_period("M")
    p["m_bhrrc"] = b.groupby(["firm_id", "year"])["ym"].nunique().reindex(idx).fillna(0).astype(int)
    for suf in ["", "_sev2", "_E", "_S", "_G", "_GW"]:
        p[f"n_pool{suf}"] = p[f"n_bhrrc{suf}"] + p[f"n_oecd{suf}"] + p[f"n_ads{suf}"]
    g = gdelt(idx)
    p = p.join(g)
    p["meses_obs"] = np.where(p.index.get_level_values("year") == 2026, 9, 12)
    p = p.reset_index().merge(firms, left_on="firm_id", right_index=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    p.to_csv(RESULTS / "panel_controversias.csv", sep=";", index=False)

    cols = [c for c in p.columns if c.startswith(("n_", "m_", "gd_"))]
    win = p[p["year"].between(2018, 2025)]
    rows = []
    for c in cols:
        x = win[c].dropna()
        if x.empty:
            continue
        rows.append({"variable": c, "n_empresa_ano": len(x), "media": x.mean(), "dt": x.std(),
                     "mediana": x.median(), "p90": x.quantile(0.9),
                     "pct_con_evento": 100 * (x > 0).mean(),
                     "ratio_var_media": x.var() / x.mean() if x.mean() > 0 else np.nan,
                     "empresas_con_evento": int((win.loc[x.index].groupby("firm_id")[c].sum() > 0).sum()),
                     "icc_entre_empresas": (win.loc[x.index].groupby("firm_id")[c].transform("mean").var()
                                            / x.var()) if x.var() > 0 else np.nan})
    fr = pd.DataFrame(rows)
    fr.to_csv(RESULTS / "frecuencias_fuente.csv", sep=";", index=False)
    key = ["n_bhrrc", "n_bhrrc_sev2", "n_bhrrc_E", "n_bhrrc_GW", "m_bhrrc", "n_oecd", "n_ads",
           "n_ads_GW", "n_pool", "gd_neg", "gd_all"]
    key = [k for k in key if k in p.columns]
    for by, fn in [("industry", "frecuencias_sector.csv"), ("country", "frecuencias_pais.csv")]:
        t = win.groupby(by)[key].mean()
        t.insert(0, "n_empresas", win.groupby(by)["firm_id"].nunique())
        t["pct_con_bhrrc"] = 100 * win.assign(_x=win["n_bhrrc"] > 0).groupby(by)["_x"].mean()
        t.to_csv(RESULTS / fn, sep=";")
    p.groupby("year")[key].mean().to_csv(RESULTS / "frecuencias_ano.csv", sep=";")
    pd.set_option("display.width", 200)
    print(fr.round(2).to_string(index=False))
    print(p.groupby("year")[key].mean().round(2).to_string())


if __name__ == "__main__":
    main()
