"""
fin_fetch_market.py
-------------------
Issue #2 (consecuencias financieras de la divergencia tono-sustancia).
Descarga los datos de mercado GRATUITOS que si cubren el panel 2018-2024 y
sondea que ofrece Yahoo Finance (via yfinance) sobre analistas y ESG.

    python scripts/fin_fetch_market.py --results-dir <ruta a results/> \
        [--pylib <dir con yfinance>] [--skip-prices] [--skip-probe]

Entradas
    metadata/empresas_tickers.csv            Empresa -> ticker Yahoo, flag financiero
                                             (empresa financiera a efectos del art. 8 de
                                             la Taxonomia, no ICB) y primer ejercicio cuyo
                                             informe corresponde al emisor del ticker
                                             (un caso de fusion: ejercicios previos fuera)
    <results-dir>/analysis_v3/firm_ids_internal.csv   Empresa -> firm_id anonimo

Salidas
    data/issue2_finance/prices_daily.csv     cierres ajustados diarios (ticker = columna;
                                             fichero interno, no citable: data/ no se versiona)
    data/issue2_finance/firm_map_internal.csv  firm_id <-> ticker (interno)
    results/issue2_finance/cobertura_yfinance.csv   por firm_id anonimo: que devuelve Yahoo
    results/issue2_finance/cobertura_yfinance.json  resumen agregado

Directrices: ninguna salida en results/ lleva nombres ni tickers; solo firm_id
('<Industria>-NN'), industria y pais. Yahoo solo ofrece la FOTO ACTUAL del
consenso (sin historico 2018-2024), por lo que aqui solo se usa para medir
cobertura; el panel historico se construye con precios (volatilidad
idiosincratica y asimetria de rendimientos, ver fin_models.py).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INDEX_TICKER = "^STOXX50E"


def load_map(results_dir: Path) -> pd.DataFrame:
    tk = pd.read_csv(ROOT / "metadata" / "empresas_tickers.csv", sep=";", encoding="utf-8")
    ids = pd.read_csv(results_dir / "analysis_v3" / "firm_ids_internal.csv", sep=";", encoding="utf-8")
    m = tk.merge(ids, left_on="Empresa", right_on="firm", how="outer", indicator=True)
    bad = m[m["_merge"] != "both"]
    if len(bad):
        raise SystemExit(f"Cruce empresa-ticker incompleto: {len(bad)} filas sin pareja")
    doc = pd.read_csv(results_dir / "analysis_v3" / "doc_level.csv", sep=";")
    meta = doc.groupby("firm_id")[["country", "industry", "supersector"]].first().reset_index()
    cols = ["firm_id", "ticker_yahoo", "sector_financiero", "primer_ejercicio_ticker"]
    return m[cols].merge(meta, on="firm_id")


def fetch_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(tickers + [INDEX_TICKER], start=start, end=end, auto_adjust=True,
                     progress=False, group_by="column", threads=True)
    close = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df[["Close"]]
    return close.sort_index()


def probe(ticker: str) -> dict:
    """Que devuelve hoy Yahoo para un ticker (solo foto actual)."""
    import yfinance as yf
    t = yf.Ticker(ticker)
    out: dict = {}
    try:
        info = t.info or {}
    except Exception:
        info = {}
    out["n_analistas_recomendacion"] = info.get("numberOfAnalystOpinions")
    out["tiene_precio_objetivo"] = info.get("targetMeanPrice") is not None
    try:
        ee = t.earnings_estimate
        row = ee.loc["0y"] if ee is not None and "0y" in ee.index else None
    except Exception:
        row = None
    if row is not None and pd.notna(row.get("avg")) and row.get("avg"):
        out["n_analistas_bpa_anual"] = row.get("numberOfAnalysts")
        out["rango_bpa_relativo"] = float((row["high"] - row["low"]) / abs(row["avg"]))
    else:
        out["n_analistas_bpa_anual"] = None
        out["rango_bpa_relativo"] = None
    try:
        s = t.sustainability
        out["esg_sustainalytics_disponible"] = bool(s is not None and len(s) > 0)
    except Exception:
        out["esg_sustainalytics_disponible"] = False
    try:
        ud = t.upgrades_downgrades
        out["historial_cambios_recomendacion"] = 0 if ud is None else int(len(ud))
    except Exception:
        out["historial_cambios_recomendacion"] = 0
    try:
        eh = t.earnings_history
        out["trimestres_sorpresa_bpa"] = 0 if eh is None else int(len(eh))
    except Exception:
        out["trimestres_sorpresa_bpa"] = 0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results")
    ap.add_argument("--pylib", type=Path, default=None,
                    help="directorio extra con yfinance (se anade al FINAL de sys.path)")
    ap.add_argument("--start", default="2017-01-01")
    ap.add_argument("--end", default="2026-07-01")
    ap.add_argument("--skip-prices", action="store_true")
    ap.add_argument("--skip-probe", action="store_true")
    a = ap.parse_args()
    if a.pylib:
        sys.path.append(str(a.pylib))

    data_dir = ROOT / "data" / "issue2_finance"
    out_dir = ROOT / "results" / "issue2_finance"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    fm = load_map(a.results_dir)
    fm.to_csv(data_dir / "firm_map_internal.csv", sep=";", index=False)
    tickers = fm["ticker_yahoo"].tolist()

    summary: dict = {"generado": time.strftime("%Y-%m-%d %H:%M:%S"), "empresas": len(fm)}

    if not a.skip_prices:
        px = fetch_prices(tickers, a.start, a.end)
        px.to_csv(data_dir / "prices_daily.csv", sep=";")
        cov = {}
        for tk in tickers:
            s = px[tk].dropna() if tk in px else pd.Series(dtype=float)
            cov[tk] = (str(s.index.min().date()) if len(s) else None, len(s))
        fm["precio_desde"] = fm["ticker_yahoo"].map(lambda t: cov[t][0])
        fm["dias_con_precio"] = fm["ticker_yahoo"].map(lambda t: cov[t][1])
        summary["precios"] = {
            "rango_pedido": [a.start, a.end],
            "empresas_con_precio_desde_2017": int((fm["precio_desde"] <= "2017-01-31").sum()),
            "empresas_con_precio": int((fm["dias_con_precio"] > 0).sum()),
            "indice_dias": int(px[INDEX_TICKER].notna().sum()) if INDEX_TICKER in px else 0,
        }

    if not a.skip_probe:
        rows = []
        for tk in tickers:
            r = probe(tk)
            r["ticker_yahoo"] = tk
            rows.append(r)
            time.sleep(1.0)
        pr = pd.DataFrame(rows)
        fm = fm.merge(pr, on="ticker_yahoo", how="left")
        summary["yahoo_foto_actual"] = {
            "con_recomendaciones": int(fm["n_analistas_recomendacion"].notna().sum()),
            "n_analistas_mediana": float(fm["n_analistas_recomendacion"].median()),
            "n_analistas_min": float(fm["n_analistas_recomendacion"].min()),
            "n_analistas_max": float(fm["n_analistas_recomendacion"].max()),
            "con_rango_bpa_anual": int(fm["rango_bpa_relativo"].notna().sum()),
            "rango_bpa_relativo_mediana": float(fm["rango_bpa_relativo"].median()),
            "con_esg_sustainalytics": int(fm["esg_sustainalytics_disponible"].sum()),
            "con_historial_cambios_recomendacion": int((fm["historial_cambios_recomendacion"] > 0).sum()),
            "trimestres_sorpresa_bpa_max": int(fm["trimestres_sorpresa_bpa"].max()),
            "nota": "Yahoo da el consenso vigente (hoy), no el historico 2018-2024.",
        }

    pub = fm.drop(columns=["ticker_yahoo"]).sort_values("firm_id")
    pub.to_csv(out_dir / "cobertura_yfinance.csv", sep=";", index=False)
    (out_dir / "cobertura_yfinance.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False),
                                                    encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
