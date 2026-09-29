"""
corpus_composition.py
---------------------
Composicion del corpus para la seccion 3 del paper (datos): documentos,
empresas, paises y tipos de documento por pais, ano, industria ICB,
supersector STOXX/ICB y empresa. Los conteos no dependen del vocabulario,
solo de que documentos entran en results.csv (343 tras la deduplicacion).

    PYTHONIOENCODING=utf-8 python scripts/corpus_composition.py
    python scripts/corpus_composition.py --results results/_prev_v3_sec/results.csv

Salidas (separador ';'):
    results/analysis_v3/corpus_composition.csv
        dimension;group;firms;documents;countries_n;countries;
        annual_report;urd;sustainability_report
    results/analysis_v3/corpus_industry_x_country.csv
        empresas por industria (filas) y pais (columnas)

Reutiliza las etiquetas y el mapa supersector -> industria de
scripts/analysis_v3.py para que ambas salidas usen la misma agrupacion.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from analysis_v3 import load_results  # noqa: E402  (misma carga, mismo cruce de sectores)

DOCTYPES = [("Annual report", "annual_report"), ("URD", "urd"),
            ("Sustainability report", "sustainability_report")]


def composition(df: pd.DataFrame, by: str | None) -> pd.DataFrame:
    g = df.assign(_all="All") if by is None else df
    key = "_all" if by is None else by
    rows = []
    for grp, d in g.groupby(key):
        ctry = sorted(d["country"].unique())
        r = {"dimension": by or "all", "group": grp, "firms": d["firm"].nunique(),
             "documents": len(d), "countries_n": len(ctry), "countries": ", ".join(ctry)}
        for lab, col in DOCTYPES:
            r[col] = int((d["doctype"] == lab).sum())
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=BASE_DIR / "results" / "metrics_v3_sec" / "results.csv")
    ap.add_argument("--sectors", type=Path, default=BASE_DIR / "metadata" / "empresas_supersector.csv")
    ap.add_argument("--out", type=Path, default=BASE_DIR / "results" / "analysis_v3")
    a = ap.parse_args()

    df = load_results(a.results, a.sectors)
    a.out.mkdir(parents=True, exist_ok=True)

    comp = pd.concat([composition(df, None)] +
                     [composition(df, by) for by in ("country", "year", "industry", "supersector", "firm")],
                     ignore_index=True)
    # Las filas por empresa llevan el nombre de carpeta: van a un fichero
    # *_internal, que nunca se reproduce en el paper (directriz: sin nombres).
    is_firm = comp["dimension"] == "firm"
    comp[~is_firm].to_csv(a.out / "corpus_composition.csv", sep=";", index=False, encoding="utf-8")
    comp[is_firm].to_csv(a.out / "corpus_composition_internal.csv", sep=";", index=False, encoding="utf-8")

    firms = df.drop_duplicates("firm")
    ixc = pd.crosstab(firms["industry"], firms["country"])
    ixc.reset_index().to_csv(a.out / "corpus_industry_x_country.csv", sep=";", index=False, encoding="utf-8")

    # Comprobaciones del panel: 49 por ano y conteo por pais constante en cada ano.
    per_year = df.groupby("year").size()
    cy = pd.crosstab(df["country"], df["year"])
    balanced_country = bool((cy.nunique(axis=1) == 1).all())
    per_firm = df.groupby("firm").size()

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(f"fuente: {a.results}")
    print(f"documentos {len(df)}, empresas {df['firm'].nunique()}, anos {sorted(df['year'].unique())}")
    print(f"por ano: {per_year.to_dict()}; conteo por pais constante en cada ano: {balanced_country}")
    print(f"documentos por empresa: min {per_firm.min()}, max {per_firm.max()}")
    print(comp.to_string(index=False))
    print(ixc.to_string())


if __name__ == "__main__":
    main()
