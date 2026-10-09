"""
ets_match.py
------------
Issue #2 (criterio externo gratuito): asigna instalaciones del EU ETS a las 49
empresas del panel y construye el panel empresa-ano de emisiones verificadas.

    python -I scripts/ets_match.py --results-dir <ruta a results/ del clon principal>

Cruce instalacion -> empresa (perimetro "alguna vez del grupo"):
  a) LEI: el LEI del operador actual (lista de operadores de la Comision) o de
     cualquier titular de una cuenta de operador ligada a la instalacion
     (EUETS.INFO, cuentas abiertas y cerradas) pertenece a la matriz o a una de
     sus filiales consolidadas segun GLEIF Level 2.
  b) Nombre: el nombre de esos titulares o de la instalacion casa con las
     expresiones de data/issue2_ets/ets_firms_internal.csv (y no con las de
     exclusion). Las coincidencias solo por nombre de instalacion se marcan
     para revision.
  Revision manual: data/issue2_ets/ets_review_internal.csv (columna drop=1
  para descartar un acierto); se aplica si existe.

Ambito: instalaciones fijas (cuentas de operador) de la UE-27 + EEE. Se
excluyen el registro britanico (GB, fuera del EU ETS desde 2021) e Irlanda del
Norte (XI) para no introducir un salto artificial en 2021. Aviacion y
transporte maritimo se cuentan aparte y no entran en el panel principal.

Salidas internas (data/issue2_ets/, no versionadas, con nombres):
    ets_matches_internal.csv       instalacion, empresa, via de cruce, nombres
    ets_panel_internal.csv         empresa-ano (firm + firm_id)
Salidas anonimas (results/issue2_ets/):
    cobertura_ets.csv              por firm_id: n.o instalaciones, via, emisiones
    cobertura_ets_resumen.json     cobertura por industria y totales
    panel_ets.csv                  firm_id-ano: emisiones fijas (t CO2e), aviacion
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "issue2_ets"
RAW = DATA / "raw"
OUT = ROOT / "results" / "issue2_ets"
YEARS = list(range(2013, 2026))
EXCL_REG = {"GB", "XI"}


def norm(s) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower() if isinstance(s, str) else ""


def load_ve() -> pd.DataFrame:
    ve = pd.read_excel(RAW / "verified_emissions_2025_en.xlsx", sheet_name="data", header=2, dtype=str)
    ve = ve.rename(columns={"REGISTRY_CODE": "reg", "INSTALLATION_IDENTIFIER": "inst",
                            "INSTALLATION_NAME": "inst_name_ve", "IDENTIFIER_IN_REG": "id_in_reg",
                            "MAIN_ACTIVITY_TYPE_CODE": "activity", "ACCOUNT_TYPE": "acc_type"})
    ve["inst"] = ve["inst"].astype(str).str.strip()
    for y in YEARS:
        ve[f"ve_{y}"] = pd.to_numeric(ve[f"VERIFIED_EMISSIONS_{y}"], errors="coerce")
    ve["kind"] = np.select([ve.acc_type.str.startswith("OPERATOR"), ve.acc_type.str.startswith("AIRCRAFT"),
                            ve.acc_type.str.startswith("MARITIME")], ["fija", "aviacion", "maritimo"], "otro")
    return ve[["reg", "inst", "inst_name_ve", "id_in_reg", "activity", "kind"] + [f"ve_{y}" for y in YEARS]]


def load_names() -> pd.DataFrame:
    """Una fila por (instalacion, nombre de titular o de instalacion, LEI, fuente)."""
    op = pd.read_excel(RAW / "policy_ets_registry_operators_ets_en.xlsx", sheet_name="data", dtype=str)
    op = op.rename(columns={"MS Registry": "reg", "Installation ID": "inst", "Account Holder Name": "holder",
                            "LEI": "lei", "Installation Name": "inst_name"})
    op["inst"] = op["inst"].astype(str).str.strip()
    rows = [op.assign(src="operador_actual")[["reg", "inst", "holder", "lei", "inst_name", "src"]]]
    z = zipfile.ZipFile(RAW / "eutl_2024_202410.zip")
    acc = pd.read_csv(z.open("account.csv"), dtype=str, usecols=["accountHolder_id", "installation_id"])
    ah = pd.read_csv(z.open("account_holder.csv"), dtype=str, usecols=["id", "name", "legalEntityIdentifier"])
    ins = pd.read_csv(z.open("installation.csv"), dtype=str, usecols=["id", "name", "registry_id"])
    acc = acc.dropna(subset=["installation_id"]).merge(ah, left_on="accountHolder_id", right_on="id", how="left")
    acc["reg"] = acc.installation_id.str.split("_").str[0]
    acc["inst"] = acc.installation_id.str.split("_", n=1).str[1]
    acc = acc.merge(ins[["id", "name"]].rename(columns={"id": "installation_id", "name": "inst_name"}),
                    on="installation_id", how="left")
    rows.append(acc.rename(columns={"name": "holder", "legalEntityIdentifier": "lei"})
                .assign(src="titular_euetsinfo")[["reg", "inst", "holder", "lei", "inst_name", "src"]])
    nm = pd.concat(rows, ignore_index=True)
    nm["holder_n"] = nm.holder.map(norm)
    nm["inst_name_n"] = nm.inst_name.map(norm)
    # El registro escribe los LEI con guiones ("5299-00...-82"): se normalizan a 20 caracteres
    nm["lei"] = nm.lei.fillna("").str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
    nm.loc[nm.lei.str.len() != 20, "lei"] = ""
    return nm


def match(nm: pd.DataFrame, firms: pd.DataFrame, parents: pd.DataFrame, kids: pd.DataFrame) -> pd.DataFrame:
    lei2firm = {}
    for p in parents.itertuples():
        if isinstance(p.lei, str):
            lei2firm[p.lei] = p.firm
    pl = parents.set_index("lei")["firm"].to_dict()
    for k in kids.itertuples():
        lei2firm.setdefault(k.child_lei, pl.get(k.parent_lei))
    out = []
    for f in firms.itertuples():
        pat = re.compile(f.patterns) if isinstance(f.patterns, str) and f.patterns else None
        exc = re.compile(f.exclude) if isinstance(f.exclude, str) and f.exclude else None

        def ok(s: str) -> bool:
            return bool(pat and s and pat.search(s) and not (exc and exc.search(s)))

        m_lei = nm.lei.map(lei2firm).eq(f.firm)
        m_h = nm.holder_n.map(ok)
        m_i = nm.inst_name_n.map(ok)
        hit = nm[m_lei | m_h | m_i].copy()
        hit["via"] = np.where(m_lei[hit.index], "lei", np.where(m_h[hit.index], "nombre_titular", "nombre_instalacion"))
        hit["firm"] = f.firm
        out.append(hit)
    h = pd.concat(out, ignore_index=True)
    rank = {"lei": 0, "nombre_titular": 1, "nombre_instalacion": 2}
    h["r"] = h.via.map(rank)
    agg = (h.sort_values("r").groupby(["reg", "inst", "firm"])
           .agg(via=("via", "first"), holders=("holder", lambda s: " | ".join(sorted(set(map(str, s.dropna())))[:6])),
                inst_name=("inst_name", "first"), srcs=("src", lambda s: ",".join(sorted(set(s)))))
           .reset_index())
    return agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True, type=Path)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    firms = pd.read_csv(DATA / "ets_firms_internal.csv", sep=";", encoding="utf-8", keep_default_na=False)
    parents = pd.read_csv(DATA / "gleif" / "parents_internal.csv", sep=";", dtype=str)
    kids = pd.read_csv(DATA / "gleif" / "children_internal.csv", sep=";", dtype=str)
    ids = pd.read_csv(a.results_dir / "analysis_v3" / "firm_ids_internal.csv", sep=";", encoding="utf-8")
    doc = pd.read_csv(a.results_dir / "analysis_v3" / "doc_level.csv", sep=";")
    meta = doc.groupby("firm_id")[["country", "industry", "supersector"]].first().reset_index()

    ve = load_ve()
    nm = load_names()
    m = match(nm, firms, parents, kids)
    m = m.merge(ve[["reg", "inst", "kind", "activity", "inst_name_ve"]], on=["reg", "inst"], how="left")
    m = m[~m.reg.isin(EXCL_REG)]

    # Revision manual (opcional)
    rev = DATA / "ets_review_internal.csv"
    if rev.exists():
        r = pd.read_csv(rev, sep=";", dtype=str, encoding="utf-8")
        drop = r[r["drop"].fillna("0").eq("1")][["reg", "inst", "firm"]]
        m = m.merge(drop.assign(_d=1), on=["reg", "inst", "firm"], how="left")
        m = m[m._d.isna()].drop(columns="_d")
    # Una instalacion, una empresa: prioridad a la via mas fiable
    m["r"] = m.via.map({"lei": 0, "nombre_titular": 1, "nombre_instalacion": 2})
    m["n_firms"] = m.groupby(["reg", "inst"]).firm.transform("nunique")
    m = m.sort_values(["reg", "inst", "r"]).drop_duplicates(["reg", "inst"], keep="first")
    m = m.merge(ids, on="firm", how="left")
    m.drop(columns="r").to_csv(DATA / "ets_matches_internal.csv", sep=";", index=False, encoding="utf-8")

    # Panel empresa-ano
    mv = m.merge(ve, on=["reg", "inst", "kind", "activity", "inst_name_ve"], how="left")
    long = mv.melt(id_vars=["firm", "firm_id", "reg", "inst", "kind", "via"],
                   value_vars=[f"ve_{y}" for y in YEARS], var_name="year", value_name="ve")
    long["year"] = long.year.str[3:].astype(int)
    fixed = long[long.kind == "fija"]
    # like-for-like: instalaciones con emisiones > 0 en t y t-1 (cambio sin altas ni bajas)
    fx = fixed.pivot_table(index=["firm_id", "reg", "inst"], columns="year", values="ve", aggfunc="sum")
    lfl = {}
    for y in YEARS[1:]:
        both = fx[(fx[y] > 0) & (fx[y - 1] > 0)]
        g = both.groupby(level="firm_id")
        lfl[y] = np.log(g[y].sum()) - np.log(g[y - 1].sum())
    lfl = pd.DataFrame(lfl).stack().rename("dlog_lfl").reset_index().rename(columns={"level_1": "year"})
    lfl.columns = ["firm_id", "year", "dlog_lfl"]
    pan = (fixed.groupby(["firm_id", "year"]).agg(em_fixed=("ve", lambda s: s.sum(min_count=1)),
                                                   n_inst_active=("ve", lambda s: int((s > 0).sum())))
           .reset_index())
    av = long[long.kind == "aviacion"].groupby(["firm_id", "year"]).ve.sum().rename("em_aviation").reset_index()
    pan = pan.merge(av, on=["firm_id", "year"], how="left").merge(lfl, on=["firm_id", "year"], how="left")
    pan["em_aviation"] = pan.em_aviation.fillna(0.0)
    pan = pan.merge(ids, on="firm_id")
    pan.to_csv(DATA / "ets_panel_internal.csv", sep=";", index=False, encoding="utf-8")
    pan.drop(columns="firm").to_csv(OUT / "panel_ets.csv", sep=";", index=False)

    # Cobertura anonima
    cov = (m.groupby(["firm_id", "kind"]).agg(n_inst=("inst", "size"),
                                                via_lei=("via", lambda s: int((s == "lei").sum())),
                                                via_nombre=("via", lambda s: int((s != "lei").sum())))
           .reset_index().pivot_table(index="firm_id", columns="kind", values=["n_inst", "via_lei", "via_nombre"],
                                      fill_value=0))
    cov.columns = [f"{a}_{b}" for a, b in cov.columns]
    cov = cov.reset_index()
    em = pan.pivot_table(index="firm_id", columns="year", values="em_fixed")
    cov = meta.merge(cov, on="firm_id", how="left").fillna({c: 0 for c in cov.columns if c != "firm_id"})
    for y in (2018, 2024):
        cov[f"em_fixed_{y}_kt"] = cov.firm_id.map((em[y] / 1e3).round(1) if y in em else {})
    pos = pan[(pan.year.between(2018, 2025)) & (pan.em_fixed > 0)].firm_id.unique()
    cov["cubierta"] = cov.firm_id.isin(pos)
    cov.sort_values(["industry", "firm_id"]).to_csv(OUT / "cobertura_ets.csv", sep=";", index=False)
    by_ind = cov.groupby("industry").agg(n_firms=("firm_id", "size"), n_cubiertas=("cubierta", "sum"),
                                         inst_fijas=("n_inst_fija", "sum")).reset_index()
    summ = {"n_firms": int(len(cov)), "n_cubiertas_fijas": int(cov.cubierta.sum()),
            "n_con_aviacion": int((cov.get("n_inst_aviacion", 0) > 0).sum()),
            "n_instalaciones_fijas": int(cov.get("n_inst_fija", pd.Series([0])).sum()),
            "instalaciones_por_via": m[m.kind == "fija"].via.value_counts().to_dict(),
            "instalaciones_multi_empresa": int((m.n_firms > 1).sum()),
            "por_industria": by_ind.to_dict(orient="records")}
    (OUT / "cobertura_ets_resumen.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summ, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
