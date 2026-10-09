"""
ctrv_sbti.py
------------
Issue #2 (H2): estado de los compromisos climaticos de las 49 empresas en el
Target Dashboard de la Science Based Targets initiative (descarga publica y
gratuita: files.sciencebasedtargets.org/production/files/companies-excel.xlsx
y targets-excel.xlsx).

    python -I scripts/ctrv_sbti.py

Un compromiso retirado o caducado ("Commitment Removed") es un candidato
natural a evento de "promesa incumplida". El fichero solo da la foto actual
(estado y fecha de ultima actualizacion), no el historial, asi que el
evento no se puede fechar con fiabilidad: se usa como descriptivo de
cobertura, no como resultado de H2.

Salida anonima: results/issue2_controversies/sbti_estado.csv
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ctrv_common import DATA, RESULTS, UA, load_terms  # noqa: E402

FILES = ["companies-excel.xlsx", "targets-excel.xlsx"]
BASE = "https://files.sciencebasedtargets.org/production/files/"


def main() -> None:
    raw = DATA / "sbti_raw"
    raw.mkdir(parents=True, exist_ok=True)
    for fn in FILES:
        if not (raw / fn).exists():
            req = urllib.request.Request(BASE + fn, headers={"User-Agent": UA})
            (raw / fn).write_bytes(urllib.request.urlopen(req, timeout=120).read())
    c = pd.read_excel(raw / "companies-excel.xlsx")
    t = pd.read_excel(raw / "targets-excel.xlsx")
    terms = load_terms()
    rows = []
    for r in terms.itertuples():
        m = c[c["company_name"].astype(str).str.contains(r.rx)]
        # matriz: prioriza la entidad con ISIN de la propia empresa o con el nombre mas corto
        m = m.assign(_k=m["isin"].isna().astype(int), _l=m["company_name"].astype(str).str.len())
        m = m.sort_values(["_k", "_l"]).head(1)
        if m.empty:
            rows.append({"firm_id": r.firm_id, "en_sbti": 0})
            continue
        x = m.iloc[0]
        tt = t[t["sbti_id"] == x["sbti_id"]]
        pub = pd.to_datetime(tt["date_published"], errors="coerce")
        rows.append({"firm_id": r.firm_id, "en_sbti": 1,
                     "tipo": x["organization_type"],
                     "near_term_status": x["near_term_status"],
                     "net_zero_status": x["net_zero_status"],
                     "status_reason": x["status_reason"],
                     "anio_actualizacion": pd.to_datetime(x["date_updated"], errors="coerce").year,
                     "primer_objetivo_publicado": pub.min().year if pub.notna().any() else None,
                     "n_objetivos": len(tt)})
    out = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS / "sbti_estado.csv", sep=";", index=False)
    print(out["near_term_status"].value_counts(dropna=False).to_string())
    print(out["net_zero_status"].value_counts(dropna=False).to_string())
    print(out["status_reason"].value_counts(dropna=False).to_string())
    print(out["primer_objetivo_publicado"].value_counts(dropna=False).sort_index().to_string())


if __name__ == "__main__":
    main()
