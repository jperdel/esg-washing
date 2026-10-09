"""Issue #2 groundwork: firm x year reporting-mandate panel (NFRD, EU Taxonomy Art. 8, CSRD/ESRS).

Legal calendar (see results/issue2_groundwork/README.md and metadata/issue2_mandates_fuentes.csv):
  * NFRD (Dir. 2014/95/EU): financial years starting on/after 1-1-2017 -> every report 2018-2024,
    until replaced by the CSRD where the CSRD was transposed and applies.
  * Taxonomy Art. 8, Disclosures DA (Reg. (EU) 2021/2178) Art. 10: eligibility-only disclosures
    published from 1-1-2022; full KPIs from 1-1-2023 (non-financial undertakings) and from 1-1-2024
    (financial undertakings: credit institutions, insurers, asset managers, investment firms).
    The trigger is the publication date of the report, so non-calendar fiscal years shift.
  * CSRD (Dir. (EU) 2022/2464) Art. 5(2)(a): wave 1 (large PIEs > 500 employees) for financial years
    starting on/after 1-1-2024, *only if the Member State of the registered office had transposed it*
    (FR, IT, FI, BE yes; DE, ES, NL no). The stop-the-clock Dir. (EU) 2025/794 only moved waves 2-3.

Firm attributes keyed by anonymous firm_id (no names): fiscal-year convention and Art. 8 template
(checked in each firm's own reports; see mandate_evidence_by_report.csv). Legal seat comes from GLEIF
(metadata/empresas_identificadores_internal.csv, gitignored).

Output: metadata/issue2_mandates.csv (firm_id x year).
"""
from __future__ import annotations

import argparse
import csv
from datetime import date
from pathlib import Path

# Art. 8 template actually used / legally applicable (checked in the reports).
CREDIT_INST = {"Financials-02", "Financials-03", "Financials-05", "Financials-06", "Financials-07",
               "Financials-08", "Financials-10", "Industrials-06"}
INSURERS = {"Financials-04", "Financials-09", "Financials-11"}

# Non-calendar fiscal years: label t -> (FY start, approx. publication date of the report).
# Default (calendar FY): start 1-1-t, published Feb-Apr of t+1.
def fy_dates(fid: str, t: int) -> tuple[date, date]:
    if fid in ("Industrials-04", "Technology-01"):      # FY Oct(t-1)-Sep(t), report Nov-Dec t
        return date(t - 1, 10, 1), date(t, 12, 1)
    if fid == "Consumer Staples-04":                     # FY Jul-Jun, report Sep
        if t == 2024:                                    # corpus label 2024 = FY ended Jun 2025
            return date(2024, 7, 1), date(2025, 9, 1)
        return date(t - 1, 7, 1), date(t, 9, 1)          # labels 2018-2023 = FY ended Jun t
    if fid == "Consumer Discretionary-07":               # FY Feb(t)-Jan(t+1), report Mar t+1
        return date(t, 2, 1), date(t + 1, 3, 15)
    return date(t, 1, 1), date(t + 1, 3, 1)


CSRD_TRANSPOSED = {  # Member State of registered office -> date the wave-1 regime became law
    "FR": date(2023, 12, 7), "IT": date(2024, 9, 25), "FI": date(2024, 1, 1), "BE": date(2024, 12, 30),
}
SRC_NFRD = {"DE": "NFRD-DE", "FR": "NFRD-FR", "IT": "NFRD-IT", "ES": "NFRD-ES", "NL": "NFRD-NL",
            "BE": "NFRD-BE", "FI": "NFRD-FI"}

NOTES = {
    ("Consumer Discretionary-09", 2018): "informe del predecesor (SA francesa): NFRD vía ley francesa (DPEF)",
    ("Consumer Discretionary-09", 2019): "informe del predecesor (SA francesa): NFRD vía ley francesa (DPEF)",
    ("Financials-07", 2018): "redomiciliada en FI el 1-10-2018; ejercicio 2018 ya bajo ley finlandesa",
    ("Industrials-06", 2018): "EIP desde 2017 (licencia bancaria); ~770 ETC medios en 2018 (> 500); el informe no rotula el estado no financiero",
    ("Consumer Staples-04", 2024): "etiqueta 2024 = ejercicio cerrado en jun-2025 (salta el cerrado en jun-2024): revisar corpus",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--firm-ids", required=True)
    ap.add_argument("--meta", default="metadata")
    a = ap.parse_args()
    meta = Path(a.meta)

    fid = {r["firm"]: r["firm_id"] for r in csv.DictReader(open(a.firm_ids, encoding="utf-8"), delimiter=";")}
    pais = {fid[r["Empresa"]]: r["País"] for r in
            csv.DictReader(open(meta / "empresas_supersector.csv", encoding="utf-8"), delimiter=";")}
    sede = {r["firm_id"]: r["sede_legal"] for r in
            csv.DictReader(open(meta / "empresas_identificadores_internal.csv", encoding="utf-8"), delimiter=";")}

    rows = []
    for f in sorted(fid.values()):
        s = sede[f]
        tmpl = "entidad_credito" if f in CREDIT_INST else "aseguradora" if f in INSURERS else "no_financiera"
        for t in range(2018, 2025):
            start, pub = fy_dates(f, t)
            seat = "FR" if (f == "Consumer Discretionary-09" and t <= 2019) else s
            elig = int(pub >= date(2022, 1, 1))
            align_from = date(2024, 1, 1) if tmpl != "no_financiera" else date(2023, 1, 1)
            align = int(pub >= align_from)
            csrd = int(start >= date(2024, 1, 1) and seat in CSRD_TRANSPOSED and CSRD_TRANSPOSED[seat] <= pub)
            nfrd = int(start >= date(2017, 1, 1) and not csrd)
            src = ["NFRD-DIR", SRC_NFRD[seat]] if nfrd else []
            if elig:
                src += ["TAX-REG", "TAX-DA"]
            if t == 2024 or csrd:
                src += ["CSRD-DIR", f"CSRD-{seat}", "STC-DIR"]
            rows.append({
                "firm_id": f, "year": t, "pais_muestra": pais[f], "sede_legal": seat,
                "plantilla_art8": tmpl, "inicio_ejercicio": start.isoformat(),
                "publicacion_aprox": pub.isoformat(),
                "nfrd": nfrd, "taxonomy_eligibility": elig, "taxonomy_alignment": align,
                "csrd": csrd, "regimen": "CSRD-ESRS" if csrd else ("NFRD" if nfrd else ""),
                "nota": NOTES.get((f, t), ""), "sources": ";".join(dict.fromkeys(src)),
            })
    with open(meta / "issue2_mandates.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    n24 = [r for r in rows if r["year"] == 2024]
    print("rows", len(rows), "| csrd 2024:", sum(r["csrd"] for r in n24),
          "| align 2022:", sum(r["taxonomy_alignment"] for r in rows if r["year"] == 2022),
          "| align 2023:", sum(r["taxonomy_alignment"] for r in rows if r["year"] == 2023))


if __name__ == "__main__":
    main()
