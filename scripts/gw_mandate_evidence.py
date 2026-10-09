"""Issue #2 groundwork: documentary evidence of reporting mandates in the reports.

For every report PDF of the panel (read-only, from the main clone's data/pdf),
counts regime markers in the extracted text: ESRS, NFRD-type statement,
EU Taxonomy eligibility / alignment, Green Asset Ratio, Disclosures Delegated
Act. The goal is to check, report by report, when Taxonomy alignment KPIs and
ESRS sustainability statements actually appear, as a complement to the legal
calendar in metadata/issue2_mandates.csv.

Output (anonymised, firm_id only):
    results/issue2_groundwork/mandate_evidence_by_report.csv

Usage:
    python scripts/gw_mandate_evidence.py --pdf-dir <main>/data/pdf \
        --firm-ids <main>/results/analysis_v3/firm_ids_internal.csv \
        --out results/issue2_groundwork
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import fitz  # PyMuPDF

PATTERNS = {
    "esrs": re.compile(r"\bESRS\b"),
    "esrs_full": re.compile(r"European Sustainability Reporting Standards?", re.I),
    "sust_statement": re.compile(
        r"sustainability statements?|Nachhaltigkeitserkl(?:ä|ae)rung|"
        r"(?:é|e)tat de durabilit(?:é|e)|rendicontazione (?:consolidata )?di sostenibilit",
        re.I),
    "nfrd_statement": re.compile(
        r"non-?financial (?:information )?statement|nichtfinanzielle(?:r)? (?:Konzern)?(?:erkl|bericht)|"
        r"d(?:é|e)claration de performance extra-?financi|\bDPEF\b|"
        r"dichiarazione (?:consolidata )?di carattere non finanziario|"
        r"estado (?:consolidado )?de informaci(?:ó|o)n no financiera|\bEINF\b|"
        r"2014/95/EU|Directive 2014/95",
        re.I),
    "csrd": re.compile(r"\bCSRD\b|2022/2464|Corporate Sustainability Reporting Directive", re.I),
    "tax_any": re.compile(r"\btaxonom", re.I),
    "tax_eligible": re.compile(r"taxonomy[- ]eligib|eligib\w* (?:under|for|according to) the (?:EU )?taxonomy", re.I),
    "tax_aligned": re.compile(r"taxonomy[- ]aligned|aligned (?:with|to) the (?:EU )?taxonomy|taxonomy[- ]alignment", re.I),
    "gar": re.compile(r"green asset ratio|\bGAR\b", re.I),
    "disclosures_da": re.compile(r"2021/2178"),
    "credit_institution": re.compile(r"credit institutions?", re.I),
    "insurance_kpi": re.compile(r"underwriting KPI|non-life (?:insurance|underwriting)[^.]{0,60}taxonom", re.I),
}

YEAR4 = re.compile(r"(?<!\d)20(1[89]|2[0-4])(?!\d)")
YEAR2 = re.compile(r"(?:ar|report)(1[89]|2[0-4])(?!\d)", re.I)


def year_of(name: str) -> int | None:
    m = YEAR4.search(name)
    if m:
        return 2000 + int(m.group(1))
    m = YEAR2.search(name)
    if m:
        return 2000 + int(m.group(1))
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", required=True)
    ap.add_argument("--firm-ids", required=True)
    ap.add_argument("--out", default="results/issue2_groundwork")
    a = ap.parse_args()

    ids = {}
    with open(a.firm_ids, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=";"):
            ids[row["firm"]] = row["firm_id"]

    rows = []
    for pdf in sorted(Path(a.pdf_dir).rglob("*")):
        if pdf.suffix.lower() != ".pdf" or "_repaired" in pdf.name:
            continue
        firm = pdf.parent.name
        country = pdf.parent.parent.name
        fid = ids.get(firm)
        yr = year_of(pdf.name)
        if fid is None or yr is None:
            print("SKIP (unmapped)", country, "/", "<firm>", "/", yr)
            continue
        try:
            doc = fitz.open(pdf)
            text = "\n".join(page.get_text() for page in doc)
            npages = doc.page_count
            doc.close()
        except Exception as e:  # noqa: BLE001
            print("ERROR", fid, yr, type(e).__name__)
            continue
        rec = {"firm_id": fid, "country_sample": country, "year": yr, "pages": npages}
        for k, p in PATTERNS.items():
            rec[k] = len(p.findall(text))
        rows.append(rec)
        print(fid, yr, rec["esrs"], rec["tax_aligned"], rec["gar"])

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda r: (r["firm_id"], r["year"]))
    with open(out / "mandate_evidence_by_report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("rows", len(rows))


if __name__ == "__main__":
    main()
