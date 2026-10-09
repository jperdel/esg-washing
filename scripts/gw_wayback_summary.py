"""Issue #2 groundwork: coverage of archived company-compiled consensus *documents*, per report year.

Reads the raw CDX listings saved by gw_wayback_cdx.py (ir_<firm_id>_*.txt) and, unlike the
capture-window measure of that script, dates each consensus PDF/XLS by the quarter-year tag in
its URL (a document archived years later still carries the forecasts of its own date).
Report year t is covered
    strict : a document for the Q1 or Q2 consensus of t+1 exists
    lenient: strict, or the Q4 consensus of t (collected before publication of report t)
Output (firm_id only): results/issue2_groundwork/wayback_ir_cobertura_docs.csv
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from gw_consensus_pilot import load_listing, tag_of

OFF_TOPIC = re.compile(r"management-unplugged|press|news|blog|career|job|scientific|consensus-statement", re.I)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--firm-ids", required=True)
    ap.add_argument("--out", default="results/issue2_groundwork")
    a = ap.parse_args()
    fids = sorted(r["firm_id"] for r in csv.DictReader(open(a.firm_ids, encoding="utf-8"), delimiter=";"))
    res = []
    for f in fids:
        rows = [r for r in load_listing(Path(a.raw_dir), f) if not OFF_TOPIC.search(r["url"])]
        tags = {tag_of(r["url"]) for r in rows if re.search(r"\.(pdf|xlsx?)($|\?)", r["url"], re.I)} - {None}
        rec = {"firm_id": f, "docs_con_trimestre": len(tags),
               "primer_ano": min((y for y, _ in tags), default=""), "ultimo_ano": max((y for y, _ in tags), default="")}
        for t in range(2018, 2025):
            strict = (t + 1, 1) in tags or (t + 1, 2) in tags
            rec[f"estricta_{t}"] = int(strict)
            rec[f"laxa_{t}"] = int(strict or (t, 4) in tags)
        rec["informes_estricta"] = sum(rec[f"estricta_{t}"] for t in range(2018, 2025))
        rec["informes_laxa"] = sum(rec[f"laxa_{t}"] for t in range(2018, 2025))
        res.append(rec)
    with open(Path(a.out) / "wayback_ir_cobertura_docs.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res[0].keys()))
        w.writeheader()
        w.writerows(res)
    print("empresas con docs:", sum(r["docs_con_trimestre"] > 0 for r in res),
          "| informes estricta:", sum(r["informes_estricta"] for r in res),
          "| laxa:", sum(r["informes_laxa"] for r in res))


if __name__ == "__main__":
    main()
