"""Issue #2 groundwork: coverage of archived company-compiled consensus *documents*, per report year.

Reads the raw CDX listings saved by gw_wayback_cdx.py (ir_<firm_id>_*.txt). Unlike the capture-window
measure of that script, each consensus PDF/XLS is dated by the quarter tag or the date in its file
name (a document archived years later still carries the forecasts of its own date); only undated
names fall back to the capture date (see gw_consensus_pilot.doc_date).
Report year t is covered
    strict : a document collected Mar-Jul of t+1 (pre-Q1 / pre-Q2 consensus, after publication)
    lenient: a document collected Jan-Jul of t+1
Output (firm_id only): results/issue2_groundwork/wayback_ir_cobertura_docs.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # own scripts dir (python -I drops it)
from gw_consensus_pilot import OFF_TOPIC, doc_date, in_window, load_listing  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--firm-ids", required=True)
    ap.add_argument("--out", default="results/issue2_groundwork")
    a = ap.parse_args()
    fids = sorted(r["firm_id"] for r in csv.DictReader(open(a.firm_ids, encoding="utf-8"), delimiter=";"))
    res = []
    for f in fids:
        rows = [r for r in load_listing(Path(a.raw_dir), f)
                if not OFF_TOPIC.search(r["url"]) and re.search(r"\.(pdf|xlsx?)($|\?)", r["url"], re.I)]
        dated = [(doc_date(r["url"], r["ts"]), r["url"].split("?")[0].lower()) for r in rows]
        dated = [(d, u) for d, u in dated if d]
        by_name = {u for d, u in dated if d[2] != "captura"}
        rec = {"firm_id": f, "ficheros": len({u for _, u in dated}), "ficheros_fechados_por_nombre": len(by_name),
               "primer_ano": min((d[0] for d, _ in dated), default=""),
               "ultimo_ano": max((d[0] for d, _ in dated), default="")}
        for t in range(2018, 2025):
            st = [d for d, _ in dated if in_window(d, t, True)]
            lx = [d for d, _ in dated if in_window(d, t, False)]
            rec[f"estricta_{t}"] = int(bool(st))
            rec[f"laxa_{t}"] = int(bool(lx))
            rec[f"solo_captura_{t}"] = int(bool(lx) and all(d[2] == "captura" for d in lx))
        rec["informes_estricta"] = sum(rec[f"estricta_{t}"] for t in range(2018, 2025))
        rec["informes_laxa"] = sum(rec[f"laxa_{t}"] for t in range(2018, 2025))
        res.append(rec)
    with open(Path(a.out) / "wayback_ir_cobertura_docs.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res[0].keys()))
        w.writeheader()
        w.writerows(res)
    print("empresas con ficheros:", sum(r["ficheros"] > 0 for r in res),
          "| con algún informe cubierto (laxa):", sum(r["informes_laxa"] > 0 for r in res),
          "| informes estricta:", sum(r["informes_estricta"] for r in res),
          "| laxa:", sum(r["informes_laxa"] for r in res))


if __name__ == "__main__":
    main()
