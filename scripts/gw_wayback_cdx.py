"""Issue #2 groundwork: is there free *historical* analyst-consensus data in the Wayback Machine?

Queries the Internet Archive CDX API (read-only, ~1 request every 3 s) for:
  1. company IR sites: archived URLs whose path mentions consensus / Konsens / consenso
     (company-compiled analyst consensus pages, PDFs and spreadsheets);
  2. Yahoo Finance "analysis" pages (EPS estimate: no. of analysts, low/avg/high);
  3. MarketScreener consensus pages (pilot firms only; needs the site slug).

Inputs (internal, gitignored): metadata/empresas_dominios_internal.csv, metadata/empresas_tickers.csv,
<main>/results/analysis_v3/firm_ids_internal.csv.
Raw CDX listings (contain URLs, hence firm names) go to --raw-dir outside the repo.
Committed outputs (firm_id only) in results/issue2_groundwork/:
  wayback_ir_consensus.csv, wayback_yahoo_analysis.csv, wayback_marketscreener_piloto.csv

Coverage definition: the report of fiscal year t is "covered" by a source when at least one
capture exists in the window Feb-Jun of t+1 (after publication of the report, when H1 measures
dispersion of the t+1 EPS forecast).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

CDX = "http://web.archive.org/cdx/search/cdx"
UA = {"User-Agent": "esg-research-groundwork/0.1 (academic; low rate)"}
YEARS = list(range(2018, 2026))
IR_RE = r"(?i).*(consensus|konsens|consenso).*"
IR_CONTEXT = re.compile(r"invest|/ir[/-]|finanz|share|aktie|action|blob|download|\.pdf|\.xls|media|document", re.I)
IR_PATHS = ["investor", "investors", "ir", "en/investor", "en/investors", "en/ir", "global/en/investor",
            "global/en/company/investor", "de/investor", "investor-relations", "en/investor-relations",
            "finance", "en/finance", "shareholders", "en/shareholders", "es/accionistas", "fr/finance",
            "it/investitori", "en/investors-relations"]
OFF_TOPIC =re.compile(r"management-unplugged|press|news|blog|career|job|scientific|consensus-statement", re.I)


def cdx(params: dict, raw: Path, tag: str, pause: float):
    url = CDX + "?" + urllib.parse.urlencode(params, doseq=True)
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=180) as r:
                txt = r.read().decode("utf-8", errors="replace")
        except Exception as e:  # noqa: BLE001
            code = getattr(e, "code", None)
            print("  error", tag, type(e).__name__, code)
            if code == 504:  # server-side 60 s scan limit: deterministic, do not hammer
                time.sleep(pause)
                return None
            time.sleep(30 * (attempt + 1))
            continue
        if "<html" in txt[:2000].lower():  # "Temporarily Offline" page: back off and retry
            print("  offline", tag)
            time.sleep(120 * (attempt + 1))
            continue
        break
    else:
        return None
    (raw / f"{tag}.txt").write_text(txt, encoding="utf-8")
    time.sleep(pause)
    rows = []
    for line in txt.splitlines():
        p = line.split(" ")
        if len(p) >= 4 and re.fullmatch(r"\d{14}", p[1]):
            rows.append({"original": p[0], "timestamp": p[1], "mimetype": p[2], "statuscode": p[3]})
    return rows


def window_cov(stamps: list[str]) -> dict:
    """report year t covered if a capture exists in Feb-Jun of t+1."""
    months = {(int(s[:4]), int(s[4:6])) for s in stamps}
    return {t: int(any((t + 1, m) in months for m in range(2, 7))) for t in range(2018, 2025)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--firm-ids", required=True)
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--meta", default="metadata")
    ap.add_argument("--out", default="results/issue2_groundwork")
    ap.add_argument("--pause", type=float, default=3.0)
    ap.add_argument("--skip-ir", action="store_true")
    ap.add_argument("--skip-yahoo", action="store_true")
    a = ap.parse_args()
    raw, out, meta = Path(a.raw_dir), Path(a.out), Path(a.meta)
    raw.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    fid = {r["firm"]: r["firm_id"] for r in csv.DictReader(open(a.firm_ids, encoding="utf-8"), delimiter=";")}
    ytk = {r["Empresa"]: r["ticker_yahoo"] for r in csv.DictReader(open(meta / "empresas_tickers.csv", encoding="utf-8"), delimiter=";")}
    dom = list(csv.DictReader(open(meta / "empresas_dominios_internal.csv", encoding="utf-8"), delimiter=";"))

    # 1. IR consensus documents
    if not a.skip_ir:
        res = []
        for d in dom:
            f = fid[d["firm"]]
            caps, estado = [], "dominio"
            for k, domain in enumerate(x for x in d["dominios_ir"].split("|") if x):
                q = {"filter": ["original:" + IR_RE], "fl": "original,timestamp,mimetype,statuscode",
                     "collapse": "timestamp:8", "limit": 20000}
                rows = cdx({"url": domain, "matchType": "domain", **q}, raw, f"ir_{f}_{k}", a.pause)
                if rows is None:
                    # domain too large for one CDX scan: fall back to usual IR path roots
                    estado = "rutas_ir"
                    rows = []
                    for j, p in enumerate(IR_PATHS):
                        r2 = cdx({"url": f"{domain}/{p}", "matchType": "prefix", **q}, raw,
                                 f"ir_{f}_{k}_p{j}", a.pause)
                        if r2 is None:
                            estado = "rutas_ir_parcial"
                        rows += r2 or []
                caps += [r for r in rows if IR_CONTEXT.search(r["original"]) and not OFF_TOPIC.search(r["original"])]
            ok = [r for r in caps if r["statuscode"] in ("200", "-")]
            files = [r for r in ok if re.search(r"\.(pdf|xlsx?)($|\?)", r["original"], re.I)]
            urls = {r["original"].split("?")[0].lower() for r in ok}
            by_year = defaultdict(set)
            for r in ok:
                by_year[int(r["timestamp"][:4])].add(r["original"].split("?")[0].lower())
            cov = window_cov([r["timestamp"] for r in ok])
            rec = {"firm_id": f, "consulta": estado, "urls_consenso": len(urls), "capturas": len(ok), "capturas_pdf_xls": len(files)}
            rec.update({f"urls_{y}": len(by_year.get(y, ())) for y in YEARS})
            rec.update({f"cubre_{t}": cov[t] for t in range(2018, 2025)})
            rec["informes_cubiertos"] = sum(cov.values())
            res.append(rec)
            print("IR", f, rec["urls_consenso"], rec["informes_cubiertos"])
        res.sort(key=lambda r: r["firm_id"])
        with open(out / "wayback_ir_consensus.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(res[0].keys()))
            w.writeheader()
            w.writerows(res)

    # 2. Yahoo analysis pages
    if not a.skip_yahoo:
        res = []
        for firm, t in ytk.items():
            f = fid[firm]
            stamps = []
            for host in ("finance.yahoo.com", "uk.finance.yahoo.com"):
                rows = cdx({"url": f"{host}/quote/{t}/analysis", "matchType": "prefix",
                            "fl": "original,timestamp,mimetype,statuscode", "collapse": "timestamp:8"},
                           raw, f"yahoo_{f}_{host.split('.')[0]}", a.pause)
                stamps += [r["timestamp"] for r in (rows or []) if r["statuscode"] == "200"]
            cov = window_cov(stamps)
            rec = {"firm_id": f, "capturas_200": len(stamps)}
            rec.update({f"capturas_{y}": sum(s[:4] == str(y) for s in stamps) for y in YEARS})
            rec.update({f"cubre_{t2}": cov[t2] for t2 in range(2018, 2025)})
            rec["informes_cubiertos"] = sum(cov.values())
            res.append(rec)
            print("YH", f, len(stamps), rec["informes_cubiertos"])
        res.sort(key=lambda r: r["firm_id"])
        with open(out / "wayback_yahoo_analysis.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(res[0].keys()))
            w.writeheader()
            w.writerows(res)

    # 3. MarketScreener pilot
    res = []
    for d in dom:
        if not d["ms_prefijo"]:
            continue
        f = fid[d["firm"]]
        stamps = []
        for k, host in enumerate(("www.marketscreener.com/quote/stock/", "www.4-traders.com/")):
            rows = cdx({"url": host + d["ms_prefijo"], "matchType": "prefix", "filter": "original:(?i).*consensus.*",
                        "fl": "original,timestamp,mimetype,statuscode", "collapse": "timestamp:8"},
                       raw, f"ms_{f}_{k}", a.pause)
            stamps += [r["timestamp"] for r in (rows or []) if r["statuscode"] == "200"]
        cov = window_cov(stamps)
        rec = {"firm_id": f, "capturas_200": len(stamps)}
        rec.update({f"capturas_{y}": sum(s[:4] == str(y) for s in stamps) for y in YEARS})
        rec.update({f"cubre_{t2}": cov[t2] for t2 in range(2018, 2025)})
        rec["informes_cubiertos"] = sum(cov.values())
        res.append(rec)
        print("MS", f, len(stamps), rec["informes_cubiertos"])
    if res:
        with open(out / "wayback_marketscreener_piloto.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(res[0].keys()))
            w.writeheader()
            w.writerows(res)


if __name__ == "__main__":
    main()
