"""Issue #2 groundwork: verify ISIN and LEI identifiers for the 49 firms.

Input (internal, gitignored, contains firm names):
    metadata/empresas_isin_internal.csv   firm;isin_candidate;sede_legal;isin_predecesor;nota
    metadata/empresas_tickers.csv          Yahoo tickers (for cross-checking)
    <main>/results/analysis_v3/firm_ids_internal.csv   firm -> firm_id

Free sources, low rate:
    GLEIF API   https://api.gleif.org/api/v1/lei-records?filter[isin]=...   (ISIN -> LEI)
    OpenFIGI    https://api.openfigi.com/v3/mapping  (no key: <=25 req/min, <=10 jobs/req)

Outputs:
    metadata/empresas_identificadores_internal.csv   (gitignored; names + ISIN + LEI + FIGI)
    metadata/issue2_identificadores_anon.csv          (committed; firm_id + verification flags only)

Raw API responses are stored under --raw-dir (outside the repo) and parsed as JSON only.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

UA = {"User-Agent": "esg-research-groundwork/0.1 (academic; low rate)"}


def http_json(url: str, data: bytes | None = None, headers: dict | None = None):
    h = dict(UA)
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, data=data, headers=h)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def gleif_by_isin(isin: str, raw: Path):
    url = "https://api.gleif.org/api/v1/lei-records?" + urllib.parse.urlencode({"filter[isin]": isin})
    js = http_json(url, headers={"Accept": "application/vnd.api+json"})
    (raw / f"gleif_{isin}.json").write_text(json.dumps(js), encoding="utf-8")
    time.sleep(1.2)
    d = js.get("data") or []
    if not d:
        return None
    a = d[0]["attributes"]
    return {
        "lei": d[0]["id"],
        "lei_legal_name": a["entity"]["legalName"]["name"],
        "lei_country": a["entity"]["legalAddress"]["country"],
        "lei_jurisdiction": a["entity"].get("jurisdiction") or "",
        "lei_entity_status": a["entity"].get("status") or "",
        "lei_reg_status": a["registration"].get("status") or "",
    }


def gleif_by_name(name: str, country: str, raw: Path, tag: str):
    """Fallback when the ISIN is not in GLEIF's ISIN map: exact legal-name filter,
    keeping the record in the expected country and skipping DUPLICATE registrations."""
    url = "https://api.gleif.org/api/v1/lei-records?" + urllib.parse.urlencode(
        {"filter[entity.legalName]": name, "page[size]": 10})
    js = http_json(url, headers={"Accept": "application/vnd.api+json"})
    (raw / f"gleif_name_{tag}.json").write_text(json.dumps(js), encoding="utf-8")
    time.sleep(1.2)
    for d in js.get("data") or []:
        a = d["attributes"]
        if a["entity"]["legalAddress"]["country"] != country or a["registration"]["status"] == "DUPLICATE":
            continue
        return {
            "lei": d["id"],
            "lei_legal_name": a["entity"]["legalName"]["name"],
            "lei_country": a["entity"]["legalAddress"]["country"],
            "lei_jurisdiction": a["entity"].get("jurisdiction") or "",
            "lei_entity_status": a["entity"].get("status") or "",
            "lei_reg_status": a["registration"].get("status") or "",
            "lei_via": "nombre",
        }
    return None


def openfigi(isins: list[str], raw: Path):
    out = {}
    for i in range(0, len(isins), 10):
        chunk = isins[i:i + 10]
        body = json.dumps([{"idType": "ID_ISIN", "idValue": x} for x in chunk]).encode()
        js = http_json("https://api.openfigi.com/v3/mapping", data=body,
                       headers={"Content-Type": "application/json"})
        (raw / f"openfigi_{i:03d}.json").write_text(json.dumps(js), encoding="utf-8")
        for isin, res in zip(chunk, js):
            out[isin] = res.get("data") or []
        time.sleep(3.0)  # well under 25 req/min
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--firm-ids", required=True)
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--meta", default="metadata")
    a = ap.parse_args()
    meta = Path(a.meta)
    raw = Path(a.raw_dir)
    raw.mkdir(parents=True, exist_ok=True)

    with open(a.firm_ids, encoding="utf-8") as f:
        fid = {r["firm"]: r["firm_id"] for r in csv.DictReader(f, delimiter=";")}
    with open(meta / "empresas_tickers.csv", encoding="utf-8") as f:
        ytk = {r["Empresa"]: r["ticker_yahoo"] for r in csv.DictReader(f, delimiter=";")}
    with open(meta / "empresas_isin_internal.csv", encoding="utf-8") as f:
        cand = list(csv.DictReader(f, delimiter=";"))

    all_isins = [c["isin_candidate"] for c in cand] + [c["isin_predecesor"] for c in cand if c["isin_predecesor"]]
    figi = openfigi(all_isins, raw)

    internal, anon = [], []
    for c in cand:
        firm, isin = c["firm"], c["isin_candidate"]
        g = gleif_by_isin(isin, raw)
        if g:
            g["lei_via"] = "isin"
        elif c.get("lei_nombre_gleif"):
            g = gleif_by_name(c["lei_nombre_gleif"], c["sede_legal"], raw, isin)
        g = g or {}
        fig = figi.get(isin, [])
        tickers = sorted({x.get("ticker", "") for x in fig})
        sec_types = sorted({x.get("securityType", "") for x in fig})
        share_cls = sorted({x.get("shareClassFIGI") or "" for x in fig} - {""})
        comp = sorted({x.get("compositeFIGI") or "" for x in fig} - {""})
        root = ytk.get(firm, "").split(".")[0].replace("-FI", "").upper()
        ticker_match = any(t.upper().replace("/", "") == root or t.upper() == root for t in tickers)
        pred = c["isin_predecesor"]
        gp = gleif_by_isin(pred, raw) if pred else {}
        if pred and not gp and c.get("lei_predecesor_nombre_gleif"):
            gp = gleif_by_name(c["lei_predecesor_nombre_gleif"], pred[:2], raw, pred)
        rec = {
            "firm": firm, "firm_id": fid.get(firm, ""), "sede_legal": c["sede_legal"],
            "ticker_yahoo": ytk.get(firm, ""), "isin": isin, **{k: g.get(k, "") for k in
            ["lei", "lei_legal_name", "lei_country", "lei_jurisdiction", "lei_entity_status", "lei_reg_status",
             "lei_via"]},
            "figi_share_class": "|".join(share_cls), "figi_composite_n": len(comp),
            "figi_security_types": "|".join(sec_types), "figi_tickers": "|".join(tickers),
            "ticker_match": int(ticker_match),
            "isin_predecesor": pred, "lei_predecesor": (gp or {}).get("lei", ""),
            "lei_predecesor_name": (gp or {}).get("lei_legal_name", ""),
            "lei_predecesor_estado": (gp or {}).get("lei_reg_status", ""),
            "nota": c["nota"],
        }
        internal.append(rec)
        anon.append({
            "firm_id": rec["firm_id"], "sede_legal": c["sede_legal"], "isin_pais": isin[:2],
            "isin_en_openfigi": int(bool(fig)),
            "tipo_valor": "|".join(sec_types),
            "ticker_coincide_yahoo": int(ticker_match),
            "lei_encontrado": int(bool(g)),
            "lei_via": g.get("lei_via", ""),
            "lei_pais_coincide_sede": int(g.get("lei_country", "") == c["sede_legal"]) if g else "",
            "lei_estado": g.get("lei_reg_status", ""),
            "tiene_predecesor": int(bool(pred)),
            "lei_predecesor_encontrado": int(bool(gp)) if pred else "",
        })
        print(rec["firm_id"], isin[:2], bool(g), g.get("lei_reg_status", ""), sec_types, int(ticker_match))

    with open(meta / "empresas_identificadores_internal.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(internal[0].keys()), delimiter=";")
        w.writeheader()
        w.writerows(internal)
    anon.sort(key=lambda r: r["firm_id"])
    with open(meta / "issue2_identificadores_anon.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(anon[0].keys()), delimiter=";")
        w.writeheader()
        w.writerows(anon)


if __name__ == "__main__":
    main()
