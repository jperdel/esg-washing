"""
unback_pilot.py
---------------
Piloto del issue #7 (historico: la version adoptada esta en src/unback.py y
main.py): UNBACK, compromisos no respaldados, como sustituto de
HEDGE. Medida de reglas, sin modelos de lenguaje.

Unidad: la frase, dentro de los parrafos del texto extraido (la misma lectura
que cb_common.paragraphs). Una frase es de COMPROMISO si nombra a la empresa
(we/our/the Group/the Company...), contiene una senal de compromiso o
aspiracion (commit to, our commitment, aim to, our target/goal/ambition,
pledge, plan/intend/seek/strive to, on track to, net zero, carbon neutral,
will + verbo de accion) y al menos una entrada del vocabulario ESG.
Se excluyen los condicionales (if, could, might, may que no es el mes), las
remisiones internas y los avisos sobre forward-looking statements.

Respaldo, en la frase o en la anterior o siguiente del mismo parrafo
(Bingler et al. 2024 etiquetan parrafos de unas 3 frases):
    Q  cantidad: los patrones de QUANT (porcentajes no nulos, magnitudes,
       unidades con cifra) mas dinero y halve/double;
    D  plazo: by/until/before/no later than/through + un ano posterior al del informe, o
       "<ano> target/goal";
    B  ano base: compared to/vs/from/relative to + un ano no posterior al del
       informe, baseline, base year.
Fuentes de Q, D y B: ESRS 2 MDR-T par. 80(b), (d), (e); ESRS E1-4 par. 34;
SBTi; CDP; TPI.

Por informe:
    UNBACK      media de (3 - Q - D - B) / 3 sobre las frases de compromiso
    UNBACK_bin  proporcion de frases de compromiso sin ningun respaldo
    UNBACK_DB   media de (2 - D - B) / 2 (sin cantidad: ortogonal a QUANT)
    *_clim      lo mismo solo con las frases de compromiso climaticas
                (palabra clave de clima en la frase), para comparar con el CTI

Salidas:
    data/unback/sentences_internal.csv.gz   frases de compromiso con sus
                                             marcas (cache sin versionar,
                                             lleva texto)
    results/unback/doc_level_unback.csv      por informe, solo firm_label
    results/unback/summary.json              correlaciones y tendencias

    <python del pipeline> scripts/unback_pilot.py
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import config  # noqa: E402
import lexical_document_filter as ldf  # noqa: E402
from cb_common import BASE_DIR, MAIN_DIR, iter_docs, load_scores, paragraphs  # noqa: E402
from esgsi_analyzer import MAY_MONTH_RE, _is_magnitude  # noqa: E402

CACHE = BASE_DIR / "data" / "unback"
OUT = BASE_DIR / "results" / "unback"
SEED = 20261014
B_BOOT = 2000
MIN_COMMIT = 5

I = re.IGNORECASE
SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(])")
COMPANY_RE = re.compile(r"\b(?:we|our|us|the group|the company|the bank|the firm)\b", I)
ACTION = (r"reduce|cut|lower|decrease|achieve|reach|become|invest|phase\s+out|eliminate|increase|double|"
          r"halve|source|switch|transition|deliver|ensure|implement|launch|expand|align|electrify|install|"
          r"cover|train|hire|recruit|replace|convert|stop|end|remove|offset|compensate|neutrali[sz]e")
COMMIT_RE = re.compile(
    r"\b(?:commit(?:s|ted)?\s+to|commitments?\s+to\b|(?:our|its|the\s+group'?s?)\s+(?:[\w-]+\s+){0,2}?commitments?\b|"
    r"aim(?:s|ing)?\s+(?:to|for)\b|"
    r"(?:our|its|the|this|these|new|an?)\s+(?:[\w-]+\s+){0,3}?(?:aims?|goals?|targets?|ambitions?|objectives?)\b|"
    r"target(?:s|ed|ing)?\s+(?:to|of|is|a|an)\b|"
    r"pledge[sd]?\b|"
    r"(?:plans?|planned|intends?|intended|seeks?|strives?|endeavou?rs?)\s+to\b|"
    r"on\s+track\s+to\b|"
    r"net[- ]zero\b|(?:carbon|climate)[- ]neutral(?:ity)?\b|"
    r"will\s+(?:continue\s+to\s+|also\s+|further\s+)?(?:" + ACTION + r")\w*\b)", I)
EXCLUDE_RE = re.compile(
    r"forward[- ]looking\s+statements?|\bif\b|\bcould\b|\bmight\b|"
    r"will\s+be\s+(?:discussed|presented|described|detailed|explained|published|reported|disclosed|submitted)|"
    r"\bsee\s+(?:section|page|chapter|note)|"
    # Contextos financieros y de gobierno que llevan palabras ESG sin ser un
    # compromiso ESG: retribucion, auditoria, apetito de riesgo, valoracion.
    r"remuneration|\bbonus|variable\s+(?:pay|compensation)|target\s+achievement|"
    r"assurance\s+(?:engagement|conclusion)|risk\s+appetite|market\s+share|cash\s+flows?|"
    r"terminal\s+value|dividends?\b|earnings\s+per\s+share|\bebit", I)
CLIMATE_RE = re.compile(r"climate|carbon|\bco2|\bghg\b|greenhouse|emission|net[- ]zero|decarboni|"
                        r"scope\s?[123]\b|1\.5\s?°?\s?c\b|fossil|renewable", I)

QPAT = {k: re.compile(v, I) for k, v in config.QUANT_PATTERNS.items()}
MONEY_RE = re.compile(r"(?:€|\$|£|\beur\b|\busd\b)\s?\d|\d[\d,.]*\s?(?:million|billion|bn|mn)\b", I)
HALVE_RE = re.compile(r"\b(?:halve|halving|by\s+half|double|doubling|triple|tripling)\b", I)
YEAR = r"((?:19|20)\d{2})"
DEADLINE_RE = re.compile(r"\b(?:by|until|till|before|no\s+later\s+than|through|towards?)\s+"
                         r"(?:the\s+end\s+of\s+|end[- ]|year[- ]end\s+)?(?:fy\s?|financial\s+year\s+)?" + YEAR + r"\b", I)
YEAR_TARGET_RE = re.compile(r"\b" + YEAR + r"\s+(?:targets?|goals?|ambitions?|objectives?|horizon|roadmap)\b", I)
BASE_RE = re.compile(r"(?:compared\s+(?:to|with)|\bvs\.?|\bversus|relative\s+to|\bfrom|\bagainst|"
                     r"\bbelow|\bover)\s+(?:the\s+)?(?:fy\s?|financial\s+year\s+)?" + YEAR + r"\b", I)
BASEWORD_RE = re.compile(r"\bbase(?:line)?[- ]?years?\b|\bbaseline\b", I)


def sentences(par: str) -> list[str]:
    return [s.strip() for s in SENT_RE.split(par) if s.strip()]


def is_commitment(s: str) -> bool:
    # Compromiso ESG: empresa + senal de compromiso + al menos una entrada del
    # vocabulario ESG (el mismo detector de la extraccion), lo que deja fuera
    # los objetivos financieros (dividendos, costes, retribucion).
    if not COMPANY_RE.search(s) or not COMMIT_RE.search(s) or not ldf._ESG_KW_RE.search(s):
        return False
    t = MAY_MONTH_RE.sub(" ", s)
    return not (EXCLUDE_RE.search(t) or re.search(r"\bmay\b", t, I))


def has_q(w: str) -> bool:
    for k, p in QPAT.items():
        for m in p.finditer(w):
            if k != "large_numbers" or _is_magnitude(w, m):
                return True
    return bool(MONEY_RE.search(w) or HALVE_RE.search(w))


def has_d(w: str, year: int) -> bool:
    return any(int(m.group(1)) > year for r in (DEADLINE_RE, YEAR_TARGET_RE) for m in r.finditer(w))


def has_b(w: str, year: int) -> bool:
    return bool(BASEWORD_RE.search(w)) or any(int(m.group(1)) <= year for m in BASE_RE.finditer(w))


def scan(scores: pd.DataFrame) -> pd.DataFrame:
    year_of = dict(zip(scores["doc_key"], scores["year"].astype(int)))
    rows = []
    for dk, zones in iter_docs(scores):
        y = year_of[dk]
        n_sent = 0
        for zone in zones:
            for par in paragraphs(zone):
                ss = sentences(par)
                n_sent += len(ss)
                for i, s in enumerate(ss):
                    if not is_commitment(s):
                        continue
                    w = " ".join(ss[max(0, i - 1): i + 2])
                    rows.append({"doc_key": dk, "text": s, "window": w,
                                 "Q": has_q(w), "D": has_d(w, y), "B": has_b(w, y),
                                 "Q0": has_q(s), "D0": has_d(s, y), "B0": has_b(s, y),
                                 "clim": bool(CLIMATE_RE.search(s))})
        rows.append({"doc_key": dk, "text": None, "n_sent": n_sent})
    return pd.DataFrame(rows)


def aggregate(S: pd.DataFrame) -> pd.DataFrame:
    nsent = S[S["text"].isna()].set_index("doc_key")["n_sent"]
    C = S[S["text"].notna()].copy()
    for suf, (q, d, b) in {"": ("Q", "D", "B"), "_w0": ("Q0", "D0", "B0")}.items():
        back = C[q].astype(int) + C[d].astype(int) + C[b].astype(int)
        C["u" + suf] = (3 - back) / 3
        C["ub" + suf] = (back == 0).astype(float)
        C["udb" + suf] = (2 - C[d].astype(int) - C[b].astype(int)) / 2

    def agg(g):
        return pd.Series({"n_commit": len(g), "UNBACK": g["u"].mean(), "UNBACK_bin": g["ub"].mean(),
                          "UNBACK_DB": g["udb"].mean(), "UNBACK_w0": g["u_w0"].mean(),
                          "UNBACK_bin_w0": g["ub_w0"].mean(), "sh_Q": g["Q"].mean(), "sh_D": g["D"].mean(),
                          "sh_B": g["B"].mean()})

    A = C.groupby("doc_key").apply(agg, include_groups=False)
    Ac = C[C["clim"]].groupby("doc_key").apply(agg, include_groups=False).add_suffix("_clim")
    out = pd.DataFrame({"n_sent": nsent}).join(A).join(Ac)
    out["commit_rate"] = 100 * out["n_commit"] / out["n_sent"]
    return out.reset_index()


def zscore(x):
    return (x - x.mean()) / x.std(ddof=0)


def boot_corr(D: pd.DataFrame, a: str, b: str, rng) -> dict:
    d = D[["firm", a, b]].dropna()
    rho = d[a].corr(d[b], method="spearman")
    firms = d["firm"].unique()
    idx = {f: np.flatnonzero(d["firm"].to_numpy() == f) for f in firms}
    from scipy.stats import rankdata
    va, vb = d[a].to_numpy(), d[b].to_numpy()
    bs = []
    for _ in range(B_BOOT):
        pick = np.concatenate([idx[f] for f in rng.choice(firms, len(firms))])
        bs.append(np.corrcoef(rankdata(va[pick]), rankdata(vb[pick]))[0, 1])
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    dm = d.groupby("firm")[[a, b]].transform(lambda x: x - x.mean())
    within = np.corrcoef(dm[a], dm[b])[0, 1]
    m = d.groupby("firm")[[a, b]].mean()
    between = np.corrcoef(m[a], m[b])[0, 1]
    return {"rho": rho, "ic95": [lo, hi], "intra_r": within, "entre_r": between, "n": len(d)}


def main():
    t0 = time.time()
    scores = load_scores()
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    S = scan(scores)
    S.to_csv(CACHE / "sentences_internal.csv.gz", sep=";", index=False)
    A = aggregate(S)
    D = scores[["doc_key", "firm", "firm_label", "year", "Z_SEN", "Z_SUS", "Z_QUANT",
                "ESGSI", "ESGSI_ext"]].merge(A, on="doc_key", how="left", validate="1:1")
    cb = pd.read_csv(MAIN_DIR / "results" / "climatebert" / "doc_level_cb.csv", sep=";")
    em = pd.read_csv(MAIN_DIR / "results" / "esg_models" / "doc_level_esgm.csv", sep=";")
    D = D.merge(cb[["firm_label", "year", "CTI", "CTI_prob", "spec_share", "commit_share", "sent_net"]],
                on=["firm_label", "year"], how="left", validate="1:1")
    D = D.merge(em[["firm_label", "year", "nz_clim", "nz_red", "action_env", "fin_tone"]],
                on=["firm_label", "year"], how="left", validate="1:1")
    for c in ["UNBACK", "UNBACK_bin", "UNBACK_DB"]:
        D["Z_" + c] = zscore(D[c])
    # ESGSI extendido con UNBACK en lugar de HEDGE (mismos pesos).
    D["ESGSI_ext_unback"] = D["Z_SEN"] - D["Z_SUS"] - 0.5 * D["Z_QUANT"] + 0.5 * D["Z_UNBACK"]

    rng = np.random.default_rng(SEED)
    measures = ["UNBACK", "UNBACK_bin", "UNBACK_DB", "UNBACK_w0", "UNBACK_bin_w0",
                "UNBACK_clim", "UNBACK_bin_clim", "UNBACK_DB_clim", "commit_rate"]
    targets = ["CTI", "CTI_prob", "spec_share", "commit_share", "nz_clim", "nz_red",
               "Z_SEN", "Z_SUS", "Z_QUANT", "sent_net", "fin_tone"]
    corr = {m: {t: boot_corr(D, m, t, rng) for t in targets} for m in measures}
    corr["ESGSI_ext_unback"] = {t: boot_corr(D, "ESGSI_ext_unback", t, rng) for t in ["ESGSI_ext", "ESGSI", "CTI"]}

    import analysis_v3 as A3
    trends = {}
    for c in ["Z_UNBACK", "Z_UNBACK_bin", "Z_UNBACK_DB", "ESGSI_ext_unback", "ESGSI_ext"]:
        s = A3.fe_slope(D, c)
        trends[c] = {"pendiente": s["pendiente"], "ci95": s["ci95"], "p": s["p"]}
    yearly = D.groupby("year")[["UNBACK", "UNBACK_bin", "UNBACK_DB", "commit_rate", "sh_Q", "sh_D",
                                "sh_B", "n_commit"]].mean()

    C = S[S["text"].notna()]
    summary = {
        "frases": int(S["n_sent"].sum()), "frases_compromiso": int(len(C)),
        "frases_compromiso_clima": int(C["clim"].sum()),
        "compromisos_por_informe": D["n_commit"].describe().round(1).to_dict(),
        "informes_con_menos_de_min": int((D["n_commit"] < MIN_COMMIT).sum()),
        "cuota_respaldo": {"Q": float(C["Q"].mean()), "D": float(C["D"].mean()), "B": float(C["B"].mean()),
                           "ninguno": float((~(C["Q"] | C["D"] | C["B"])).mean())},
        "descriptivos": D[["UNBACK", "UNBACK_bin", "UNBACK_DB", "UNBACK_clim"]].describe().round(3).to_dict(),
        "correlaciones": corr, "tendencias_ef": trends,
        "medias_anuales": yearly.round(4).to_dict(),
        "segundos": round(time.time() - t0),
    }
    D.drop(columns=["doc_key", "firm"]).to_csv(OUT / "doc_level_unback.csv", sep=";", index=False)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float),
                                      encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["frases", "frases_compromiso", "frases_compromiso_clima",
                                              "informes_con_menos_de_min", "cuota_respaldo"]}, indent=1))
    for m in measures + ["ESGSI_ext_unback"]:
        print(m, {t: (round(v["rho"], 2), [round(x, 2) for x in v["ic95"]], round(v["intra_r"], 2))
                  for t, v in corr[m].items()})
    print("tendencias:", {k: (round(v["pendiente"], 3), round(v["p"], 3)) for k, v in trends.items()})
    print(yearly.round(3).T)
    print(f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
