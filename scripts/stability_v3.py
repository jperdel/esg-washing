"""
stability_v3.py
---------------
Pruebas de estabilidad del indice frente al vocabulario, con la extraccion
CONGELADA en la del vocabulario v3 con sectoriales (ESG_RUN_TAG=v3,
ESG_INCLUDE_SECTORAL=1). Solo cambia el vocabulario con el que se puntua el
SUS; SEN, QUANT y HEDGE son los de esa misma ejecucion y no dependen del
vocabulario.

    python scripts/stability_v3.py

Congelar la extraccion es deliberado: el vocabulario decide tambien que
parrafos se extraen, y quitar entradas de punta a punta mezclaria lo que se
puntua con lo que se extrae. La re-extraccion se simula aparte
(stability_v3_e2e.py).

Solo se perturba el vocabulario v3 (directrices, 2): no hay comparaciones con
vocabularios ni ejecuciones anteriores.

Referencias: ESGSI = z(SEN) - z(SUS_density) con el v3 completo, y
ESGSI_ext = ESGSI - 0.5 z(QUANT) + 0.5 z(HEDGE). Para cada variante del
vocabulario se miden, contra la referencia del mismo indice, solo medidas de
score (directrices, 5):
    rho            correlacion de Spearman con el score de referencia
    r              correlacion de Pearson
    top10, bot10   solapamiento de los deciles extremos: fraccion de los 34
                   informes del 10 % superior (inferior) de la referencia que
                   siguen en el 10 % superior (inferior) de la variante
    pendiente_mco  pendiente MCO del indice sobre el ano
    pendiente_ef   pendiente con efectos fijos de empresa (estimacion puntual;
                   errores agrupados y wild bootstrap en robustness_v3.py)

Pruebas
    1. Sin sectoriales.
    2. Sin cada pilar (E, S, G, TRANS), sin los terminos con aviso y sin
       'board director'.
    3. Borrado aleatorio de k entradas (k = 10..50 % del vocabulario), y un
       nulo del mismo tamano que cada variante de 1 y 2, para situarla: el
       percentil de su rho y de su solapamiento de deciles dentro del nulo
       (fraccion de sorteos con un valor <= el de la variante; bajo = la
       variante mueve el indice mas que un borrado aleatorio del mismo tamano).
    4. Quitar un termino cada vez: que termino mueve mas el indice.

Salidas
    results/stability_v3/summary.json
    results/stability_v3/variantes.csv
    results/stability_v3/borrado_aleatorio.csv
    results/stability_v3/quitar_uno.csv
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import CountVectorizer

BASE_DIR = Path(__file__).resolve().parent.parent
METADATA_DIR = BASE_DIR / "metadata"
RUN_SUFFIX = "_v3_sec"
CLEAN_CSV = BASE_DIR / "data" / f"clean{RUN_SUFFIX}" / "processed_texts.csv"
RESULTS_CSV = BASE_DIR / "results" / f"metrics{RUN_SUFFIX}" / "results.csv"
OUT_DIR = BASE_DIR / "results" / "stability_v3"

SEED = 20260918
FRACTIONS = [0.10, 0.20, 0.30, 0.40, 0.50]
N_DRAWS = 2000
N_DRAWS_MATCHED = 5000
W_QUANT, W_HEDGE = 0.5, 0.5          # pesos adoptados del ESGSI_ext (config.ESGSI_EXT_WEIGHTS)
DECILE = 0.10
INDICES = ("ESGSI", "ESGSI_ext")
MEASURES = ("rho", "r", "top10", "bot10", "pendiente_mco", "pendiente_ef")

sys.path.insert(0, str(BASE_DIR / "src"))


# ---------------------------------------------------------------------------
# Medidas de score (compartidas con stability_v3_e2e.py y robustness_v3.py)
# ---------------------------------------------------------------------------

def decile_overlap(a: np.ndarray, b: np.ndarray, q: float = DECILE) -> dict:
    """Fraccion del q superior (inferior) de a que esta tambien en el q
    superior (inferior) de b. k = round(q n) informes por extremo."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    k = max(1, int(round(q * len(a))))
    oa, ob = np.argsort(a, kind="stable"), np.argsort(b, kind="stable")
    top = len(set(oa[-k:]) & set(ob[-k:])) / k
    bot = len(set(oa[:k]) & set(ob[:k])) / k
    return {"top": top, "bottom": bot, "k": k}


def z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    s = x.std()
    return np.zeros_like(x) if s == 0 else (x - x.mean()) / s


class Comparer:
    """Compara un score con su referencia con las medidas de score."""

    def __init__(self, ref: np.ndarray, years: np.ndarray, firms: np.ndarray, q: float = DECILE):
        self.ref = np.asarray(ref, float)
        self.years = np.asarray(years, float)
        self.ref_rank = stats.rankdata(self.ref)
        codes, _ = pd.factorize(pd.Series(firms))
        self.codes = codes
        self.nf = codes.max() + 1
        self.cnt = np.bincount(codes)
        self.xd = self.years - (np.bincount(codes, weights=self.years) / self.cnt)[codes]
        self.sxx = float((self.xd ** 2).sum())
        self.k = max(1, int(round(q * len(ref))))
        o = np.argsort(self.ref, kind="stable")
        self.top_ref, self.bot_ref = set(o[-self.k:]), set(o[:self.k])

    def slope_fe(self, idx: np.ndarray) -> float:
        yd = idx - (np.bincount(self.codes, weights=idx) / self.cnt)[self.codes]
        return float(yd @ self.xd / self.sxx)

    def __call__(self, idx: np.ndarray) -> dict:
        idx = np.asarray(idx, float)
        o = np.argsort(idx, kind="stable")
        lr = stats.linregress(self.years, idx)
        return {
            "rho": float(np.corrcoef(stats.rankdata(idx), self.ref_rank)[0, 1]),
            "r": float(np.corrcoef(idx, self.ref)[0, 1]),
            "top10": len(self.top_ref & set(o[-self.k:])) / self.k,
            "bot10": len(self.bot_ref & set(o[:self.k])) / self.k,
            "pendiente_mco": float(lr.slope), "p_pendiente_mco": float(lr.pvalue),
            "pendiente_ef": self.slope_fe(idx),
        }


def null_position(obs: dict, null: pd.DataFrame, prefix: str = "") -> dict:
    """Situa una variante en el nulo de borrados aleatorios del mismo tamano."""
    out = {}
    for m in ("rho", "top10", "bot10"):
        s = null[f"{prefix}{m}"]
        out[f"nulo_mediana_{m}"] = float(s.median())
        out[f"nulo_p05_p95_{m}"] = [float(s.quantile(0.05)), float(s.quantile(0.95))]
        out[f"percentil_{m}"] = float((s <= obs[m]).mean())
    return out


# ---------------------------------------------------------------------------

def read_lexicon(text: str) -> list[str]:
    return [l.strip().lower() for l in text.splitlines() if l.strip() and not l.startswith("#")]


def load_corpus() -> pd.DataFrame:
    """Mismo corpus y misma deduplicacion que main.py, con SEN/QUANT/HEDGE de la ejecucion."""
    df = pd.read_csv(CLEAN_CSV, sep=";", usecols=["Documento", "País", "Compañía", "Año", "clean_text"])
    digest = df["clean_text"].astype(str).map(lambda t: hashlib.md5(t.encode("utf-8")).hexdigest())
    df = df[~digest.duplicated()].reset_index(drop=True)
    res = pd.read_csv(RESULTS_CSV, sep=";")
    key = ["Documento", "País", "Compañía", "Año"]
    df["Año"] = df["Año"].astype(str)
    res["Año"] = res["Año"].astype(str)
    df = df.merge(res[key + ["SEN_Score", "SUS_density", "QUANT_Score", "HEDGE_Score",
                             "ESGSI", "ESGSI_ext"]],
                  on=key, how="inner", validate="one_to_one")
    if len(df) != len(res):
        raise ValueError(f"cruce incompleto: {len(df)} de {len(res)} documentos")
    return df


def counts_for(texts: list[str], vocab: list[str]):
    max_n = max(len(v.split()) for v in vocab)
    cv = CountVectorizer(vocabulary=sorted(set(vocab)), ngram_range=(1, max_n))
    return cv.fit_transform(texts).toarray().astype(float), list(cv.get_feature_names_out())


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_corpus()
    texts = df["clean_text"].astype(str).tolist()
    lengths = np.array([max(len(t.split()), 1) for t in texts], dtype=float)
    years = df["Año"].astype(int).to_numpy()
    firms = df["Compañía"].to_numpy()
    zsen = z(df["SEN_Score"].to_numpy())
    # Parte del ESGSI_ext que no depende del vocabulario, -wq z(QUANT) + wh z(HEDGE).
    # Se toma como ESGSI_ext - ESGSI de la ejecucion (4 decimales cada uno): QUANT y
    # HEDGE estan redondeados a 4 decimales en results.csv y recalcularla desde ellos
    # introduciria un error de redondeo mayor. Se comprueba que ambas coinciden.
    ext_extra = (df["ESGSI_ext"] - df["ESGSI"]).to_numpy(float)
    ext_gap = float(np.abs(ext_extra - (-W_QUANT * z(df["QUANT_Score"]) + W_HEDGE * z(df["HEDGE_Score"]))).max())
    if ext_gap > 0.05:
        raise ValueError(f"ESGSI_ext - ESGSI no es -wq z(QUANT) + wh z(HEDGE) (dif. max {ext_gap:.3g})")
    print(f"corpus: {len(df)} documentos ({time.time() - t0:.0f}s)")

    base = read_lexicon((METADATA_DIR / "esg_terms_lemmatized.txt").read_text(encoding="utf-8"))
    sect = read_lexicon((METADATA_DIR / "esg_terms_sectorial_lemmatized.txt").read_text(encoding="utf-8"))
    vocab = base + [t for t in sect if t not in set(base)]
    C, cols = counts_for(texts, vocab)
    col = {t: i for i, t in enumerate(cols)}
    print(f"matriz de recuentos v3: {C.shape} ({time.time() - t0:.0f}s)")

    def indices(counts_sum: np.ndarray) -> dict:
        e = zsen - z(counts_sum / lengths * 100)
        return {"ESGSI": e, "ESGSI_ext": e + ext_extra}

    # Las referencias recalculadas deben reproducir los indices de la ejecucion.
    ref = indices(C.sum(axis=1))
    for name in INDICES:
        gap = np.abs(ref[name] - df[name].to_numpy()).max()
        if gap > 1e-3:
            raise ValueError(f"la referencia no reproduce el {name} de la ejecucion (dif. max {gap:.4g})")
    cmp = {name: Comparer(ref[name], years, firms) for name in INDICES}

    def compare(counts_sum: np.ndarray) -> dict:
        idx = indices(counts_sum)
        return {name: cmp[name](idx[name]) for name in INDICES}

    def flat(res: dict) -> dict:
        return {f"{name}_{k}": v for name in INDICES for k, v in res[name].items()}

    mass = C.sum(axis=0)
    total_mass = mass.sum()

    # Pilar de cada columna: se lematiza cada termino del v3 con el pipeline real.
    from text_processor import TextProcessor
    sw = read_lexicon((METADATA_DIR / "personal_stopwords.txt").read_text(encoding="utf-8"))
    tp = TextProcessor(extra_sw=sw, spacy_model="en_core_web_md")
    v3 = list(csv.DictReader(open(METADATA_DIR / "ESG_terms_v3.csv", encoding="utf-8-sig"), delimiter=";"))
    pillar, aviso = {}, set()
    for r in v3:
        if r["tipo"] != "termino":
            continue
        lemma = tp.preprocess(r["termino"]).strip()
        if lemma in col:
            pillar.setdefault(lemma, r["pilar"])
            if r["aviso"] and not r["aviso"].startswith("colocacion"):
                aviso.add(lemma)
    sect_cols = [col[t] for t in sect if t in col and t not in set(base)]

    ref_row = {"ESGSI": {"pendiente_mco": cmp["ESGSI"](ref["ESGSI"])["pendiente_mco"],
                         "pendiente_ef": cmp["ESGSI"].slope_fe(ref["ESGSI"])},
               "ESGSI_ext": {"pendiente_mco": cmp["ESGSI_ext"](ref["ESGSI_ext"])["pendiente_mco"],
                             "pendiente_ef": cmp["ESGSI_ext"].slope_fe(ref["ESGSI_ext"])}}
    rows = [{"variante": "v3 completo (referencia)", "entradas": len(cols), "entradas_quitadas": 0,
             "masa_quitada": 0.0, **flat(compare(C.sum(axis=1)))}]

    all_on = np.ones(len(cols), dtype=bool)
    named: dict[str, dict] = {}

    def variant(name: str, keep_mask: np.ndarray) -> None:
        res = compare(C[:, keep_mask].sum(axis=1))
        named[name] = {"entradas": int(keep_mask.sum()), "entradas_quitadas": int((~keep_mask).sum()),
                       "masa_quitada": float(1 - mass[keep_mask].sum() / total_mass), **res}
        rows.append({"variante": name, "entradas": int(keep_mask.sum()),
                     "entradas_quitadas": int((~keep_mask).sum()),
                     "masa_quitada": named[name]["masa_quitada"], **flat(res)})

    m = all_on.copy(); m[sect_cols] = False
    variant("sin sectoriales", m)
    for p in ("E", "S", "G", "TRANS"):
        variant(f"sin pilar {p}", np.array([pillar.get(t) != p for t in cols]))
    variant("sin los terminos con aviso", np.array([t not in aviso for t in cols]))
    if "board director" not in col:
        raise ValueError("'board director' no esta en el vocabulario")
    variant("sin 'board director'", np.array([t != "board director" for t in cols]))

    # Borrado aleatorio.
    rng = np.random.default_rng(SEED)
    n = len(cols)

    def draws(k: int, n_draws: int) -> pd.DataFrame:
        out = []
        for _ in range(n_draws):
            drop = rng.choice(n, size=k, replace=False)
            keep = all_on.copy(); keep[drop] = False
            res = compare(C[:, keep].sum(axis=1))
            out.append({"k": k, "fraccion_entradas": k / n,
                        "masa_quitada": 1 - mass[keep].sum() / total_mass, **flat(res)})
        return pd.DataFrame(out)

    rand = pd.concat([draws(round(f * n), N_DRAWS) for f in FRACTIONS], ignore_index=True)
    rand.to_csv(OUT_DIR / "borrado_aleatorio.csv", sep=";", index=False)
    print(f"borrado aleatorio: {len(rand)} sorteos ({time.time() - t0:.0f}s)")

    # Nulo del mismo tamano para cada variante nombrada.
    for name, res in named.items():
        null = draws(res["entradas_quitadas"], N_DRAWS_MATCHED)
        res["nulo_k"] = res["entradas_quitadas"]
        res["nulo_sorteos"] = N_DRAWS_MATCHED
        res["nulo_masa_quitada_media"] = float(null["masa_quitada"].mean())
        for ix in INDICES:
            res[ix].update(null_position(res[ix], null, prefix=f"{ix}_"))
    print(f"nulos igualados: hecho ({time.time() - t0:.0f}s)")

    # Quitar un termino cada vez.
    lo = []
    for j, t in enumerate(cols):
        keep = all_on.copy(); keep[j] = False
        res = compare(C[:, keep].sum(axis=1))
        lo.append({"termino": t, "pilar": pillar.get(t, ""), "masa": int(mass[j]),
                   "cuota_masa": float(mass[j] / total_mass), "aviso": t in aviso, **flat(res)})
    lo = pd.DataFrame(lo).sort_values(["ESGSI_rho", "masa"], ascending=[True, False])
    lo.to_csv(OUT_DIR / "quitar_uno.csv", sep=";", index=False)

    pd.DataFrame(rows).to_csv(OUT_DIR / "variantes.csv", sep=";", index=False)

    def q05(s): return s.quantile(0.05)
    def q50(s): return s.quantile(0.50)
    def q95(s): return s.quantile(0.95)
    agg = {"sorteos": ("k", "size"), "fraccion_entradas": ("fraccion_entradas", "mean"),
           "masa_quitada": ("masa_quitada", "mean")}
    for ix in INDICES:
        for mm in MEASURES:
            for qn, fn in (("p05", q05), ("mediana", q50), ("p95", q95)):
                agg[f"{ix}_{mm}_{qn}"] = (f"{ix}_{mm}", fn)
            agg[f"{ix}_{mm}_media"] = (f"{ix}_{mm}", "mean")
    by_k = rand.groupby("k").agg(**agg).reset_index()

    loo = {}
    for ix in INDICES:
        s = lo.sort_values(f"{ix}_rho")
        loo[ix] = {"rho_min": float(s[f"{ix}_rho"].min()), "top10_min": float(lo[f"{ix}_top10"].min()),
                   "bot10_min": float(lo[f"{ix}_bot10"].min()),
                   "entradas_con_deciles_intactos": int(((lo[f"{ix}_top10"] == 1) &
                                                         (lo[f"{ix}_bot10"] == 1)).sum()),
                   "pendiente_ef_rango": [float(lo[f"{ix}_pendiente_ef"].min()),
                                          float(lo[f"{ix}_pendiente_ef"].max())],
                   "top": s.head(15).to_dict(orient="records")}
    summary = {
        "documentos": len(df), "entradas_v3": n, "decil_k": cmp["ESGSI"].k,
        "pesos_ext": {"w_quant": W_QUANT, "w_hedge": W_HEDGE},
        "ext_parte_fija_dif_max_vs_recalculo_redondeado": ext_gap,
        "medidas": "rho Spearman, r Pearson, top10/bot10 solapamiento de deciles extremos, "
                   "pendiente MCO y de efectos fijos de empresa (puntual)",
        "referencia": ref_row,
        "pilares_por_entrada": pd.Series(pillar).value_counts().to_dict(),
        "sin_pilar_asignado": [t for t in cols if t not in pillar],
        "terminos_con_aviso": sorted(aviso),
        "variantes": named, "borrado_aleatorio": by_k.to_dict(orient="records"),
        "quitar_uno": loo, "segundos": round(time.time() - t0),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1,
                                                     default=float), encoding="utf-8")

    pd.set_option("display.width", 250)
    show = ["variante", "entradas_quitadas", "masa_quitada"] + [f"{ix}_{mm}" for ix in INDICES
                                                                for mm in ("rho", "top10", "bot10", "pendiente_ef")]
    print(pd.DataFrame(rows)[show].round(4).to_string(index=False))
    print(by_k[["k"] + [f"ESGSI_{mm}_mediana" for mm in ("rho", "top10", "bot10", "pendiente_ef")]]
          .round(4).to_string(index=False))
    print(lo.head(10)[["termino", "masa", "ESGSI_rho", "ESGSI_top10", "ESGSI_bot10"]].round(4).to_string(index=False))
    for name, r in named.items():
        e = r["ESGSI"]
        print(f"{name:28s} pct rho {e['percentil_rho']:.3f} top {e['percentil_top10']:.3f} "
              f"bot {e['percentil_bot10']:.3f}")
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
