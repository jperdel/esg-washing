"""
stability_v3.py
---------------
Pruebas de estabilidad del indice frente al vocabulario, con la extraccion
CONGELADA en la del vocabulario v3 con sectoriales (ESG_RUN_TAG=v3,
ESG_INCLUDE_SECTORAL=1). Solo cambia el vocabulario con el que se puntua el
SUS; el SEN es el de esa misma ejecucion y no depende del vocabulario.

    python scripts/stability_v3.py

Congelar la extraccion es deliberado: el vocabulario decide tambien que
parrafos se extraen, y comparar dos vocabularios de punta a punta mezclaria
lo que se puntua con lo que se extrae. La comparacion de punta a punta con el
indice publicado se da aparte (seccion 'punta a punta').

Referencia: ESGSI(density) = z(SEN) - z(SUS_density) con el v3 completo.
Para cada variante del vocabulario se mide, contra la referencia:
    etiquetas que cambian (umbral ESGSI > 0), correlacion de Pearson y de
    Spearman del indice, informes senalados y pendiente temporal (MCO del
    ESGSI sobre el ano, como en el paper).

Pruebas
    1. Sin sectoriales (peticion de Francisco).
    2. Vocabularios anteriores sobre la misma extraccion: 154, 289, 289+22.
    3. Sin cada pilar, y sin los 6 terminos con aviso.
    4. Borrado aleatorio de k entradas (k = 10..50 % del vocabulario), y un
       nulo del mismo tamano que cada variante de 1 y 3, para situarla.
    5. Quitar un termino cada vez: que termino mueve mas el indice.

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
import os
import subprocess
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
PAPER_CSV = BASE_DIR / "results" / "metrics_baseline_paper" / "results.csv"
OUT_DIR = BASE_DIR / "results" / "stability_v3"

SEED = 20260918
FRACTIONS = [0.10, 0.20, 0.30, 0.40, 0.50]
N_DRAWS = 2000
N_DRAWS_MATCHED = 5000
OLD_VOCABS = {                       # commit donde vive cada vocabulario anterior
    "154": ("ce6804a", "metadata/esg_terms_lemmatized.txt", None),
    "289": ("6a13a9f", "metadata/esg_terms_lemmatized.txt", None),
    "289+22 sect.": ("6a13a9f", "metadata/esg_terms_lemmatized.txt",
                     "metadata/esg_terms_sectorial_lemmatized.txt"),
}

sys.path.insert(0, str(BASE_DIR / "src"))


def read_lexicon(text: str) -> list[str]:
    return [l.strip().lower() for l in text.splitlines() if l.strip() and not l.startswith("#")]


def git_file(commit: str, path: str) -> str:
    return subprocess.run(["git", "show", f"{commit}:{path}"], cwd=BASE_DIR,
                          capture_output=True, text=True, encoding="utf-8", check=True).stdout


def load_corpus() -> pd.DataFrame:
    """Mismo corpus y misma deduplicacion que main.py, con el SEN de la ejecucion."""
    df = pd.read_csv(CLEAN_CSV, sep=";", usecols=["Documento", "País", "Compañía", "Año", "clean_text"])
    digest = df["clean_text"].astype(str).map(lambda t: hashlib.md5(t.encode("utf-8")).hexdigest())
    df = df[~digest.duplicated()].reset_index(drop=True)
    res = pd.read_csv(RESULTS_CSV, sep=";")
    key = ["Documento", "País", "Compañía", "Año"]
    df["Año"] = df["Año"].astype(str)
    res["Año"] = res["Año"].astype(str)
    df = df.merge(res[key + ["SEN_Score", "SUS_density", "ESGSI"]], on=key, how="inner",
                  validate="one_to_one")
    if len(df) != len(res):
        raise ValueError(f"cruce incompleto: {len(df)} de {len(res)} documentos")
    return df


def counts_for(texts: list[str], vocab: list[str]) -> np.ndarray:
    max_n = max(len(v.split()) for v in vocab)
    cv = CountVectorizer(vocabulary=sorted(set(vocab)), ngram_range=(1, max_n))
    return cv.fit_transform(texts).toarray().astype(float), list(cv.get_feature_names_out())


def z(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return np.zeros_like(x) if s == 0 else (x - x.mean()) / s


class Scorer:
    def __init__(self, sen: np.ndarray, lengths: np.ndarray, years: np.ndarray, ref: np.ndarray):
        self.zsen, self.len, self.years, self.ref = z(sen), lengths, years, ref
        self.ref_flag = ref > 0

    def index(self, counts_sum: np.ndarray) -> np.ndarray:
        return self.zsen - z(counts_sum / self.len * 100)

    def compare(self, idx: np.ndarray) -> dict:
        flag = idx > 0
        slope = stats.linregress(self.years, idx)
        return {
            "senalados": int(flag.sum()),
            "cambios": int((flag != self.ref_flag).sum()),
            "r": float(np.corrcoef(idx, self.ref)[0, 1]),
            "rho": float(stats.spearmanr(idx, self.ref)[0]),
            "pendiente": float(slope.slope), "p_pendiente": float(slope.pvalue),
        }


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_corpus()
    texts = df["clean_text"].astype(str).tolist()
    lengths = np.array([max(len(t.split()), 1) for t in texts], dtype=float)
    years = df["Año"].astype(int).to_numpy()
    sen = df["SEN_Score"].to_numpy()
    print(f"corpus: {len(df)} documentos ({time.time() - t0:.0f}s)")

    base = read_lexicon((METADATA_DIR / "esg_terms_lemmatized.txt").read_text(encoding="utf-8"))
    sect = read_lexicon((METADATA_DIR / "esg_terms_sectorial_lemmatized.txt").read_text(encoding="utf-8"))
    vocab = base + [t for t in sect if t not in set(base)]
    C, cols = counts_for(texts, vocab)
    col = {t: i for i, t in enumerate(cols)}
    print(f"matriz de recuentos v3: {C.shape} ({time.time() - t0:.0f}s)")

    # La referencia recalculada debe reproducir el indice de la ejecucion.
    ref_idx = z(sen) - z(C.sum(axis=1) / lengths * 100)
    gap = np.abs(ref_idx - df["ESGSI"].to_numpy()).max()
    if gap > 1e-3:
        raise ValueError(f"la referencia no reproduce el ESGSI de la ejecucion (dif. max {gap:.4g})")
    sc = Scorer(sen, lengths, years, ref_idx)
    ref = sc.compare(ref_idx)
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

    rows = [{"variante": "v3 completo (referencia)", "entradas": len(cols), "masa_quitada": 0.0, **ref}]

    def variant(name: str, keep_mask: np.ndarray) -> dict:
        out = {"variante": name, "entradas": int(keep_mask.sum()),
               "masa_quitada": float(1 - mass[keep_mask].sum() / total_mass),
               **sc.compare(sc.index(C[:, keep_mask].sum(axis=1)))}
        rows.append(out)
        return out

    all_on = np.ones(len(cols), dtype=bool)
    named = {}
    m = all_on.copy(); m[sect_cols] = False
    named["sin sectoriales"] = variant("sin sectoriales", m)
    for p in ("E", "S", "G", "TRANS"):
        m = np.array([pillar.get(t) != p for t in cols])
        named[f"sin pilar {p}"] = variant(f"sin pilar {p}", m)
    m = np.array([t not in aviso for t in cols])
    named["sin los terminos con aviso"] = variant("sin los terminos con aviso", m)

    # Vocabularios anteriores sobre la misma extraccion (recuentos propios).
    for name, (commit, path, sect_path) in OLD_VOCABS.items():
        old = read_lexicon(git_file(commit, path))
        if sect_path:
            old += [t for t in read_lexicon(git_file(commit, sect_path)) if t not in set(old)]
        Co, _ = counts_for(texts, old)
        rows.append({"variante": f"vocabulario anterior {name}", "entradas": len(set(old)),
                     "masa_quitada": float("nan"), **sc.compare(sc.index(Co.sum(axis=1)))})
        print(f"vocabulario {name}: hecho ({time.time() - t0:.0f}s)")

    # Borrado aleatorio.
    rng = np.random.default_rng(SEED)
    n = len(cols)

    def draws(k: int, n_draws: int) -> pd.DataFrame:
        out = []
        for _ in range(n_draws):
            drop = rng.choice(n, size=k, replace=False)
            keep = all_on.copy(); keep[drop] = False
            res = sc.compare(sc.index(C[:, keep].sum(axis=1)))
            out.append({"k": k, "cambios": res["cambios"], "r": res["r"],
                        "pendiente": res["pendiente"],
                        "masa_quitada": 1 - mass[keep].sum() / total_mass})
        return pd.DataFrame(out)

    rand = pd.concat([draws(round(f * n), N_DRAWS) for f in FRACTIONS], ignore_index=True)
    rand.to_csv(OUT_DIR / "borrado_aleatorio.csv", sep=";", index=False)
    print(f"borrado aleatorio: {len(rand)} sorteos ({time.time() - t0:.0f}s)")

    # Nulo del mismo tamano para cada variante nombrada.
    for name, res in named.items():
        k = n - res["entradas"]
        null = draws(k, N_DRAWS_MATCHED)
        res["nulo_k"] = k
        res["nulo_mediana_cambios"] = float(null["cambios"].median())
        res["nulo_p05_p95_cambios"] = [float(null["cambios"].quantile(q)) for q in (0.05, 0.95)]
        res["nulo_r_medio"] = float(null["r"].mean())
        res["percentil_cambios"] = float((null["cambios"] <= res["cambios"]).mean())
        res["nulo_masa_quitada_media"] = float(null["masa_quitada"].mean())
    print(f"nulos igualados: hecho ({time.time() - t0:.0f}s)")

    # Quitar un termino cada vez.
    lo = []
    for j, t in enumerate(cols):
        keep = all_on.copy(); keep[j] = False
        res = sc.compare(sc.index(C[:, keep].sum(axis=1)))
        lo.append({"termino": t, "pilar": pillar.get(t, ""), "masa": int(mass[j]),
                   "cuota_masa": float(mass[j] / total_mass), "cambios": res["cambios"],
                   "r": res["r"], "aviso": t in aviso})
    lo = pd.DataFrame(lo).sort_values(["cambios", "masa"], ascending=False)
    lo.to_csv(OUT_DIR / "quitar_uno.csv", sep=";", index=False)

    # Punta a punta: indice publicado (154 sobre su propia extraccion) frente al v3.
    paper = pd.read_csv(PAPER_CSV, sep=";")
    paper["Año"] = paper["Año"].astype(str)
    both = df.drop(columns="ESGSI").assign(idx=ref_idx).merge(paper[["Documento", "País", "Compañía", "Año", "ESGSI"]],
                                        on=["Documento", "País", "Compañía", "Año"], how="inner")
    e2e = {"documentos_cruzados": len(both),
           "senalados_publicado": int((both["ESGSI"] > 0).sum()),
           "senalados_v3": int((both["idx"] > 0).sum()),
           "cambios": int(((both["ESGSI"] > 0) != (both["idx"] > 0)).sum()),
           "r": float(np.corrcoef(both["ESGSI"], both["idx"])[0, 1]),
           "rho": float(stats.spearmanr(both["ESGSI"], both["idx"])[0]),
           "pendiente_publicado": float(stats.linregress(both["Año"].astype(int), both["ESGSI"]).slope)}

    pd.DataFrame(rows).to_csv(OUT_DIR / "variantes.csv", sep=";", index=False)
    by_k = rand.groupby("k").agg(
        sorteos=("cambios", "size"), mediana_cambios=("cambios", "median"),
        p05=("cambios", lambda s: s.quantile(0.05)), p95=("cambios", lambda s: s.quantile(0.95)),
        r_medio=("r", "mean"), r_p05=("r", lambda s: s.quantile(0.05)),
        masa_quitada=("masa_quitada", "mean"),
        pendiente_p05=("pendiente", lambda s: s.quantile(0.05)),
        pendiente_p95=("pendiente", lambda s: s.quantile(0.95))).reset_index()
    summary = {
        "documentos": len(df), "entradas_v3": n, "referencia": ref,
        "pilares_por_entrada": pd.Series(pillar).value_counts().to_dict(),
        "sin_pilar_asignado": [t for t in cols if t not in pillar],
        "terminos_con_aviso": sorted(aviso),
        "variantes": rows, "borrado_aleatorio": by_k.to_dict(orient="records"),
        "quitar_uno_top": lo.head(15).to_dict(orient="records"),
        "punta_a_punta_vs_publicado": e2e, "segundos": round(time.time() - t0),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1,
                                                     default=float), encoding="utf-8")

    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows)[["variante", "entradas", "masa_quitada", "senalados", "cambios",
                              "r", "pendiente"]].round(4).to_string(index=False))
    print(by_k.round(4).to_string(index=False))
    print(lo.head(10).round(4).to_string(index=False))
    print(json.dumps(e2e, indent=1))
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
