"""
stability_v3_e2e.py
-------------------
Pruebas de estabilidad del indice frente al vocabulario, en dos modos:

  fija        el texto extraido es siempre el del v3 y solo cambia la lista
              con la que se puntua el SUS (como stability_v3.py).
  re-extraida al quitar terminos se vuelve a aplicar la regla de extraccion,
              asi que desaparecen los parrafos que solo se extraian por ellos,
              y SUS y SEN se recalculan sobre el texto que queda.

    python scripts/stability_v3_e2e.py            # usa la cache si existe
    python scripts/stability_v3_e2e.py --rebuild  # rehace la cache (~40 min)

POR QUE LA RE-EXTRACCION SE PUEDE SIMULAR SIN LEER LOS PDF

Con un vocabulario que es un subconjunto del v3, un parrafo "caliente"
(>= 1 termino por cada 100 palabras) lo era tambien con el v3, y sus vecinos
de contexto tambien se extrajeron con el v3. Asi que el texto que se
extraeria es un subconjunto de los parrafos ya extraidos, y basta con
reaplicar la regla sobre ellos: parrafos calientes, un parrafo de contexto a
cada lado dentro de la misma zona, fusion de ventanas contiguas y descarte
de zonas de menos de 150 caracteres.

Aproximaciones (se miden en la validacion del principio):
  * Cada parrafo se lematiza por separado; el pipeline lematiza el documento
    entero. El etiquetado de spaCy depende poco del contexto.
  * Al quitar una entrada, su aparicion deja de contar para decidir si el
    parrafo esta caliente; no se comprueba si otra entrada mas corta la habria
    capturado ("carbon" dentro de "carbon footprint"). Subestima un poco los
    disparos de la variante, asi que exagera ligeramente lo que se pierde.

UNIVERSO DE ENTRADAS

Las 434 entradas del v3 (402 terminos y 32 patrones). Quitar una entrada la
quita de la extraccion y, si es un termino, tambien su columna del SUS (salvo
que otro termino que se queda tenga el mismo lema). Los patrones solo actuan
en la extraccion: en el modo fija quitarlos no cambia nada.

Salidas en results/stability_v3/e2e_*.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path

os.environ["ESG_INCLUDE_SECTORAL"] = "1"
os.environ["ESG_RUN_TAG"] = "v3"

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
from sklearn.feature_extraction.text import CountVectorizer

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

import lexical_document_filter as LDF  # noqa: E402

METADATA_DIR = BASE_DIR / "metadata"
JSON_DIR = BASE_DIR / "data" / "chunks_lexical_v3_sec"
RESULTS_CSV = BASE_DIR / "results" / "metrics_v3_sec" / "results.csv"
OUT_DIR = BASE_DIR / "results" / "stability_v3"
CACHE = OUT_DIR / "e2e_cache.pkl"

KW_THRESHOLD = 1.0
MIN_ZONE_LEN = 150
SEED = 20260919
FRACTIONS = [0.10, 0.20, 0.30, 0.40, 0.50]
N_DRAWS = 1000
N_DRAWS_MATCHED = 1000


def read_lexicon(path: Path) -> list[str]:
    return [l.strip().lower() for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#")]


# ---------------------------------------------------------------------------
# Cache por parrafo
# ---------------------------------------------------------------------------

def entry_regex() -> tuple[re.Pattern, list[str], list[str]]:
    """El mismo regex que usa el extractor, con un grupo por entrada para saber
    cual ha disparado. Mismo orden de alternativas, luego mismas apariciones."""
    terms = sorted(LDF._ESG_TERMS, key=lambda t: (-len(t.split()), -len(t)))
    pats = list(LDF._EXTRA_PATTERNS)
    for p in pats:
        if re.compile(p).groups:
            raise ValueError(f"el patron {p!r} tiene grupos de captura")
    pieces = [LDF._term_to_pattern(t) for t in terms] + pats
    rx = re.compile(r"\b(?:" + "|".join(f"({p})" for p in pieces) + r")\b", re.IGNORECASE)
    return rx, terms + pats, ["termino"] * len(terms) + ["patron"] * len(pats)


def build_cache() -> dict:
    from text_processor import TextProcessor
    import pysentiment2 as ps

    t0 = time.time()
    res = pd.read_csv(RESULTS_CSV, sep=";")
    res["Año"] = res["Año"].astype(str)
    keys = {(r["País"], r["Compañía"], r["Documento"]): i for i, r in res.iterrows()}

    paras, para_doc, para_zone = [], [], []
    zone_id = 0
    for f in sorted(JSON_DIR.rglob("*.json")):
        key = (f.parent.parent.name, f.parent.name, f.name)
        if key not in keys:
            continue                      # el duplicado que main.py descarta
        d = json.loads(f.read_text(encoding="utf-8"))
        for z in d["zones"]:
            for p in z["text"].split("\n\n"):
                paras.append(p)
                para_doc.append(keys[key])
                para_zone.append(zone_id)
            zone_id += 1
    if len(set(para_doc)) != len(res):
        raise ValueError("no se han encontrado todos los documentos del results.csv")
    print(f"{len(paras):,} parrafos de {len(res)} documentos ({time.time() - t0:.0f}s)")

    # Disparos del extractor por parrafo y entrada.
    rx, entries, kinds = entry_regex()
    rows, cols = [], []
    for i, p in enumerate(paras):
        for m in rx.finditer(p):
            rows.append(i)
            cols.append(m.lastindex - 1)
    R = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(paras), len(entries)))
    kw_total = np.array([len(LDF._ESG_KW_RE.findall(p)) for p in paras[:20000]])
    if not np.array_equal(kw_total, np.asarray(R[:20000].sum(axis=1)).ravel()):
        raise ValueError("el regex por entradas no reproduce los disparos del extractor")
    print(f"disparos por entrada: {int(R.sum()):,} ({time.time() - t0:.0f}s)")

    # Lematizacion por parrafo, identica a TextProcessor.preprocess.
    sw = read_lexicon(METADATA_DIR / "personal_stopwords.txt")
    tp = TextProcessor(extra_sw=sw, spacy_model="en_core_web_md")
    url = re.compile(r"https?://\S+|www\.\S+")
    lemmas = []
    for i, doc in enumerate(tp.nlp.pipe((url.sub("", p).lower() for p in paras),
                                        batch_size=2000, n_process=6)):
        lemmas.append(" ".join(
            t.lemma_ for t in doc
            if not t.is_stop and not t.is_punct and not t.is_space
            and tp._is_content_token(t.text) and len(t.text) > 2
            and t.lemma_ not in tp.custom_stopwords))
        if i % 100000 == 0:
            print(f"  lematizados {i:,} ({time.time() - t0:.0f}s)")

    base = read_lexicon(METADATA_DIR / "esg_terms_lemmatized.txt")
    sect = read_lexicon(METADATA_DIR / "esg_terms_sectorial_lemmatized.txt")
    vocab = base + [t for t in sect if t not in set(base)]
    max_n = max(len(v.split()) for v in vocab)
    cv = CountVectorizer(vocabulary=sorted(set(vocab)), ngram_range=(1, max_n))
    L = cv.fit_transform(lemmas).tocsr().astype(float)
    sus_cols = list(cv.get_feature_names_out())
    tokens = np.array([len(x.split()) for x in lemmas], dtype=float)

    lm = ps.LM()
    pos = np.zeros(len(paras)); neg = np.zeros(len(paras))
    for i, x in enumerate(lemmas):
        s = lm.get_score(lm.tokenize(x))
        pos[i], neg[i] = s["Positive"], s["Negative"]
    print(f"SUS y SEN por parrafo ({time.time() - t0:.0f}s)")

    # Entrada -> columna del SUS (lema del termino, si es columna).
    col_of = {t: j for j, t in enumerate(sus_cols)}
    entry_col = []
    for e, k in zip(entries, kinds):
        lemma = tp.preprocess(e).strip() if k == "termino" else ""
        entry_col.append(col_of.get(lemma, -1))

    cache = {
        "entries": entries, "kinds": kinds, "entry_col": np.array(entry_col),
        "sus_cols": sus_cols, "R": R, "L": L, "tokens": tokens, "pos": pos, "neg": neg,
        "words": np.array([max(len(p.split()), 1) for p in paras], dtype=float),
        "raw_words": np.array([len(p.split()) for p in paras], dtype=float),
        "chars": np.array([len(p) for p in paras], dtype=float),
        "doc": np.array(para_doc), "zone": np.array(para_zone),
        "results": res,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "wb") as fh:
        pickle.dump(cache, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"cache guardada ({time.time() - t0:.0f}s)")
    return cache


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------

def z(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return np.zeros_like(x) if s == 0 else (x - x.mean()) / s


class Model:
    def __init__(self, c: dict):
        self.c = c
        self.n_docs = len(c["results"])
        zone = c["zone"]
        self.same_prev = np.r_[False, zone[1:] == zone[:-1]]
        self.same_next = np.r_[zone[:-1] == zone[1:], False]
        self.years = c["results"]["Año"].astype(int).to_numpy()
        self.EPS = 1e-6
        full = np.ones(len(c["entries"]), dtype=bool)
        self.all_sel = np.ones(len(zone), dtype=bool)
        self.ref_fixed = self.index(full, reextract=False)
        self.ref_e2e = self.index(full, reextract=True)

    def cols_kept(self, keep_e: np.ndarray) -> np.ndarray:
        """Una columna del SUS se queda si la mantiene alguna entrada que sigue."""
        ec = self.c["entry_col"]
        keep_c = np.zeros(len(self.c["sus_cols"]), dtype=bool)
        keep_c[ec[keep_e & (ec >= 0)]] = True
        return keep_c

    def selection(self, keep_e: np.ndarray) -> np.ndarray:
        c = self.c
        hits = c["R"] @ keep_e.astype(float)
        hot = hits / c["words"] * 100 >= KW_THRESHOLD
        hot &= c["raw_words"] > 0
        sel = hot.copy()
        sel[1:] |= hot[:-1] & self.same_prev[1:]
        sel[:-1] |= hot[1:] & self.same_next[:-1]
        # Tramos contiguos de seleccionados dentro de una zona = zonas nuevas.
        start = sel & ~(np.r_[False, sel[:-1]] & self.same_prev)
        run = np.cumsum(start) - 1
        idx = np.flatnonzero(sel)
        run_len = np.bincount(run[idx], weights=c["chars"][idx] + 2) - 2
        ok = np.zeros(len(sel), dtype=bool)
        ok[idx] = run_len[run[idx]] >= MIN_ZONE_LEN
        return ok

    def components(self, keep_e: np.ndarray, reextract: bool):
        c = self.c
        sel = self.selection(keep_e) if reextract else self.all_sel
        w = sel.astype(float)
        counts = c["L"] @ self.cols_kept(keep_e).astype(float)
        n = self.n_docs
        sus_num = np.bincount(c["doc"], weights=counts * w, minlength=n)
        toks = np.maximum(np.bincount(c["doc"], weights=c["tokens"] * w, minlength=n), 1)
        if reextract:
            p = np.bincount(c["doc"], weights=c["pos"] * w, minlength=n)
            q = np.bincount(c["doc"], weights=c["neg"] * w, minlength=n)
        else:
            p = np.bincount(c["doc"], weights=c["pos"], minlength=n)
            q = np.bincount(c["doc"], weights=c["neg"], minlength=n)
        sen = (p - q) / (p + q + self.EPS)
        return sus_num / toks * 100, sen, sel

    def index(self, keep_e: np.ndarray, reextract: bool) -> np.ndarray:
        sus, sen, _ = self.components(keep_e, reextract)
        return z(sen) - z(sus)

    def compare(self, keep_e: np.ndarray, reextract: bool) -> dict:
        ref = self.ref_e2e if reextract else self.ref_fixed
        sus, sen, sel = self.components(keep_e, reextract)
        idx = z(sen) - z(sus)
        slope = stats.linregress(self.years, idx)
        words = self.c["raw_words"]
        return {
            "senalados": int((idx > 0).sum()),
            "cambios": int(((idx > 0) != (ref > 0)).sum()),
            "r": float(np.corrcoef(idx, ref)[0, 1]),
            "pendiente": float(slope.slope), "p_pendiente": float(slope.pvalue),
            "texto_conservado": float(words[sel].sum() / words.sum()),
        }


# ---------------------------------------------------------------------------
# Experimentos
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()
    if "--rebuild" in sys.argv or not CACHE.exists():
        c = build_cache()
    else:
        with open(CACHE, "rb") as fh:
            c = pickle.load(fh)
    m = Model(c)
    res = c["results"]

    # Validacion: el modelo con el v3 completo frente al pipeline real.
    sus_full, sen_full, sel_full = m.components(np.ones(len(c["entries"]), bool), True)
    val = {
        "parrafos_seleccionados_con_v3": float(sel_full.mean()),
        "r_SUS_vs_pipeline": float(np.corrcoef(sus_full, res["SUS_density"])[0, 1]),
        "r_SEN_vs_pipeline": float(np.corrcoef(sen_full, res["SEN_Score"])[0, 1]),
        "r_ESGSI_vs_pipeline": float(np.corrcoef(m.ref_e2e, res["ESGSI"])[0, 1]),
        "etiquetas_distintas_vs_pipeline": int(((m.ref_e2e > 0) != (res["ESGSI"] > 0)).sum()),
        "senalados_modelo": int((m.ref_e2e > 0).sum()),
        "senalados_pipeline": int((res["ESGSI"] > 0).sum()),
    }
    print("validacion:", json.dumps(val, indent=1))

    # Metadatos de cada entrada.
    v3 = list(csv.DictReader(open(METADATA_DIR / "ESG_terms_v3.csv", encoding="utf-8-sig"),
                             delimiter=";"))
    meta = {}
    for r in v3:
        k = r["termino"].strip() if r["tipo"] == "patron" else r["termino"].strip().lower()
        meta[k] = r
    missing = [e for e in c["entries"] if e not in meta]
    if missing:
        raise ValueError(f"entradas sin fila en ESG_terms_v3.csv: {missing[:5]}")
    pillar = np.array([meta[e]["pilar"] for e in c["entries"]])
    sector = np.array([meta[e]["sectorial"] == "1" for e in c["entries"]])
    aviso = np.array([bool(meta[e]["aviso"]) and not meta[e]["aviso"].startswith("colocacion")
                      for e in c["entries"]])
    n = len(c["entries"])
    full = np.ones(n, dtype=bool)
    rng = np.random.default_rng(SEED)

    def both(keep):
        return {"fija": m.compare(keep, False), "re-extraida": m.compare(keep, True)}

    variants = {"sin sectoriales": ~sector, "sin los terminos con aviso": ~aviso}
    for p in ("E", "S", "G", "TRANS"):
        variants[f"sin pilar {p}"] = pillar != p
    out_var = {}
    for name, keep in variants.items():
        k = int((~keep).sum())
        r = {"entradas_quitadas": k, **both(keep)}
        null = [both(_drop(rng, n, k)) for _ in range(N_DRAWS_MATCHED)]
        for mode in ("fija", "re-extraida"):
            flips = np.array([d[mode]["cambios"] for d in null])
            r[mode]["nulo_mediana_cambios"] = float(np.median(flips))
            r[mode]["percentil_cambios"] = float((flips <= r[mode]["cambios"]).mean())
        out_var[name] = r
        print(f"{name}: hecho ({time.time() - t0:.0f}s)")

    draws = []
    for f in FRACTIONS:
        k = round(f * n)
        for _ in range(N_DRAWS):
            d = both(_drop(rng, n, k))
            draws.append({"fraccion": f, "k": k,
                          **{f"{mo}_{key}": val for mo in d for key, val in d[mo].items()}})
        print(f"borrado {f:.0%}: hecho ({time.time() - t0:.0f}s)")
    draws = pd.DataFrame(draws)
    draws.to_csv(OUT_DIR / "e2e_borrado_aleatorio.csv", sep=";", index=False)

    agg = []
    for (f, k), g in draws.groupby(["fraccion", "k"]):
        row = {"fraccion": f, "k": k, "sorteos": len(g)}
        for mo in ("fija", "re-extraida"):
            row[f"{mo}_cambios_media"] = g[f"{mo}_cambios"].mean()
            row[f"{mo}_cambios_mediana"] = g[f"{mo}_cambios"].median()
            row[f"{mo}_r_media"] = g[f"{mo}_r"].mean()
            row[f"{mo}_pendiente_p05"] = g[f"{mo}_pendiente"].quantile(0.05)
            row[f"{mo}_pendiente_p95"] = g[f"{mo}_pendiente"].quantile(0.95)
            row[f"{mo}_pendiente_positiva"] = int((g[f"{mo}_pendiente"] >= 0).sum())
        row["texto_conservado_media"] = g["re-extraida_texto_conservado"].mean()
        agg.append(row)

    summary = {"validacion": val, "entradas": n,
               "referencia_fija": m.compare(full, False),
               "referencia_re-extraida": m.compare(full, True),
               "variantes": out_var, "borrado_aleatorio": agg,
               "segundos": round(time.time() - t0)}
    (OUT_DIR / "e2e_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1, default=float), encoding="utf-8")

    pd.set_option("display.width", 250)
    for name, r in out_var.items():
        print(f"{name:28s} k={r['entradas_quitadas']:3d} | fija: {r['fija']['cambios']:3d} "
              f"r={r['fija']['r']:.3f} pte={r['fija']['pendiente']:+.3f} | re-extraida: "
              f"{r['re-extraida']['cambios']:3d} r={r['re-extraida']['r']:.3f} "
              f"pte={r['re-extraida']['pendiente']:+.3f} texto={r['re-extraida']['texto_conservado']:.1%}")
    print(pd.DataFrame(agg).round(4).T.to_string())
    print(f"total {time.time() - t0:.0f}s")


def _drop(rng, n: int, k: int) -> np.ndarray:
    keep = np.ones(n, dtype=bool)
    keep[rng.choice(n, size=k, replace=False)] = False
    return keep


if __name__ == "__main__":
    main()
