"""
stability_v3_e2e.py
-------------------
Pruebas de estabilidad del indice frente al vocabulario v3, en dos modos:

  fija        el texto extraido es siempre el del v3 y solo cambia la lista
              con la que se puntua el SUS (como stability_v3.py).
  re-extraida al quitar terminos se vuelve a aplicar la regla de extraccion,
              asi que desaparecen los parrafos que solo se extraian por ellos,
              y SUS, SEN, QUANT y HEDGE se recalculan sobre el texto que queda.

    python scripts/stability_v3_e2e.py            # usa la cache si existe
    python scripts/stability_v3_e2e.py --rebuild  # rehace la cache (~40 min)

Solo se perturba el vocabulario v3 (directrices, 2). Medidas de score, no de
etiquetas (directrices, 5), para el ESGSI y el ESGSI_ext, contra la referencia
del mismo modo y del mismo indice: Spearman (rho), Pearson (r), solapamiento
de los deciles extremos (top10, bot10) y pendiente temporal (MCO y efectos
fijos de empresa, estimacion puntual; inferencia en robustness_v3.py). Para
cada supresion con nombre, percentil de su rho y de sus solapamientos dentro
de un nulo de borrados aleatorios del mismo tamano.

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
  * QUANT y HEDGE se cuentan por parrafo sobre el texto crudo (el pipeline los
    cuenta sobre el documento entero con los espacios colapsados).

UNIVERSO DE ENTRADAS

Las 434 entradas del v3 (402 terminos y 32 patrones). Quitar una entrada la
quita de la extraccion y, si es un termino, tambien su columna del SUS (salvo
que otro termino que se queda tenga el mismo lema). Los patrones solo actuan
en la extraccion: en el modo fija quitarlos no cambia nada.

Salidas en results/stability_v3/e2e_*.
"""

from __future__ import annotations

import csv
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
from sklearn.feature_extraction.text import CountVectorizer

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR / "scripts"))

import lexical_document_filter as LDF  # noqa: E402
from stability_v3 import Comparer, null_position  # noqa: E402

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
W_QUANT, W_HEDGE = 0.5, 0.5
MODES = (("fija", False), ("re-extraida", True))
INDICES = ("ESGSI", "ESGSI_ext")
MEASURES = ("rho", "r", "top10", "bot10", "pendiente_mco", "pendiente_ef")


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


def read_paragraphs(res: pd.DataFrame):
    """Parrafos de las zonas extraidas, en el orden fijo de la cache."""
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
    return paras, para_doc, para_zone


def quant_hedge_counts(paras: list[str]) -> dict:
    """Aciertos SEN, QUANT y HEDGE por parrafo sobre el texto crudo, con las
    mismas funciones del analizador que usa el pipeline (ESGSIAnalyzer)."""
    import config
    from esgsi_analyzer import ESGSIAnalyzer
    an = ESGSIAnalyzer(keywords=list(config.ESG_KEYWORDS), hedge_words=config.HEDGE_KEYWORDS,
                       quant_patterns=config.QUANT_PATTERNS, positive_words=config.LM_POSITIVE,
                       negative_words=config.LM_NEGATIVE)
    n = len(paras)
    qh = np.zeros(n); hh = np.zeros(n); ht = np.zeros(n); pos = np.zeros(n); neg = np.zeros(n)
    for i, p in enumerate(paras):
        qh[i] = an.quant_hits(p)
        hh[i], ht[i] = an.hedge_counts(p)
        pos[i], neg[i] = an.sen_counts(p)
    return {"quant_hits": qh, "hedge_hits": hh, "hedge_tokens": ht, "pos": pos, "neg": neg,
            "hedge_lexicon_size": len(an.hedge_words)}


def build_cache() -> dict:
    from text_processor import TextProcessor

    t0 = time.time()
    res = pd.read_csv(RESULTS_CSV, sep=";")
    res["Año"] = res["Año"].astype(str)
    paras, para_doc, para_zone = read_paragraphs(res)
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
    lemmas = []
    for i, doc in enumerate(tp.nlp.pipe((tp.prepare(p) for p in paras),
                                        batch_size=2000, n_process=6)):
        lemmas.append(" ".join(tp.lemmas_from_doc(doc)))
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

    print(f"SUS por parrafo ({time.time() - t0:.0f}s)")

    # Entrada -> columna del SUS (lema del termino, si es columna).
    col_of = {t: j for j, t in enumerate(sus_cols)}
    entry_col = []
    for e, k in zip(entries, kinds):
        lemma = tp.preprocess(e).strip() if k == "termino" else ""
        entry_col.append(col_of.get(lemma, -1))

    cache = {
        "entries": entries, "kinds": kinds, "entry_col": np.array(entry_col),
        "sus_cols": sus_cols, "R": R, "L": L, "tokens": tokens,
        "words": np.array([max(len(p.split()), 1) for p in paras], dtype=float),
        "raw_words": np.array([len(p.split()) for p in paras], dtype=float),
        "chars": np.array([len(p) for p in paras], dtype=float),
        "doc": np.array(para_doc), "zone": np.array(para_zone),
        "results": res, **quant_hedge_counts(paras),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "wb") as fh:
        pickle.dump(cache, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"cache guardada ({time.time() - t0:.0f}s)")
    return cache


def ensure_quant_hedge(c: dict) -> dict:
    """Anade a una cache anterior los recuentos QUANT/HEDGE por parrafo, releyendo
    los mismos JSON en el mismo orden (comprobado con el numero de caracteres)."""
    if "quant_hits" in c and "pos" in c:
        return c
    t0 = time.time()
    paras, para_doc, _ = read_paragraphs(c["results"])
    if len(paras) != len(c["doc"]) or not np.array_equal(np.array(para_doc), c["doc"]) \
            or not np.array_equal(np.array([len(p) for p in paras], dtype=float), c["chars"]):
        raise ValueError("los parrafos releidos no coinciden con la cache: usa --rebuild")
    c.update(quant_hedge_counts(paras))
    with open(CACHE, "wb") as fh:
        pickle.dump(c, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"cache ampliada con QUANT/HEDGE por parrafo ({time.time() - t0:.0f}s)")
    return c


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
        self.firms = c["results"]["Compañía"].to_numpy()
        self.EPS = 1e-6
        full = np.ones(len(c["entries"]), dtype=bool)
        self.all_sel = np.ones(len(zone), dtype=bool)
        self.ref = {mo: self.indices(full, rex) for mo, rex in MODES}
        self.cmp = {mo: {ix: Comparer(self.ref[mo][ix], self.years, self.firms) for ix in INDICES}
                    for mo, _ in MODES}
        self.ref_fixed = self.ref["fija"]["ESGSI"]
        self.ref_e2e = self.ref["re-extraida"]["ESGSI"]

    def cols_kept(self, keep_e: np.ndarray) -> np.ndarray:
        """Una columna del SUS se queda si la mantiene alguna entrada que sigue."""
        ec = self.c["entry_col"]
        keep_c = np.zeros(len(self.c["sus_cols"]), dtype=bool)
        keep_c[ec[keep_e & (ec >= 0)]] = True
        return keep_c

    def selection(self, keep_e: np.ndarray, threshold: float = KW_THRESHOLD,
                  window: int = 1) -> np.ndarray:
        """Regla de extraccion sobre los parrafos ya extraidos. Un umbral mayor
        o una ventana menor que los del pipeline seleccionan un subconjunto, asi
        que tambien se pueden simular (no un umbral menor ni una ventana mayor)."""
        if threshold < KW_THRESHOLD or window > 1:
            raise ValueError("solo se simulan reglas mas estrictas que la del pipeline")
        c = self.c
        hits = c["R"] @ keep_e.astype(float)
        hot = hits / c["words"] * 100 >= threshold
        hot &= c["raw_words"] > 0
        sel = hot.copy()
        if window == 1:
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

    def components(self, keep_e: np.ndarray, reextract: bool, **rule):
        c = self.c
        sel = self.selection(keep_e, **rule) if reextract else self.all_sel
        w = sel.astype(float)
        counts = c["L"] @ self.cols_kept(keep_e).astype(float)
        n = self.n_docs

        def dsum(x):
            return np.bincount(c["doc"], weights=x * w, minlength=n)

        sus_num = dsum(counts)
        toks = np.maximum(dsum(c["tokens"]), 1)
        p, q = dsum(c["pos"]), dsum(c["neg"])
        sen = (p - q) / (p + q + self.EPS)
        quant = dsum(c["quant_hits"]) / np.maximum(dsum(c["raw_words"]), 1)
        hedge = dsum(c["hedge_hits"]) / np.maximum(dsum(c["hedge_tokens"]), 1)
        return {"sus": sus_num / toks * 100, "sen": sen, "quant": quant, "hedge": hedge, "sel": sel}

    @staticmethod
    def combine(comp: dict) -> dict:
        e = z(comp["sen"]) - z(comp["sus"])
        return {"ESGSI": e, "ESGSI_ext": e - W_QUANT * z(comp["quant"]) + W_HEDGE * z(comp["hedge"])}

    def indices(self, keep_e: np.ndarray, reextract: bool) -> dict:
        return self.combine(self.components(keep_e, reextract))

    def index(self, keep_e: np.ndarray, reextract: bool) -> np.ndarray:
        return self.indices(keep_e, reextract)["ESGSI"]

    def compare(self, keep_e: np.ndarray, reextract: bool, **rule) -> dict:
        mo = "re-extraida" if reextract else "fija"
        comp = self.components(keep_e, reextract, **rule)
        idx = self.combine(comp)
        words = self.c["raw_words"]
        out = {"texto_conservado": float(words[comp["sel"]].sum() / words.sum())}
        for ix in INDICES:
            out[ix] = self.cmp[mo][ix](idx[ix])
        return out


def flat(d: dict) -> dict:
    """{'ESGSI': {'rho': ..}, 'texto_conservado': ..} -> {'ESGSI_rho': .., 'texto_conservado': ..}"""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update({f"{k}_{kk}": vv for kk, vv in v.items()})
        else:
            out[k] = v
    return out


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
    c = ensure_quant_hedge(c)
    # La cache guarda una copia del results.csv de cuando se construyo; las
    # validaciones se hacen contra el results.csv actual (mismos documentos y orden).
    cur = pd.read_csv(RESULTS_CSV, sep=";")
    cur["Año"] = cur["Año"].astype(str)
    key = ["País", "Compañía", "Documento", "Año"]
    if not cur[key].reset_index(drop=True).equals(c["results"][key].reset_index(drop=True)):
        raise ValueError("results.csv ya no tiene los documentos de la cache: usa --rebuild")
    c["results"] = cur
    m = Model(c)
    res = c["results"]

    # Validacion: el modelo con el v3 completo frente al pipeline real.
    from scipy import stats
    full = np.ones(len(c["entries"]), dtype=bool)
    comp = m.components(full, True)
    ref_e2e = m.ref["re-extraida"]
    val = {
        "parrafos_seleccionados_con_v3": float(comp["sel"].mean()),
        "r_SUS_vs_pipeline": float(np.corrcoef(comp["sus"], res["SUS_density"])[0, 1]),
        "r_SEN_vs_pipeline": float(np.corrcoef(comp["sen"], res["SEN_Score"])[0, 1]),
        "r_QUANT_vs_pipeline": float(np.corrcoef(comp["quant"], res["QUANT_Score"])[0, 1]),
        "r_HEDGE_vs_pipeline": float(np.corrcoef(comp["hedge"], res["HEDGE_Score"])[0, 1]),
    }
    for ix in INDICES:
        val[f"r_{ix}_vs_pipeline"] = float(np.corrcoef(ref_e2e[ix], res[ix])[0, 1])
        val[f"rho_{ix}_vs_pipeline"] = float(stats.spearmanr(ref_e2e[ix], res[ix])[0])
        val[f"dif_abs_max_{ix}_vs_pipeline"] = float(np.abs(ref_e2e[ix] - res[ix]).max())
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
    rng = np.random.default_rng(SEED)

    def both(keep):
        return {mo: m.compare(keep, rex) for mo, rex in MODES}

    variants = {"sin sectoriales": ~sector, "sin los terminos con aviso": ~aviso}
    for p in ("E", "S", "G", "TRANS"):
        variants[f"sin pilar {p}"] = pillar != p
    # 'board director': todas las entradas cuyo lema es esa columna del SUS.
    bd_col = c["sus_cols"].index("board director")
    variants["sin 'board director'"] = c["entry_col"] != bd_col

    # Admision dependiente del final de la muestra (revision): las entradas que
    # con solo los informes de 2018-2019 no habrian pasado la regla de
    # dispersion (>= 50 apariciones en >= 5 empresas, con el umbral de
    # apariciones escalado a 2 de 7 anos) y las que concentran su masa en
    # 2022-2024. Son borrados del vocabulario actual, no otro vocabulario.
    hits_doc = sp.csr_matrix((np.ones(len(c["doc"])), (c["doc"], np.arange(len(c["doc"])))),
                             shape=(m.n_docs, len(c["doc"]))) @ c["R"]
    hits_doc = np.asarray(hits_doc.todense())
    early = np.isin(m.years, [2018, 2019])
    late = np.isin(m.years, [2022, 2023, 2024])
    occ_early = hits_doc[early].sum(axis=0)
    firms_early = np.array([len(set(m.firms[early][hits_doc[early][:, j] > 0])) for j in range(n)])
    occ_total = hits_doc.sum(axis=0)
    late_share = np.divide(hits_doc[late].sum(axis=0), occ_total, out=np.zeros(n), where=occ_total > 0)
    LATE_SHARE = 0.75
    variants["admision tardia (2018-2019)"] = ~((occ_early < 50 * 2 / 7) | (firms_early < 5))
    variants[f"masa >= {LATE_SHARE:.0%} en 2022-2024"] = ~(late_share >= LATE_SHARE)
    temporal = {"umbral_apariciones_2018_2019": 50 * 2 / 7, "umbral_cuota_2022_2024": LATE_SHARE,
                "cuota_esperada_2022_2024_por_texto": float(hits_doc[late].sum() / hits_doc.sum()),
                "entradas_admision_tardia": [e for e, k in zip(c["entries"], variants["admision tardia (2018-2019)"]) if not k],
                "entradas_masa_tardia": [e for e, k in zip(c["entries"], variants[f"masa >= {LATE_SHARE:.0%} en 2022-2024"]) if not k]}

    out_var = {}
    for name, keep in variants.items():
        k = int((~keep).sum())
        r = {"entradas_quitadas": k, "nulo_sorteos": N_DRAWS_MATCHED, **both(keep)}
        null = pd.DataFrame([flat({mo: flat(d) for mo, d in both(_drop(rng, n, k)).items()})
                             for _ in range(N_DRAWS_MATCHED)])
        for mo, _ in MODES:
            for ix in INDICES:
                r[mo][ix].update(null_position(r[mo][ix], null, prefix=f"{mo}_{ix}_"))
        out_var[name] = r
        print(f"{name}: hecho ({time.time() - t0:.0f}s)", flush=True)

    # Sensibilidad a la regla de extraccion (revision), con el vocabulario
    # completo y frente a la referencia re-extraida.
    rules = {"umbral 2 por 100 palabras": {"threshold": 2.0}, "sin parrafos de contexto": {"window": 0},
             "umbral 1.5 por 100 palabras": {"threshold": 1.5}}
    out_rule = {name: m.compare(full, True, **rule) for name, rule in rules.items()}
    for name, r in out_rule.items():
        e = r["ESGSI"]
        print(f"regla {name}: rho={e['rho']:.3f} top={e['top10']:.2f} bot={e['bot10']:.2f} "
              f"pte_ef={e['pendiente_ef']:+.3f} texto={r['texto_conservado']:.1%}", flush=True)

    draws = []
    for f in FRACTIONS:
        k = round(f * n)
        for _ in range(N_DRAWS):
            d = both(_drop(rng, n, k))
            draws.append({"fraccion": f, "k": k, **flat({mo: flat(x) for mo, x in d.items()})})
        print(f"borrado {f:.0%}: hecho ({time.time() - t0:.0f}s)", flush=True)
    draws = pd.DataFrame(draws)
    draws.to_csv(OUT_DIR / "e2e_borrado_aleatorio.csv", sep=";", index=False)

    agg = []
    for (f, k), g in draws.groupby(["fraccion", "k"]):
        row = {"fraccion": f, "k": k, "sorteos": len(g)}
        for mo, _ in MODES:
            for ix in INDICES:
                for mm in MEASURES:
                    s = g[f"{mo}_{ix}_{mm}"]
                    row[f"{mo}_{ix}_{mm}_media"] = s.mean()
                    row[f"{mo}_{ix}_{mm}_p05"] = s.quantile(0.05)
                    row[f"{mo}_{ix}_{mm}_mediana"] = s.median()
                    row[f"{mo}_{ix}_{mm}_p95"] = s.quantile(0.95)
                row[f"{mo}_{ix}_pendiente_ef_positiva"] = int((g[f"{mo}_{ix}_pendiente_ef"] >= 0).sum())
        row["texto_conservado_media"] = g["re-extraida_texto_conservado"].mean()
        agg.append(row)

    summary = {"validacion": val, "entradas": n, "decil_k": m.cmp["fija"]["ESGSI"].k,
               "pesos_ext": {"w_quant": W_QUANT, "w_hedge": W_HEDGE},
               "referencia": {mo: {ix: {"pendiente_mco": m.cmp[mo][ix](m.ref[mo][ix])["pendiente_mco"],
                                        "pendiente_ef": m.cmp[mo][ix].slope_fe(m.ref[mo][ix])}
                                   for ix in INDICES} for mo, _ in MODES},
               "variantes": out_var, "admision_temporal": temporal, "reglas_extraccion": out_rule,
               "borrado_aleatorio": agg,
               "segundos": round(time.time() - t0)}
    (OUT_DIR / "e2e_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1, default=float), encoding="utf-8")

    pd.set_option("display.width", 250)
    for name, r in out_var.items():
        line = f"{name:28s} k={r['entradas_quitadas']:3d}"
        for mo, _ in MODES:
            e = r[mo]["ESGSI"]
            line += (f" | {mo}: rho={e['rho']:.3f} top={e['top10']:.2f} bot={e['bot10']:.2f} "
                     f"pte_ef={e['pendiente_ef']:+.3f} pct_rho={e['percentil_rho']:.2f}")
        print(line + f" texto={r['re-extraida']['texto_conservado']:.1%}")
    show = ["fraccion"] + [f"{mo}_ESGSI_{mm}_mediana" for mo, _ in MODES
                           for mm in ("rho", "top10", "bot10", "pendiente_ef")]
    print(pd.DataFrame(agg)[show].round(4).T.to_string())
    print(f"total {time.time() - t0:.0f}s")


def _drop(rng, n: int, k: int) -> np.ndarray:
    keep = np.ones(n, dtype=bool)
    keep[rng.choice(n, size=k, replace=False)] = False
    return keep


if __name__ == "__main__":
    main()
