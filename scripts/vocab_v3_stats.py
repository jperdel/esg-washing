"""
vocab_v3_stats.py
-----------------
Cifras del paper (seccion 5, vocabulario v3) que describen el vocabulario
VIGENTE, recalculadas desde los ficheros del repositorio. No escribe nada
fuera de results/analysis_v3/vocab_stats.json.

    PYTHONIOENCODING=utf-8 ESG_RUN_TAG=v3 ESG_INCLUDE_SECTORAL=1 \
        python scripts/vocab_v3_stats.py

Bloques
    v3                  composicion por pilar x tipo (terminos simples,
                        colocaciones, patrones, sectoriales)
    exported            entradas de los ficheros exportados al pipeline
    lemmatised          vocabulario de puntuacion lematizado (V), estructura de
                        n-gramas, frases protegidas y lo que recuperan
    mass                masa de ocurrencias del vocabulario de puntuacion sobre
                        la extraccion v3 (informes unicos): entradas principales,
                        cuotas por pilar y de las sectoriales
    board_of_directors  frase exacta y lema, ambos sobre la extraccion v3
    hedge               lexico HEDGE (maestro L&M) y sus aciertos sobre el
                        texto crudo; cota inferior de 'may' como mes

Fuera por directiva del autor (paper/_research/directrices.md): auditoria KWIC,
precision, rondas de consenso, tablas de retirados y cualquier cifra previa al
v3 o de antes/despues.

Entradas: metadata/ESG_terms_v3.csv, ESG_terms_final.csv (solo la marca de
colocacion), esg_terms*.txt, esg_patterns*.txt, esg_terms*_lemmatized.txt,
esg_protected_phrases.txt, lm_hedge.txt, personal_stopwords.txt;
data/clean_v3_sec/processed_texts.csv (solo lectura).

El vocabulario lematizado se recalcula con el MISMO TextProcessor que usa
build_lexicons.py (sin escribir los ficheros generados) y se comprueba que
coincide con metadata/esg_terms*_lemmatized.txt y con config.ESG_KEYWORDS.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("ESG_RUN_TAG", "v3")
os.environ.setdefault("ESG_INCLUDE_SECTORAL", "1")

BASE = Path(__file__).resolve().parent.parent
MET = BASE / "metadata"
OUT = BASE / "results" / "analysis_v3" / "vocab_stats.json"
PROCESSED = BASE / "data" / "clean_v3_sec" / "processed_texts.csv"

sys.path.insert(0, str(BASE / "scripts"))
sys.path.insert(0, str(BASE / "src"))


def rd(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh, delimiter=";"))


def read_list(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as fh:
        return [l.strip() for l in fh if l.strip() and not l.lstrip().startswith("#")]


BOARD_RE = re.compile(r"\bboards?[\s\-]+of[\s\-]+directors?\b", re.IGNORECASE)


def _dedup(df):
    """La deduplicacion de main.py: md5 del clean_text, primera aparicion."""
    import hashlib
    seen, keep = set(), []
    for i, text in df["clean_text"].items():
        h = hashlib.md5(str(text).encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            keep.append(i)
    return df.loc[keep]


def load_v3_texts():
    import pandas as pd
    return _dedup(pd.read_csv(PROCESSED, sep=";", encoding="utf-8"))


def board_of_directors_counts(v3) -> dict:
    """
    'board of directors' sobre los informes unicos de la extraccion v3:
      - frase exacta (board(s) of director(s)) en el texto crudo;
      - el lema 'board director' que cuenta el SUS sobre el texto lematizado.
    """
    from sklearn.feature_extraction.text import CountVectorizer
    exact_v3 = int(sum(len(BOARD_RE.findall(str(t))) for t in v3["raw_text"]))
    cv = CountVectorizer(vocabulary=["board director"], ngram_range=(2, 2), lowercase=True)
    lemma_v3 = int(cv.transform(v3["clean_text"].fillna("").astype(str)).sum())
    return {"exact_phrase_v3_raw": exact_v3, "lemma_v3_clean": lemma_v3, "docs_v3": len(v3)}


def hedge_raw_audit(v3) -> dict:
    """
    Que cuenta HEDGE sobre el texto crudo (esgsi_analyzer.calculate_hedge_scores):
    palabras = rachas de letras con guiones internos, en minusculas; acierto si
    la palabra esta en lm_hedge.txt (formas originales del maestro L&M). Se dan
    las palabras que mas aportan y se acota por abajo el residuo de 'may' como
    mes ('May 2023', '15 May').
    """
    hedge = set(read_list(MET / "lm_hedge.txt"))
    word_re = re.compile(r"[a-z]+(?:-[a-z]+)*")
    month_re = re.compile(r"\bMay\s+\d|\d\s+May\b")
    hits, words, month = Counter(), 0, 0
    for text in v3["raw_text"].fillna("").astype(str):
        toks = word_re.findall(text.lower())
        words += len(toks)
        hits.update(t for t in toks if t in hedge)
        month += len(month_re.findall(text))
    total = sum(hits.values())
    top = hits.most_common(15)
    return {
        "raw_words": words, "raw_hits": total,
        "raw_hits_per_100_words": round(100 * total / words, 3),
        "raw_hits_top": top,
        "raw_hits_top5_share": round(sum(n for _, n in top[:5]) / total, 4),
        "forms_matched": len(hits),
        "may_hits": hits.get("may", 0), "may_month_lower_bound": month,
        "may_month_share_of_may_lower_bound": round(month / hits["may"], 4) if hits.get("may") else None,
        "may_month_share_of_all_hits_lower_bound": round(month / total, 4),
    }


def main() -> None:
    S: dict = {}
    final = rd(MET / "ESG_terms_final.csv")
    v3 = rd(MET / "ESG_terms_v3.csv")
    final_by = {r["termino"]: r for r in final}

    # ------------------------------------------------------------ composicion v3
    # Colocacion: entrada marcada como colocacion en la lista de trabajo o
    # introducida como colocacion en v3 (de ESG_terms_final.csv solo se usa la marca).
    def is_colloc(r):
        return r["origen"].startswith("v3: sustituye") or final_by.get(r["termino"], {}).get("colocacion") == "1"

    comp = {}
    for p in ("E", "S", "G", "TRANS", "all"):
        rows = [r for r in v3 if p == "all" or r["pilar"] == p]
        comp[p] = {
            "patterns": sum(r["tipo"] == "patron" for r in rows),
            "collocations": sum(r["tipo"] == "termino" and is_colloc(r) for r in rows),
            "single_terms": sum(r["tipo"] == "termino" and not is_colloc(r) for r in rows),
            "sectoral": sum(r["sectorial"] == "1" for r in rows),
            "total": len(rows),
        }
    S["v3"] = {
        "entries": len(v3),
        "by_type": dict(Counter(r["tipo"] for r in v3)),
        "composition": comp,
        "sectoral_by_type": dict(Counter(r["tipo"] for r in v3 if r["sectorial"] == "1")),
    }

    # ------------------------------------------------------------ ficheros exportados
    S["exported"] = {f: len(read_list(MET / f)) for f in (
        "esg_terms.txt", "esg_terms_sectorial.txt", "esg_patterns.txt", "esg_patterns_sectorial.txt",
        "esg_terms_lemmatized.txt", "esg_terms_sectorial_lemmatized.txt", "esg_protected_phrases.txt")}

    # ------------------------------------------------------------ lematizacion
    from text_processor import TextProcessor, load_protected_phrases
    from build_lexicons import COLLAPSE_ALLOWED
    sw = [l.strip().lower() for l in open(MET / "personal_stopwords.txt", encoding="utf-8")
          if l.strip() and not l.startswith("#")]
    # Mismo esquema que build_lexicons.py: el procesador del pipeline lee las
    # frases protegidas de metadata/; uno sin proteccion muestra que les pasaria.
    pairs = load_protected_phrases()
    proc = TextProcessor(extra_sw=sw, spacy_model="en_core_web_md")
    plain = TextProcessor(extra_sw=sw, spacy_model="en_core_web_md", protected_phrases=[])
    protected_tokens = {proc.preprocess(tok).strip() for _, tok in pairs}
    lem_report = {}
    union: list[str] = []
    for src, gen in (("esg_terms.txt", "esg_terms_lemmatized.txt"),
                     ("esg_terms_sectorial.txt", "esg_terms_sectorial_lemmatized.txt")):
        terms = [t.lower() for t in read_list(MET / src)]
        groups, dropped, coll_ex, coll_ok, prot = defaultdict(list), [], [], [], []
        for t in terms:
            lem = proc.preprocess(t).strip()
            if not lem:
                dropped.append(t)
                continue
            if lem in protected_tokens:
                prot.append([t, plain.preprocess(t).strip(), lem])
            elif len(t.split()) > 1 and len(lem.split()) == 1:
                if t not in COLLAPSE_ALLOWED:
                    coll_ex.append([t, lem])
                    continue
                coll_ok.append([t, lem])
            groups[lem].append(t)
        vocab = sorted(groups)
        assert vocab == sorted(read_list(MET / gen)), f"{gen} no coincide con la lematizacion actual"
        lem_report[src] = {
            "source_terms": len(terms), "lemmas": len(vocab), "dropped": dropped,
            "protected": prot, "collapsed_excluded": coll_ex, "collapsed_allowed": coll_ok,
            "collisions": {k: v for k, v in groups.items() if len(v) > 1},
        }
        union += [v for v in vocab if v not in set(union)]
    ngram = Counter(len(v.split()) for v in union)
    S["lemmatised"] = {"per_file": lem_report, "union_size": len(union),
                       "protected_phrases": [tok for _, tok in pairs],
                       "protected_terms_recovered": sum(len(r["protected"]) for r in lem_report.values()),
                       "protected_recovered_legend": "[termino, lema sin proteccion, token protegido]",
                       "union_ngram": dict(sorted(ngram.items())),
                       "multiword_share": round(1 - ngram[1] / len(union), 4),
                       "max_ngram": max(ngram),
                       "personal_stopwords": len(sw)}

    # ------------------------------------------------------------ masa en la extraccion v3
    import config
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer
    from analysis_v3 import load_vocab_and_pillars
    texts = load_v3_texts()
    vocab = list(config.ESG_KEYWORDS)
    assert sorted(vocab) == sorted(union), "config.ESG_KEYWORDS no coincide con los lexicos lematizados"
    _, pillar, sectoral, _, _, _ = load_vocab_and_pillars(MET / "ESG_terms_v3.csv")
    cv = CountVectorizer(vocabulary=vocab, ngram_range=(1, max(len(v.split()) for v in vocab)))
    C = cv.transform(texts["clean_text"].fillna("").astype(str))
    occ = np.asarray(C.sum(axis=0)).ravel().astype(float)
    dfj = np.asarray((C > 0).sum(axis=0)).ravel()
    names = list(cv.get_feature_names_out())
    tot = float(occ.sum())
    order = np.argsort(-occ, kind="stable")
    by_p = defaultdict(float)
    for j, e in enumerate(names):
        by_p[pillar.get(e, "?")] += occ[j]
    S["mass"] = {
        "docs": len(texts), "V": len(names), "occurrences_total": int(tot),
        "entries_never_fire": int((occ == 0).sum()),
        "share_top10": round(float(occ[order[:10]].sum() / tot), 4),
        "share_top50": round(float(occ[order[:50]].sum() / tot), 4),
        "share_by_pillar": {p: round(v / tot, 4) for p, v in sorted(by_p.items())},
        "entries_by_pillar": dict(sorted(Counter(pillar.get(e, "?") for e in names).items())),
        "sectoral_entries": len(sectoral),
        "sectoral_share": round(float(sum(occ[j] for j, e in enumerate(names) if e in sectoral) / tot), 4),
        "top20": [{"entry": names[j], "pillar": pillar.get(names[j]), "sectoral": names[j] in sectoral,
                   "occurrences": int(occ[j]), "docs": int(dfj[j]),
                   "share": round(float(occ[j] / tot), 4)} for j in order[:20]],
    }

    # ------------------------------------------------------------ board of directors
    S["board_of_directors"] = board_of_directors_counts(texts)

    # ------------------------------------------------------------ HEDGE
    # Maestro Loughran-McDonald (pysentiment2), la misma fuente que SEN.
    import pandas as pd
    import pysentiment2
    master = pd.read_csv(Path(pysentiment2.__file__).parent / "static" / "LM.csv")
    S["hedge"] = {
        "master_counts": {"Uncertainty": int((master["Uncertainty"] != 0).sum()),
                          "Constraining": int((master["Constraining"] != 0).sum()),
                          "WeakModal": int((master["Modal"] == 3).sum()),
                          "StrongModal": int((master["Modal"] == 1).sum())},
        "hedge_forms": len(read_list(MET / "lm_hedge.txt")),
    }
    S["hedge"].update(hedge_raw_audit(texts))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(S, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(S, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
