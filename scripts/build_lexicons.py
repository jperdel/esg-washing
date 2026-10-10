"""
build_lexicons.py
-----------------
Genera los léxicos lematizados que consume el pipeline en tiempo de ejecución.

    python scripts/build_lexicons.py

Produce dos ficheros en metadata/:

  · esg_terms_lemmatized.txt   ← esg_terms.txt
        Vocabulario TF-IDF para el SUS. Cada término se pasa por el MISMO
        TextProcessor que preprocesa el corpus, de modo que vocabulario y
        corpus quedan en la misma representación. Sin esto, términos como
        "co2" o "human rights" nunca podrían puntuar.

  · lm_hedge.txt               ← pysentiment2/static/LM.csv
        Palabras de las categorías Uncertainty y WeakModal del diccionario
        maestro de Loughran-McDonald, sin "risk"/"risks", en su forma
        original: el HEDGE se calcula sobre el texto crudo.

  · lm_positive.txt, lm_negative.txt  ← pysentiment2/static/LM.csv
        Listas Positive y Negative del mismo diccionario, en su forma
        original: el SEN se calcula sobre el texto crudo, sin reducir a raíz.

    python scripts/build_lexicons.py --lm-only   # solo los léxicos de L&M

Este script NO importa config.py a propósito: config.py lee los ficheros que
este script genera, así que importarlo crearía una dependencia circular.
Las constantes de abajo deben mantenerse sincronizadas con config.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR     = Path(__file__).resolve().parent.parent
METADATA_DIR = BASE_DIR / "metadata"
sys.path.insert(0, str(BASE_DIR / "src"))

import re

import pandas as pd
from loguru import logger
from text_processor import TextProcessor, PROTECTED_PHRASES_FILE

# --- Sincronizar con config.py -------------------------------------------
SPACY_MODEL    = "en_core_web_md"
HEDGE_CATEGORIES = ["Uncertainty", "WeakModal"]
# "risk"/"risks" están en Uncertainty, pero nombrar un riesgo no es matizar una
# afirmación: son casi un tercio de los aciertos y miden cuánto se habla de
# riesgo, que en los informes es contenido exigido (TCFD, ESRS).
HEDGE_EXCLUDED = {"risk", "risks"}

# Colapsos revisados a mano y admitidos: la palabra superviviente es lo
# bastante específica en este corpus como para no generar falsos positivos.
#   "due diligence" -> "diligence" : 4.330 ocurrencias, casi todas del propio
#                                    sintagma; "diligence" suelto es raro.
#   "eu taxonomy"   -> "taxonomy"  : 16.662 ocurrencias, en memorias europeas
#                                    "taxonomy" es la Taxonomía de la UE.
# Cualquier otro colapso se excluye del TF-IDF automáticamente.
COLLAPSE_ALLOWED = {"due diligence", "eu taxonomy"}


def _read_terms(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as fh:
        return [
            line.strip().lower()
            for line in fh
            if line.strip() and not line.startswith("#")
        ]


def build_esg_vocabulary(
    processor: TextProcessor,
    source: str = "esg_terms.txt",
    target: str = "esg_terms_lemmatized.txt",
    protected_tokens: frozenset[str] = frozenset(),
) -> list[str]:
    """
    Lematiza el vocabulario con el pipeline real y avisa de los términos que no
    sobreviven al preprocesado (los que el filtro de stopwords o de dígitos
    deja vacíos, y que por tanto serían keywords muertas en el TF-IDF).

    Se invoca dos veces: para el vocabulario base y para el sectorial, que se
    mantiene en un fichero aparte porque su inclusión es una decisión de diseño
    y no un detalle de implementación.
    """
    src = METADATA_DIR / source
    terms = _read_terms(src)
    logger.info(f"{source}: {len(terms)} términos canónicos.")

    lemmatized: dict[str, str] = {}
    dropped:    list[str]      = []
    changed:    list[tuple[str, str]] = []
    collapsed:  list[tuple[str, str]] = []

    for term in terms:
        lemma = processor.preprocess(term).strip()
        if not lemma:
            dropped.append(term)
            continue
        if lemma != term:
            changed.append((term, lemma))
        # Una expresión de varias palabras que se queda en una sola ha perdido
        # justo la parte que la hacía específica: "paris agreement" -> "agreement".
        # Como entrada de TF-IDF dispararía con cualquier uso genérico, así que
        # se EXCLUYE del vocabulario salvo que esté revisada en COLLAPSE_ALLOWED.
        # Sigue activa en el regex sobre texto crudo, que no lematiza.
        if lemma in protected_tokens:
            pass                          # frase protegida: un token a propósito
        elif len(term.split()) > 1 and len(lemma.split()) == 1:
            collapsed.append((term, lemma))
            if term not in COLLAPSE_ALLOWED:
                continue
        lemmatized.setdefault(lemma, term)

    if dropped:
        logger.warning(
            f"{len(dropped)} términos NO sobreviven al preprocesado y quedan "
            f"fuera del TF-IDF (siguen activos en el regex sobre texto crudo): {dropped}"
        )
    if collapsed:
        logger.warning(
            f"{len(collapsed)} expresiones colapsan a una sola palabra al lematizar. "
            "Revisa si la palabra superviviente es lo bastante específica; si no lo es, "
            "sácalas de esg_terms.txt y ponlas como patrón en _EXTRA_PATTERNS:"
        )
        for original, lemma in collapsed:
            logger.warning(f"    {original:32s} -> {lemma}")
    if changed:
        logger.info(f"{len(changed)} términos cambian de forma al lematizar. Ejemplos:")
        for original, lemma in changed[:12]:
            logger.info(f"    {original:32s} -> {lemma}")

    vocab = sorted(lemmatized)
    max_n = max(len(v.split()) for v in vocab)
    logger.success(
        f"Vocabulario TF-IDF: {len(vocab)} entradas únicas, n-grama máximo = {max_n}."
    )

    out = METADATA_DIR / target
    header = (
        "# GENERADO POR scripts/build_lexicons.py — NO EDITAR A MANO.\n"
        f"# Fuente: {source} | Vocabulario TF-IDF del SUS (formas lematizadas).\n"
    )
    out.write_text(header + "\n".join(vocab) + "\n", encoding="utf-8")
    logger.success(f"Escrito: {out}")
    return vocab


def _protected_token(term: str) -> str:
    """'well-being' -> 'wellbeing', '2030 agenda' -> 'agenda2030'."""
    token = re.sub(r"[^a-z0-9]", "", term.lower())
    lead = re.match(r"\d+", token)
    return token[lead.end():] + lead.group() if lead else token


def _protected_regex(term: str) -> str:
    """Como el regex del extractor, pero admite también la forma soldada."""
    parts = re.split(r"[\s\-]+", term.strip().lower())
    return r"[\s\-]*".join(re.escape(p) for p in parts) + "s?"


def build_protected_phrases(plain: TextProcessor, sources: list[str]) -> list[tuple[str, str]]:
    """
    Frases del vocabulario que el preprocesado destruiría: las que se quedan
    vacías ("well-being": sus dos palabras son stopwords) o colapsan a una sola
    palabra genérica ("iso 14001" -> "iso", "say on pay" -> "pay"). Antes se
    excluían del TF-IDF; ahora se protegen como un único token en el propio
    corpus, de modo que puntúan con su sentido completo y sin generar falsos
    positivos con la palabra superviviente. COLLAPSE_ALLOWED queda como está.

    `plain` debe ser un TextProcessor SIN protección: es el que revela el colapso.
    """
    pairs: dict[str, str] = {}
    for source in sources:
        for term in _read_terms(METADATA_DIR / source):
            lemma = plain.preprocess(term).strip()
            empty = not lemma
            collapsed = (len(re.split(r"[\s\-]+", term)) > 1 and len(lemma.split()) == 1
                         and term not in COLLAPSE_ALLOWED)
            if empty or collapsed:
                pairs.setdefault(_protected_regex(term), _protected_token(term))
    ordered = sorted(pairs.items(), key=lambda kv: -len(kv[0]))   # las largas primero
    header = (
        "# GENERADO POR scripts/build_lexicons.py — NO EDITAR A MANO.\n"
        "# Frases que el preprocesado destruiría; TextProcessor las sustituye por un\n"
        "# único token antes de tokenizar. Formato: regex<TAB>token.\n"
    )
    PROTECTED_PHRASES_FILE.write_text(
        header + "".join(f"{rx}\t{tok}\n" for rx, tok in ordered), encoding="utf-8")
    logger.success(f"{len(ordered)} frases protegidas: {[t for _, t in ordered]}")
    return ordered


def _lm_master() -> pd.DataFrame:
    """Diccionario maestro de L&M tal como lo distribuye pysentiment2."""
    import pysentiment2
    return pd.read_csv(Path(pysentiment2.__file__).parent / "static" / "LM.csv")


def build_hedge_lexicon() -> list[str]:
    """
    Léxico HEDGE: las formas del diccionario maestro de Loughran-McDonald en
    las categorías Uncertainty y WeakModal, tal como las distribuye
    pysentiment2 (el mismo diccionario que usa el SEN), sin HEDGE_EXCLUDED.
    En el maestro, Modal = 3 es WeakModal.

    StrongModal (must, will) y Constraining (requirements, required) quedan
    fuera: expresan obligación o compromiso, lo contrario de una cautela.

    Se guardan las formas originales, sin lematizar: el HEDGE se calcula sobre
    el texto CRUDO, donde aparecen flexionadas. No se parte de
    RAW_LM_dictionary.csv, que no es el maestro de L&M sino una versión
    ampliada con sinónimos (767 Uncertainty frente a 297).
    """
    df = _lm_master()
    mask = (df["Uncertainty"] != 0) | (df["Modal"] == 3)
    counts = {"Uncertainty": int((df["Uncertainty"] != 0).sum()),
              "WeakModal": int((df["Modal"] == 3).sum())}
    vocab = sorted(set(df.loc[mask, "Word"].astype(str).str.lower().str.strip()) - HEDGE_EXCLUDED)
    logger.success(f"Léxico HEDGE: {len(vocab)} formas únicas de L&M {counts}, sin {sorted(HEDGE_EXCLUDED)}.")

    out = METADATA_DIR / "lm_hedge.txt"
    header = (
        "# GENERADO POR scripts/build_lexicons.py — NO EDITAR A MANO.\n"
        "# Fuente: diccionario maestro Loughran-McDonald (pysentiment2/static/LM.csv).\n"
        f"# Categorias: {counts}, sin {sorted(HEDGE_EXCLUDED)}. Formas originales, para cruzar con texto crudo.\n"
    )
    out.write_text(header + "\n".join(vocab) + "\n", encoding="utf-8")
    logger.success(f"Escrito: {out}")
    return vocab


def build_sentiment_lexicons() -> dict[str, list[str]]:
    """
    Listas Positive y Negative de L&M en sus formas originales, para el SEN.
    El diccionario es una lista de formas flexionadas; se cruza con las
    palabras del texto crudo tal cual, sin lematizar ni reducir a raíz.
    """
    df = _lm_master()
    out_lists = {}
    for cat, fname in (("Positive", "lm_positive.txt"), ("Negative", "lm_negative.txt")):
        vocab = sorted(set(df.loc[df[cat] != 0, "Word"].astype(str).str.lower().str.strip()))
        header = (
            "# GENERADO POR scripts/build_lexicons.py — NO EDITAR A MANO.\n"
            f"# Fuente: diccionario maestro Loughran-McDonald (pysentiment2/static/LM.csv), categoria {cat}.\n"
            "# Formas originales, para cruzar con texto crudo sin reducir a raiz.\n"
        )
        (METADATA_DIR / fname).write_text(header + "\n".join(vocab) + "\n", encoding="utf-8")
        logger.success(f"Léxico {cat}: {len(vocab)} formas -> {fname}")
        out_lists[cat] = vocab
    return out_lists


def main():
    logger.remove()
    logger.add(sys.stderr, format="<level>{level: <8}</level> | {message}", level="INFO")

    if "--lm-only" in sys.argv[1:]:
        build_hedge_lexicon()
        build_sentiment_lexicons()
        return

    sw_path = METADATA_DIR / "personal_stopwords.txt"
    personal_sw = _read_terms(sw_path)
    logger.info(f"personal_stopwords.txt: {len(personal_sw)} stopwords.")

    plain = TextProcessor(extra_sw=personal_sw, spacy_model=SPACY_MODEL, protected_phrases=[])
    sources = ["esg_terms.txt", "esg_terms_sectorial.txt"]
    pairs = build_protected_phrases(plain, sources)

    processor = TextProcessor(extra_sw=personal_sw, spacy_model=SPACY_MODEL, protected_phrases=pairs)
    # Cada token debe llegar al TF-IDF como UNA palabra. spaCy puede
    # lematizarlo ("wellbeing" -> "wellbee"), y da igual: el corpus pasa por la
    # misma lematización, así que término y texto siguen coincidiendo.
    tokens = frozenset(processor.preprocess(tok) for _, tok in pairs)
    for (_, tok), lemma in zip(pairs, (processor.preprocess(t) for _, t in pairs)):
        if len(lemma.split()) != 1:
            raise ValueError(f"el token protegido '{tok}' no sobrevive como una palabra: '{lemma}'")

    build_esg_vocabulary(processor, protected_tokens=tokens)
    build_esg_vocabulary(processor, source="esg_terms_sectorial.txt",
                         target="esg_terms_sectorial_lemmatized.txt", protected_tokens=tokens)
    build_hedge_lexicon()
    build_sentiment_lexicons()

    logger.success("Léxicos regenerados. Recuerda re-ejecutar el pipeline con "
                   "run_preproc=True si has cambiado el preprocesado.")


if __name__ == "__main__":
    main()
