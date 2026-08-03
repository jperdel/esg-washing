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

  · lm_hedge_lemmatized.txt    ← RAW_LM_dictionary.csv
        Palabras de las categorías Uncertainty, WeakModal, StrongModal y
        Constraining de Loughran-McDonald, lematizadas para poder cruzarse
        con el corpus. El diccionario original está en formas flexionadas.

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

import pandas as pd
from loguru import logger
from text_processor import TextProcessor

# --- Sincronizar con config.py -------------------------------------------
SPACY_MODEL    = "en_core_web_md"
HEDGE_CATEGORIES = ["Uncertainty", "WeakModal", "StrongModal", "Constraining"]

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
        if len(term.split()) > 1 and len(lemma.split()) == 1:
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


def build_hedge_lexicon(processor: TextProcessor) -> list[str]:
    """
    Lematiza las categorías de incertidumbre/modalidad de Loughran-McDonald.

    Se guarda la unión de la forma original y su lema: las originales que ya
    son lemas siguen cruzando, y las flexionadas aportan su lema. Las formas
    que no aparecen en el corpus simplemente nunca hacen match.
    """
    src = METADATA_DIR / "RAW_LM_dictionary.csv"
    df  = pd.read_csv(src)
    words = (
        df[df["sentiment"].isin(HEDGE_CATEGORIES)]["word"]
        .astype(str).str.lower().str.strip().tolist()
    )
    logger.info(f"RAW_LM_dictionary.csv: {len(words)} palabras en {HEDGE_CATEGORIES}.")

    hedge: set[str] = set(words)
    for token in processor.nlp.pipe(words, batch_size=500):
        for tok in token:
            if tok.lemma_.strip():
                hedge.add(tok.lemma_.lower().strip())

    vocab = sorted(w for w in hedge if w)
    logger.success(
        f"Léxico HEDGE: {len(vocab)} formas "
        f"({len(vocab) - len(set(words))} añadidas por lematización)."
    )

    out = METADATA_DIR / "lm_hedge_lemmatized.txt"
    header = (
        "# GENERADO POR scripts/build_lexicons.py — NO EDITAR A MANO.\n"
        "# Fuente: RAW_LM_dictionary.csv | Categorias: "
        + ", ".join(HEDGE_CATEGORIES) + "\n"
        "# Union de la forma original y su lema, para cruzar con corpus lematizado.\n"
    )
    out.write_text(header + "\n".join(vocab) + "\n", encoding="utf-8")
    logger.success(f"Escrito: {out}")
    return vocab


def main():
    logger.remove()
    logger.add(sys.stderr, format="<level>{level: <8}</level> | {message}", level="INFO")

    sw_path = METADATA_DIR / "personal_stopwords.txt"
    personal_sw = _read_terms(sw_path)
    logger.info(f"personal_stopwords.txt: {len(personal_sw)} stopwords.")

    processor = TextProcessor(extra_sw=personal_sw, spacy_model=SPACY_MODEL)

    build_esg_vocabulary(processor)
    build_esg_vocabulary(processor, source="esg_terms_sectorial.txt",
                         target="esg_terms_sectorial_lemmatized.txt")
    build_hedge_lexicon(processor)

    logger.success("Léxicos regenerados. Recuerda re-ejecutar el pipeline con "
                   "run_preproc=True si has cambiado el preprocesado.")


if __name__ == "__main__":
    main()
