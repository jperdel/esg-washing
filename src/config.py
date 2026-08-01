from pathlib import Path
from loguru import logger

# PATHS
BASE_DIR = Path(__file__).resolve().parent.parent

METADATA_DIR = BASE_DIR / "metadata"

DATA_DIR = BASE_DIR / "data"
PDF_DATA_DIR = DATA_DIR / "pdf"
LEXICAL_DATA_DIR        = DATA_DIR / "chunks_lexical"
CLEAN_DATA_DIR = DATA_DIR / "clean"

RESULTS_DIR = BASE_DIR / "results"
METRICS_RESULTS_DIR = RESULTS_DIR / "metrics"
LDA_RESULTS_DIR = RESULTS_DIR / "lda"

# METADATA
METADATA_EXCEL = METADATA_DIR / "muestras_informes.xlsx"

# LDA PARAMETERS
K_TOPICS_LIST = [8, 12, 16, 20, 24]  # exploración enfocada — un run por k, sin estabilidad
ALPHA_LIST = ['auto']
K_ITERS = 5              # 5 runs por k para promediar variabilidad de inicialización
LDA_N_SEEDS = 0          # 0 = skip stability analysis
LDA_TOPN_STABILITY = 15

# LEXICAL DOCUMENT FILTER PARAMETERS
LEXICAL_KW_THRESHOLD  = 1.0   # keywords per 100 words to flag a paragraph as ESG-hot
LEXICAL_CONTEXT_PARAS = 1     # paragraphs of context window around each hot paragraph
LEXICAL_MIN_ZONE_LEN  = 150   # minimum chars for a zone to be emitted

# NLP PARAMETERS
SPACY_MODEL = 'en_core_web_md'


# ---------------------------------------------------------------------------
# Léxicos generados por scripts/build_lexicons.py
# ---------------------------------------------------------------------------

def _load_generated_lexicon(generated: Path, source: Path) -> list[str]:
    """
    Carga un léxico generado, comprobando que existe y que no está obsoleto
    respecto a su fichero fuente.

    Estos ficheros no se editan a mano: los produce scripts/build_lexicons.py
    pasando cada término por el mismo TextProcessor que preprocesa el corpus,
    de forma que vocabulario y corpus queden en la misma representación.
    """
    if not generated.exists():
        raise FileNotFoundError(
            f"Falta el léxico generado '{generated.name}'. "
            f"Ejecuta:  python scripts/build_lexicons.py"
        )

    if source.exists() and source.stat().st_mtime > generated.stat().st_mtime:
        logger.warning(
            f"'{source.name}' se ha modificado después de generar '{generated.name}'. "
            f"Vuelve a ejecutar:  python scripts/build_lexicons.py"
        )

    with open(generated, "r", encoding="utf-8") as f:
        return [
            line.strip().lower()
            for line in f
            if line.strip() and not line.startswith("#")
        ]


# Vocabulario TF-IDF del SUS — formas lematizadas, alineadas con el corpus.
ESG_KEYWORDS = _load_generated_lexicon(
    METADATA_DIR / "esg_terms_lemmatized.txt",
    METADATA_DIR / "esg_terms.txt",
)

# HEDGE — categorías L&M indicativas de lenguaje impreciso/especulativo,
# lematizadas para poder cruzarse con el corpus (el diccionario original
# viene en formas flexionadas y solo cruzaba el 46 % de sus entradas).
HEDGE_KEYWORDS = set(_load_generated_lexicon(
    METADATA_DIR / "lm_hedge_lemmatized.txt",
    METADATA_DIR / "RAW_LM_dictionary.csv",
))

with open(METADATA_DIR / 'personal_stopwords.txt', 'r', encoding='utf-8') as f:
    PERSONAL_SW = f.read().split("\n")

# LDA STOPWORDS — adicionales, solo para topic modeling (no afectan ESGSI)
with open(METADATA_DIR / 'lda_stopwords.txt', 'r', encoding='utf-8') as f:
    LDA_STOPWORDS = set(
        line.strip().lower()
        for line in f
        if line.strip() and not line.startswith("#")
    )

# QUANT SCORE — patrones RegEx para contenido cuantificable y marcos regulatorios.
# IMPORTANTE: se aplican sobre el texto CRUDO (raw_text), nunca sobre el texto
# preprocesado: el preprocesado elimina los tokens numéricos, así que sobre él
# 'percentages' y 'large_numbers' darían siempre cero.
QUANT_PATTERNS = {
    # Porcentajes (ej. "42%", "3.5 %")
    "percentages":     r'\b\d+(?:[.,]\d+)?\s*%',
    # Cifras métricas grandes (4+ dígitos), excluyendo años 19xx/20xx: un informe
    # no es más cuantitativo por citar muchas fechas.
    "large_numbers":   r'\b(?!(?:19|20)\d{2}\b)\d{4,}\b',
    # Unidades de emisiones y energía
    "units":           r'\b(?:tonne|ton|mt|ktco2|co2e?|ghg|kwh|mwh|gwh|twh|mw|gw|litre|liter|m3|cubic meter)\b',
    # Marcos regulatorios de referencia
    "frameworks":      r'\b(?:gri|tcfd|sasb|issb|sdg|ungc|sfdr|csrd|un global compact|paris agreement|taxonomy)\b',
}

# ESGSI EXTENDIDO — pesos para los dos nuevos componentes
# ESGSI_ext = Z(SEN) - Z(SUS) - w_quant*Z(QUANT) + w_hedge*Z(HEDGE)
# QUANT alto → menos washing (el informe tiene datos duros) → se resta
# HEDGE alto → más washing (el informe elude compromisos concretos) → se suma
ESGSI_EXT_WEIGHTS = {
    "w_quant": 0.5,
    "w_hedge": 0.5,
}
