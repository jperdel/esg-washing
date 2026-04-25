from pathlib import Path
import pandas as pd

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

with open(METADATA_DIR / "esg_terms.txt", "r", encoding="utf-8") as f:
    ESG_KEYWORDS = [
        line.strip().lower()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

with open(METADATA_DIR / 'personal_stopwords.txt', 'r', encoding='utf-8') as f:
    PERSONAL_SW = f.read().split("\n")

# LDA STOPWORDS — adicionales, solo para topic modeling (no afectan ESGSI)
with open(METADATA_DIR / 'lda_stopwords.txt', 'r', encoding='utf-8') as f:
    LDA_STOPWORDS = set(
        line.strip().lower()
        for line in f
        if line.strip() and not line.startswith("#")
    )

# HEDGE KEYWORDS — categorías L&M indicativas de lenguaje impreciso/especulativo
_lm_raw = pd.read_csv(METADATA_DIR / "RAW_LM_dictionary.csv")
HEDGE_KEYWORDS = set(
    _lm_raw[_lm_raw['sentiment'].isin(['Uncertainty', 'WeakModal', 'StrongModal', 'Constraining'])]
    ['word'].str.lower().tolist()
)

# QUANT SCORE — patrones RegEx para contenido cuantificable y marcos regulatorios
QUANT_PATTERNS = {
    # Porcentajes (ej. "42%", "3.5 %")
    "percentages":     r'\b\d+(?:[.,]\d+)?\s*%',
    # Cifras métricas grandes: 4+ dígitos (excluye años si se quiere, pero se normaliza)
    "large_numbers":   r'\b\d{4,}\b',
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
