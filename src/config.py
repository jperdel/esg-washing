from pathlib import Path
import pandas as pd

# PATHS
BASE_DIR = Path(__file__).resolve().parent.parent

METADATA_DIR = BASE_DIR / "metadata"

DATA_DIR = BASE_DIR / "data"
PDF_DATA_DIR = DATA_DIR / "pdf"
CHUNKED_DATA_DIR = DATA_DIR / "chunks"
FILTERED_DATA_DIR = DATA_DIR / "filtered"
CLEAN_DATA_DIR = DATA_DIR / "clean"

RESULTS_DIR = BASE_DIR / "results"
METRICS_RESULTS_DIR = RESULTS_DIR / "metrics"
LDA_RESULTS_DIR = RESULTS_DIR / "lda"

# LDA PARAMETERS
K_TOPICS_LIST = range(5, 11, 5)
ALPHA_LIST = ['auto'] # [0.01, 0.1, 1]
K_ITERS = 2

# FILTERING DOCUMENTS PARAMETERS W/ VERTEXAI
LOCATION = "us-central1"
EMBEDDING_MODEL = "text-embedding-004"
CHUNK_SIZE = 1200
OVERLAP = 300
SIMILARITY_THRESHOLD = 0.55

with open(METADATA_DIR / "anchor_queries_v2.txt", "r", encoding="utf-8") as f:
    ANCHOR_QUERIES = f.read().split("\n")

# NLP PARAMETERS
SPACY_MODEL = 'en_core_web_md'
ESG_KEYWORDS = pd.read_csv(METADATA_DIR / "LM_dictionary.csv")['word'].drop_duplicates()

with open(METADATA_DIR / 'personal_stopwords.txt', 'r', encoding='utf-8') as f:
    PERSONAL_SW = f.read().split("\n")