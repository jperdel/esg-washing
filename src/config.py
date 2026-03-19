from pathlib import Path

# PATHS
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PDF_DATA_DIR = DATA_DIR / "pdf"
RESULTS_DIR = BASE_DIR / "results"
# MD_DATA_DIR = DATA_DIR / "markdown" # UNUSED

# NLP PARAMS
KEYWORDS = [
    'environment', 'social', 'governance', 'sustainability', 
    'carbon', 'emissions', 'diversity', 'inclusion', 'waste',
    'renewable', 'ethics', 'transparency', 'human rights'
]