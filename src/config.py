from pathlib import Path

# PATHS
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PDF_DATA_DIR = DATA_DIR / "pdf"
MD_DATA_DIR = DATA_DIR / "markdown"
SAMPLE_DATA_DIR = PDF_DATA_DIR / "samples"
RESULTS_DIR = BASE_DIR / "results"

# NLP PARAMS
KEYWORDS = [
    'environment', 'social', 'governance', 'sustainability', 
    'carbon', 'emissions', 'diversity', 'inclusion', 'waste',
    'renewable', 'ethics', 'transparency', 'human rights'
]