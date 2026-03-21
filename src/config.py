from pathlib import Path

# PATHS
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PDF_DATA_DIR = DATA_DIR / "pdf"
RESULTS_DIR = BASE_DIR / "results"
METRICS_RESULTS_DIR = RESULTS_DIR / "metrics"
LDA_RESULTS_DIR = RESULTS_DIR / "lda"

# LDA PARAMETERS
K_TOPICS = 10

# NLP PARAMETERS
SPACY_MODEL = 'en_core_web_md'
ESG_KEYWORDS = [
    'environment', 'social', 'governance', 'sustainability', 
    'carbon', 'emissions', 'diversity', 'inclusion', 'waste',
    'renewable', 'ethics', 'transparency', 'human rights'
]

PERSONAL_SW = [
    # --- NOMBRES DE EMPRESAS Y ENTIDADES ---
    'abengoa', 'accenture', 'acerinox', 'acs', 'adidas', 'aena', 'airbus', 
    'almirall', 'amadeus', 'apple', 'arcelormittal', 'avangrid', 'bankia', 
    'bankinter', 'bbva', 'caixabank', 'deloitte', 'goldman', 'iberdrola', 
    'inditex', 'kpmg', 'naturgy', 'pwc', 'repsol', 'sachs', 'santander', 
    'telefonica', 'abn', 'wwwiberdrolacom', 'iberdrola', 'santanders',

    # --- NOMBRES PROPIOS Y APELLIDOS ---
    'aboukhair', 'alvarez', 'botin', 'galan', 'huerta', 'ignacio', 'javier', 
    'jose', 'manuel', 'palao', 'sanchez', 'vincent', 'violeta', 'vallecana', 
    'verona',

    # --- TOKENS DE LIMPIEZA / BASURA ---
    'aaa', 'ab', 'abif', 'abs', 'adj', 'af', 'ag', 'ah', 'ai', 'aj', 'al', 
    'am', 'as', 'at', 'au', 'av', 'tran', 'tttt', 'vppa', 'webapp', 
    'websitewwwcnmves', 'traetubolsa',

    # --- CONECTORES Y VERBOS CORPORATIVOS VACÍOS ---
    'abovementioned', 'abroad', 'academic', 'academy', 'accessed', 'according', 
    'accordingly', 'achieve', 'achieved', 'across', 'addition', 'additional', 
    'additionally', 'address', 'abide', 'abrupt', 'absence', 'absent', 
    'absorb', 'absorbing', 'absorbs', 'absorption', 'accept', 'acceptance', 
    'accepting', 'accepts', 'unaffected', 'undetectable', 'unfold', 'vivid', 'banking',
    
    # TÉRMINOS BANCARIOS
    'banco', 'hedge', 'grupo'
]