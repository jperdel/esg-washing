from pathlib import Path

# PATHS
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PDF_DATA_DIR = DATA_DIR / "pdf"
CHUNKED_DATA_DIR = DATA_DIR / "chunks"
FILTERED_DATA_DIR = DATA_DIR / "filtered"
CLEAN_DATA_DIR = DATA_DIR / "clean"
RESULTS_DIR = BASE_DIR / "results"
METRICS_RESULTS_DIR = RESULTS_DIR / "metrics"
LDA_RESULTS_DIR = RESULTS_DIR / "lda"

# LDA PARAMETERS
K_TOPICS_LIST = [10, 15, 20, 25]
ALPHA_LIST = ['asymmetric']
K_ITERS = 5

# FILTERING DOCUMENTS PARAMETERS W/ VERTEXAI
LOCATION = "us-central1"
EMBEDDING_MODEL = "text-embedding-004"
CHUNK_SIZE = 1200
OVERLAP = 300
ANCHOR_QUERIES = [
    # Vector faro para E
    "Climate change mitigation and adaptation strategies. Reduction of greenhouse gas emissions, including Scope 1, Scope 2, and Scope 3 data. Decarbonization roadmaps, net-zero targets, and carbon neutrality commitments. Energy efficiency, transition to renewable energy sources like wind or solar. Water stewardship, waste management, circular economy principles, and biodiversity protection. Environmental impact of supply chain and manufacturing processes.",
    # Vector faro para S
    "Human rights due diligence and labor standards. Diversity, equity, and inclusion (DEI) policies, gender pay gap reporting, and representation of minorities. Occupational health and safety (OHS) protocols and employee well-being programs. Talent attraction, retention, and professional development. Corporate social responsibility (CSR), community engagement, and philanthropic initiatives. Social impact of products and services on customers and local communities.",
    # Vector faro para G
    "Corporate governance framework, board of directors composition, independence, and diversity. Executive compensation linked to ESG performance and sustainability targets. Anti-corruption policies, whistleblowing mechanisms, and business ethics. Risk management systems, internal audits, and compliance with non-financial reporting directives. Shareholder rights, transparency in tax strategy, and data privacy security.",
    # Vector faro para "commitment"
    "Commitment to international standards such as GRI, SASB, TCFD, or UN Sustainable Development Goals (SDGs). Forward-looking statements regarding sustainability ambitions, long-term visions, and strategic ESG integration. Stakeholder engagement processes and materiality assessment results."
]
SIMILARITY_THRESHOLD = 0.6
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