import re
import fitz  # PyMuPDF
import nltk
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from loguru import logger

class TextProcessor:
    def __init__(self, language='english', extra_sw=[]):
        # Descarga de recursos una sola vez al instanciar
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words(language) + extra_sw)
        logger.debug("TextProcessor inicializado.")

    def extract_from_pdf(self, pdf_path: Path) -> str:
        """Extrae texto bruto de un PDF."""
        try:
            with fitz.open(pdf_path) as doc:
                text = "".join([page.get_text() for page in doc])
            return text
        except Exception as e:
            logger.error(f"Error extrayendo {pdf_path.name}: {e}")
            return ""

    def preprocess(self, text: str) -> str:
        """Limpia, normaliza y lematiza el texto."""
        if not text: return ""
        
        # 1. Limpieza básica (URLs y minúsculas)
        text = re.sub(r'http\S+', '', text.lower())
        # 2. Eliminar todo lo que no sean letras
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # 3. Tokenización y Lematización eficiente
        tokens = [
            self.lemmatizer.lemmatize(word) 
            for word in text.split() 
            if word not in self.stop_words and len(word) > 2
        ]
        return " ".join(tokens)