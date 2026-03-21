import re
import fitz  # PyMuPDF
from pathlib import Path

from loguru import logger

import spacy

# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer



class TextProcessor:
    # def __init__(self, language='english', extra_sw=[]):
    #     # Descarga de recursos una sola vez al instanciar
    #     nltk.download('punkt', quiet=True)
    #     nltk.download('stopwords', quiet=True)
    #     nltk.download('wordnet', quiet=True)
        
    #     self.lemmatizer = WordNetLemmatizer()
    #     self.stop_words = set(stopwords.words(language) + extra_sw)
    #     logger.debug("TextProcessor inicializado.")

    def __init__(self, extra_sw:list=[], spacy_model:str="en_core_web_md"):
        
        model_name = spacy_model
        try:
            # Intentar cargar el modelo
            self.nlp = spacy.load(model_name, disable=["ner", "parser"])
            self.nlp.max_length = 5000000 # Aumentar la RAM para informes tan largos
        except OSError:
            logger.warning(f"Modelo {model_name} no encontrado. Descargando...")
            # Comando para descargar el modelo desde el script
            from spacy.cli import download
            download(model_name)
            self.nlp = spacy.load(model_name, disable=["ner", "parser"])
            logger.success(f"Modelo {model_name} descargado e instalado.")

        # Añadimos tus Custom Stopwords
        self.custom_stopwords = extra_sw 
        logger.debug("TextProcessor con spaCy inicializado.")

    def extract_from_pdf(self, pdf_path: Path) -> str:
        """Extrae texto bruto de un PDF."""
        try:
            with fitz.open(pdf_path) as doc:
                text = "".join([page.get_text() for page in doc])
            return text
        except Exception as e:
            logger.error(f"Error extrayendo {pdf_path.name}: {e}")
            return ""

    # def preprocess(self, text: str) -> str:
    #     """Limpia, normaliza y lematiza el texto."""
    #     if not text: return ""
        
    #     # 1. Limpieza básica (URLs y minúsculas)
    #     text = re.sub(r'http\S+', '', text.lower())
    #     # 2. Eliminar todo lo que no sean letras
    #     text = re.sub(r'[^a-zA-Z\s]', '', text)
        
    #     # 3. Tokenización y Lematización eficiente
    #     tokens = [
    #         self.lemmatizer.lemmatize(word) 
    #         for word in text.split() 
    #         if word not in self.stop_words and len(word) > 2
    #     ]
    #     return " ".join(tokens)

    def preprocess(self, text: str) -> str:
        if not text: return ""
        
        # 1. Limpieza de URLs antes de pasar a minúsculas y spaCy
        # Esta regex cubre http, https, ftp y URLs que empiezan por www.
        url_pattern = r'https?://\S+|www\.\S+'
        text = re.sub(url_pattern, '', text)
        
        # 2. Normalización básica
        text = text.lower()
        
        # 3. Procesamiento con spaCy
        # Recordatorio: usa disable=["ner", "parser"] en el init para mayor velocidad
        doc = self.nlp(text)
        
        tokens = []
        for token in doc:
            # Filtros de spaCy:
            # is_stop: Usa la lista base de spaCy (muy completa)
            # is_alpha: Elimina números y restos de símbolos/puntuación
            if (not token.is_stop and 
                not token.is_punct and 
                not token.is_space and
                token.is_alpha and
                len(token.text) > 2 and 
                token.lemma_ not in self.custom_stopwords):
                
                # Guardamos el lema (ej: "banking" -> "bank", "recognized" -> "recognize")
                tokens.append(token.lemma_)
        
        return " ".join(tokens)