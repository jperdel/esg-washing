import re
import json
from pathlib import Path
from typing import Iterable

from loguru import logger

import spacy

#: Frases del vocabulario que el preprocesado destruiría (sus palabras son
#: stopwords o se quedan en una sola genérica: "well-being" -> "", "iso 14001"
#: -> "iso"). Las genera scripts/build_lexicons.py; aquí solo se leen.
PROTECTED_PHRASES_FILE = Path(__file__).resolve().parent.parent / "metadata" / "esg_protected_phrases.txt"


def load_protected_phrases(path: Path = PROTECTED_PHRASES_FILE) -> list[tuple[str, str]]:
    """Lee pares (regex, token) separados por tabulador; lista vacía si no existe."""
    if not path.exists():
        return []
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            regex, token = line.split("\t")
            pairs.append((regex, token))
    return pairs


class TextProcessor:

    def __init__(self, extra_sw:list=[], spacy_model:str="en_core_web_md",
                 protected_phrases: Iterable[tuple[str, str]] | None = None):
        
        model_name = spacy_model
        try:
            self.nlp = spacy.load(model_name, disable=["ner", "parser"])
            self.nlp.max_length = 5000000
        except OSError:
            logger.warning(f"Modelo {model_name} no encontrado. Descargando...")
            from spacy.cli import download
            download(model_name)
            self.nlp = spacy.load(model_name, disable=["ner", "parser"])
            logger.success(f"Modelo {model_name} descargado e instalado.")

        self.custom_stopwords = extra_sw

        # Protección de frases: se sustituyen por un único token ANTES de
        # tokenizar, para que sobrevivan al filtro de stopwords. Por defecto se
        # leen de metadata/ (así todos los scripts preprocesan igual); una lista
        # vacía la desactiva, que es lo que necesita build_lexicons para
        # detectar qué frases colapsan.
        pairs = load_protected_phrases() if protected_phrases is None else list(protected_phrases)
        self.protected = [(re.compile(rf"\b(?:{rx})\b"), tok) for rx, tok in pairs]
        logger.debug(f"TextProcessor con spaCy inicializado ({len(self.protected)} frases protegidas).")

    # def extract_from_pdf(self, pdf_path: Path) -> str:
    #     """Extrae texto bruto de un PDF."""
    #     try:
    #         with fitz.open(pdf_path) as doc:
    #             text = "".join([page.get_text() for page in doc])
    #         return text
    #     except Exception as e:
    #         logger.error(f"Error extrayendo {pdf_path.name}: {e}")
    #         return ""
        
    def extract_from_json(self, json_path: Path) -> str:
        """Extrae texto bruto de un PDF."""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                result = json.load(f)
                relevant_text = result['relevant_text']
            return relevant_text
        except Exception as e:
            logger.error(f"Error extrayendo {json_path.name}: {e}")
            return ""

    def protect(self, text: str) -> str:
        """Sustituye las frases protegidas por su token (el texto ya en minúsculas)."""
        for rx, tok in self.protected:
            text = rx.sub(f" {tok} ", text)
        return text

    def lemmas_from_doc(self, doc) -> list[str]:
        """Filtro de tokens del preprocesado, aplicado a un Doc de spaCy."""
        return [
            token.lemma_ for token in doc
            if (not token.is_stop and
                not token.is_punct and
                not token.is_space and
                self._is_content_token(token.text) and
                len(token.text) > 2 and
                token.lemma_ not in self.custom_stopwords)
        ]

    def prepare(self, text: str) -> str:
        """URLs fuera, minúsculas y frases protegidas: la entrada de spaCy."""
        url_pattern = r'https?://\S+|www\.\S+'
        return self.protect(re.sub(url_pattern, '', text).lower())

    def preprocess(self, text: str) -> str:
        if not text: return ""
        return " ".join(self.lemmas_from_doc(self.nlp(self.prepare(text))))

    @staticmethod
    def _is_content_token(text: str) -> bool:
        """
        Acepta tokens alfanuméricos con al menos dos letras.

        Sustituye al antiguo filtro `token.is_alpha`, que descartaba términos
        ESG imprescindibles por contener un dígito: "co2", "co2e", "sf6",
        "ftse4good", "iso14001". Al exigir dos letras seguimos descartando
        cifras puras ("2023", "14001") y referencias de página ("p12", "q1"),
        que solo aportarían ruido al TF-IDF y al LDA.
        """
        return text.isalnum() and sum(c.isalpha() for c in text) >= 2