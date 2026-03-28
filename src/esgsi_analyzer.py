import numpy as np
from typing import List
# from textblob import TextBlob
import pysentiment2 as ps
from sklearn.feature_extraction.text import TfidfVectorizer
from loguru import logger

class ESGSIAnalyzer:
    def __init__(self, keywords: List[str]):
        self.keywords = keywords
        # Inicializamos el vectorizador con el vocabulario fijo de ESG
        self.vectorizer = TfidfVectorizer(vocabulary=self.keywords, binary=False)
        logger.debug(f"Pipeline ESGSI listo con {len(self.keywords)} palabras clave.")

    def calculate_sus_scores(self, texts: List[str]) -> np.ndarray:
        """Calcula la puntuación de sostenibilidad (SUS) mediante TF-IDF."""
        logger.info("Calculando métricas SUS (TF-IDF)...")
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        return tfidf_matrix.toarray().sum(axis=1)

    def calculate_sen_scores(self, texts: List[str]) -> List[float]:
        """Calcula el sentimiento (SEN) usando Pysentiment2."""
        lm = ps.LM()

        logger.info("Calculando métricas de Sentimiento...")
        tokenized_texts = [lm.tokenize(text) for text in texts]
        scores = [lm.get_score(text)['Polarity'] for text in tokenized_texts]

        return scores
        
        # OLD VERSION: ORIGINAL LAGASIO PAPER
        # """Calcula el sentimiento (SEN) usando TextBlob."""
        # logger.info("Calculando métricas de Sentimiento...")
        # return [TextBlob(txt).sentiment.polarity for txt in texts]

    def _z_score_normalization(self, data: np.ndarray) -> np.ndarray:
        """Aplica la normalización Z necesaria para el índice de Lagasio."""
        std = np.std(data)
        if std == 0: return np.zeros_like(data)
        return (data - np.mean(data)) / std

    def compute_index(self, sus_scores: np.ndarray, sen_scores: List[float]) -> np.ndarray:
        """Calcula el ESGSI final: Z(SEN) - Z(SUS)."""
        logger.info("Generando índice ESGSI...")
        norm_sen = self._z_score_normalization(np.array(sen_scores))
        norm_sus = self._z_score_normalization(sus_scores)
        return norm_sen - norm_sus