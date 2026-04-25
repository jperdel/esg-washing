from __future__ import annotations
import re
import numpy as np
from typing import List
import pysentiment2 as ps
from sklearn.feature_extraction.text import TfidfVectorizer
from loguru import logger


class ESGSIAnalyzer:
    """
    Calcula el ESG-washing Severity Index (ESGSI) y su versión extendida (ESGSI_ext).

    Índice original (Lagasio 2024):
        ESGSI = Z(SEN) - Z(SUS)

    Índice extendido (esta implementación):
        ESGSI_ext = Z(SEN) - Z(SUS) - w_quant·Z(QUANT) + w_hedge·Z(HEDGE)

        · QUANT: densidad de contenido cuantificable (cifras, marcos regulatorios…)
          Un informe rico en datos duros es menos susceptible de washing → se resta.
        · HEDGE: densidad de lenguaje impreciso / especulativo (diccionario L&M)
          Un informe lleno de evasivas eleva el riesgo de washing → se suma.
    """

    def __init__(
        self,
        keywords: List[str],
        hedge_words: set[str],
        quant_patterns: dict[str, str],
        ext_weights: dict[str, float] | None = None,
    ):
        self.keywords = keywords
        self.hedge_words = hedge_words
        self.quant_patterns = {k: re.compile(v, re.IGNORECASE) for k, v in quant_patterns.items()}
        self.ext_weights = ext_weights or {"w_quant": 0.5, "w_hedge": 0.5}

        self.vectorizer = TfidfVectorizer(vocabulary=self.keywords, binary=False)
        logger.debug(
            f"ESGSIAnalyzer listo — {len(self.keywords)} keywords ESG, "
            f"{len(self.hedge_words)} hedge words, "
            f"{len(self.quant_patterns)} patrones QUANT."
        )

    # ------------------------------------------------------------------
    # Scores individuales
    # ------------------------------------------------------------------

    def calculate_sus_scores(self, texts: List[str]) -> np.ndarray:
        """Densidad de términos ESG via TF-IDF (media por documento)."""
        logger.info("Calculando SUS scores (TF-IDF sobre keywords ESG)...")
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        return tfidf_matrix.toarray().mean(axis=1)

    def calculate_sen_scores(self, texts: List[str]) -> np.ndarray:
        """Polaridad de sentimiento via diccionario Loughran-McDonald."""
        lm = ps.LM()
        logger.info("Calculando SEN scores (Loughran-McDonald)...")
        tokenized = [lm.tokenize(t) for t in texts]
        return np.array([lm.get_score(t)["Polarity"] for t in tokenized])

    def calculate_quant_scores(self, texts: List[str]) -> np.ndarray:
        """
        Densidad de contenido cuantificable: suma de ocurrencias de los patrones
        QUANT (porcentajes, cifras, unidades físicas, marcos regulatorios),
        normalizada por el número de tokens del documento para hacerla
        comparable entre textos de distinta longitud.
        """
        logger.info("Calculando QUANT scores (RegEx sobre cifras y marcos regulatorios)...")
        scores = []
        for text in texts:
            n_tokens = max(len(text.split()), 1)
            hits = sum(len(pat.findall(text)) for pat in self.quant_patterns.values())
            scores.append(hits / n_tokens)
        return np.array(scores)

    def calculate_hedge_scores(self, texts: List[str]) -> np.ndarray:
        """
        Densidad de lenguaje especulativo/impreciso: proporción de tokens del
        documento que pertenecen a las categorías Uncertainty, WeakModal,
        StrongModal y Constraining del diccionario L&M.
        El texto de entrada debe estar ya lematizado (pipeline TextProcessor).
        """
        logger.info("Calculando HEDGE scores (palabras L&M de incertidumbre)...")
        scores = []
        for text in texts:
            tokens = text.split()
            n_tokens = max(len(tokens), 1)
            hedge_count = sum(1 for t in tokens if t in self.hedge_words)
            scores.append(hedge_count / n_tokens)
        return np.array(scores)

    # ------------------------------------------------------------------
    # Normalización
    # ------------------------------------------------------------------

    def _z_score(self, data: np.ndarray) -> np.ndarray:
        """Z-score estándar; devuelve ceros si la desviación típica es 0."""
        std = np.std(data)
        if std == 0:
            return np.zeros_like(data, dtype=float)
        return (data - np.mean(data)) / std

    # ------------------------------------------------------------------
    # Índices compuestos
    # ------------------------------------------------------------------

    def compute_index(
        self,
        sus_scores: np.ndarray,
        sen_scores: np.ndarray,
    ) -> np.ndarray:
        """ESGSI original: Z(SEN) - Z(SUS)."""
        logger.info("Calculando ESGSI original...")
        return self._z_score(sen_scores) - self._z_score(sus_scores)

    def compute_extended_index(
        self,
        sus_scores: np.ndarray,
        sen_scores: np.ndarray,
        quant_scores: np.ndarray,
        hedge_scores: np.ndarray,
    ) -> np.ndarray:
        """
        ESGSI extendido:
            Z(SEN) - Z(SUS) - w_quant·Z(QUANT) + w_hedge·Z(HEDGE)
        """
        logger.info("Calculando ESGSI extendido...")
        w_q = self.ext_weights["w_quant"]
        w_h = self.ext_weights["w_hedge"]
        return (
            self._z_score(sen_scores)
            - self._z_score(sus_scores)
            - w_q * self._z_score(quant_scores)
            + w_h * self._z_score(hedge_scores)
        )
