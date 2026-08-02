from __future__ import annotations
import re
import numpy as np
from typing import List
import pysentiment2 as ps
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from loguru import logger


#: Especificaciones disponibles para el componente de sustancia (SUS).
#: Se conservan todas para poder reportar la tabla de robustez del paper.
SUS_MODES = ("density", "tfidf_length", "lagasio")


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

    ------------------------------------------------------------------
    El componente SUS: tres especificaciones
    ------------------------------------------------------------------
    El TF-IDF es una representación pensada para RECUPERACIÓN DE INFORMACIÓN,
    no un instrumento de medida. Sus dos ingredientes están diseñados para
    buscar documentos y ambos son contraproducentes al cuantificar cuánta
    divulgación ESG contiene un informe:

      · el IDF premia los términos raros, cuando en un corpus ESG la rareza
        es inversamente proporcional a la centralidad temática;
      · la normalización L2 hace la representación invariante a la longitud
        del vector, que es exactamente la magnitud que queremos medir.

    Verificado sobre el corpus de 344 informes: bajo 'lagasio' el SUS
    correlaciona 0,97 con la amplitud de vocabulario y solo 0,41 con la
    densidad real de términos ESG. Un informe al que se le añade relleno sin
    contenido ESG mantiene su SUS intacto mientras su densidad cae a un sexto.

    Modos disponibles:
      · 'density'      (por defecto) — menciones ESG por cada 100 palabras.
                       Independiente del corpus y comparable entre estudios.
      · 'tfidf_length' — ponderado por IDF y normalizado por longitud. Es el
                       TF-IDF de Lagasio corregido; correlaciona 0,99 con
                       'density', luego sirve de robustez a la ponderación.
      · 'lagasio'      — TF-IDF con normalización L2 y media sobre el
                       vocabulario, tal como se replicó del paper original.
                       Se conserva para la comparación metodológica.
    """

    def __init__(
        self,
        keywords: List[str],
        hedge_words: set[str],
        quant_patterns: dict[str, str],
        ext_weights: dict[str, float] | None = None,
        sus_mode: str = "density",
    ):
        if sus_mode not in SUS_MODES:
            raise ValueError(f"sus_mode debe ser uno de {SUS_MODES}, recibido '{sus_mode}'.")

        self.keywords = keywords
        self.hedge_words = hedge_words
        self.quant_patterns = {k: re.compile(v, re.IGNORECASE) for k, v in quant_patterns.items()}
        self.ext_weights = ext_weights or {"w_quant": 0.5, "w_hedge": 0.5}
        self.sus_mode = sus_mode

        # El vocabulario contiene expresiones multipalabra ya lematizadas
        # ("human right", "circular economy"). Con el ngram_range por defecto
        # (1,1) el analizador solo generaría unigramas y esas entradas
        # puntuarían siempre cero, sin lanzar ningún error.
        max_n = max((len(k.split()) for k in self.keywords), default=1)
        self.counter = CountVectorizer(vocabulary=self.keywords, ngram_range=(1, max_n))
        self.vectorizer = TfidfVectorizer(vocabulary=self.keywords, ngram_range=(1, max_n))

        self._cache: dict[str, np.ndarray] = {}

        logger.debug(
            f"ESGSIAnalyzer listo — {len(self.keywords)} keywords ESG "
            f"(n-grama máximo {max_n}), sus_mode='{sus_mode}', "
            f"{len(self.hedge_words)} hedge words, "
            f"{len(self.quant_patterns)} patrones QUANT."
        )

    # ------------------------------------------------------------------
    # Utilidades internas
    # ------------------------------------------------------------------

    def _counts(self, texts: List[str]) -> np.ndarray:
        """Matriz documento × término con los conteos brutos del vocabulario."""
        if "counts" not in self._cache:
            self._cache["counts"] = self.counter.fit_transform(texts).toarray()
        return self._cache["counts"]

    def _idf(self, texts: List[str]) -> np.ndarray:
        """Pesos IDF estimados sobre el corpus (dependientes de él, por diseño)."""
        if "idf" not in self._cache:
            self.vectorizer.fit(texts)
            self._cache["idf"] = self.vectorizer.idf_
        return self._cache["idf"]

    @staticmethod
    def _lengths(texts: List[str]) -> np.ndarray:
        return np.array([max(len(t.split()), 1) for t in texts], dtype=float)

    def reset_cache(self):
        """Limpia los cachés; obligatorio si se cambia de corpus."""
        self._cache.clear()

    # ------------------------------------------------------------------
    # Scores individuales
    # ------------------------------------------------------------------

    def calculate_sus_scores(self, texts: List[str], mode: str | None = None) -> np.ndarray:
        """
        Componente de sustancia. Ver la docstring de la clase para el porqué
        de cada especificación.
        """
        mode = mode or self.sus_mode
        if mode not in SUS_MODES:
            raise ValueError(f"sus_mode debe ser uno de {SUS_MODES}, recibido '{mode}'.")

        logger.info(f"Calculando SUS scores (modo '{mode}')...")

        if mode == "lagasio":
            # TF-IDF con normalización L2 (por defecto en sklearn) y media
            # sobre el vocabulario, tal cual se replicó del paper original.
            return self.vectorizer.fit_transform(texts).toarray().mean(axis=1)

        counts = self._counts(texts)
        n_tokens = self._lengths(texts)

        if mode == "density":
            return counts.sum(axis=1) / n_tokens * 100

        # mode == "tfidf_length"
        return (counts * self._idf(texts)).sum(axis=1) / n_tokens * 100

    def calculate_sus_variants(self, texts: List[str]) -> dict[str, np.ndarray]:
        """Las tres especificaciones a la vez, para la tabla de robustez."""
        return {mode: self.calculate_sus_scores(texts, mode=mode) for mode in SUS_MODES}

    def calculate_breadth_scores(self, texts: List[str]) -> np.ndarray:
        """
        Amplitud temática: número EFECTIVO de términos ESG distintos, medido
        como la exponencial de la entropía de Shannon del reparto de menciones.

        No forma parte del índice: es la dimensión que el SUS de Lagasio
        capturaba sin pretenderlo, y se reporta por separado porque describe
        algo real (cuántos temas toca el informe) que no es sustancia.
        """
        logger.info("Calculando amplitud temática (nº efectivo de términos)...")
        counts = self._counts(texts).astype(float)
        totals = np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
        p = counts / totals
        with np.errstate(divide="ignore", invalid="ignore"):
            entropy = -np.where(p > 0, p * np.log(p), 0.0).sum(axis=1)
        return np.exp(entropy)

    def calculate_sen_scores(self, texts: List[str]) -> np.ndarray:
        """Polaridad de sentimiento via diccionario Loughran-McDonald."""
        lm = ps.LM()
        logger.info("Calculando SEN scores (Loughran-McDonald)...")
        tokenized = [lm.tokenize(t) for t in texts]
        return np.array([lm.get_score(t)["Polarity"] for t in tokenized])

    def calculate_quant_scores(self, raw_texts: List[str]) -> np.ndarray:
        """
        Densidad de contenido cuantificable: suma de ocurrencias de los patrones
        QUANT (porcentajes, cifras, unidades físicas, marcos regulatorios),
        normalizada por el número de tokens del documento para hacerla
        comparable entre textos de distinta longitud.

        ATENCIÓN: recibe el texto CRUDO extraído del PDF, no el preprocesado.
        El preprocesado descarta los tokens numéricos, así que sobre él los
        patrones 'percentages' y 'large_numbers' devuelven siempre cero.
        """
        logger.info("Calculando QUANT scores (RegEx sobre cifras y marcos regulatorios)...")
        scores = []
        for text in raw_texts:
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
