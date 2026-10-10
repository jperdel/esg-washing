from __future__ import annotations
import re
import numpy as np
from pathlib import Path
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from loguru import logger


#: Especificaciones disponibles para el componente de sustancia (SUS).
#: Se conservan todas para poder reportar la tabla de robustez del paper.
SUS_MODES = ("density", "tfidf_length", "lagasio")

_METADATA_DIR = Path(__file__).resolve().parent.parent / "metadata"


def _read_lexicon(name: str) -> set[str]:
    with open(_METADATA_DIR / name, encoding="utf-8") as fh:
        return {line.strip() for line in fh if line.strip() and not line.startswith("#")}


#: Palabras del texto crudo: letras, con guiones internos ("ill-defined").
WORD_RE = re.compile(r"[a-z]+(?:-[a-z]+)*")

#: Negaciones de Loughran y McDonald (2011): una palabra positiva precedida de
#: una de ellas a tres palabras o menos no cuenta como positiva.
NEGATIONS = frozenset({"no", "not", "none", "neither", "never", "nobody"})

#: "May" como mes, no como modal: en mayúscula y junto a un día o un año
#: ("15 May", "May 2023", "May 15", "in May"). Se busca en el texto sin
#: pasar a minúsculas y se descuenta de los aciertos de "may".
MAY_MONTH_RE = re.compile(
    r"\b\d{1,2}(?:st|nd|rd|th)?\s+May\b"
    r"|\bMay\s+\d{1,4}\b"
    r"|\bMay\b(?=\s*,?\s*(?:19|20)\d\d)"
    r"|\b(?:in|of|end|since|until|from|early|late|mid|by|on)\s+May\b"
)

#: QUANT, cifras grandes: lo que NO es una magnitud aunque tenga 4+ dígitos.
#: Rangos de años pegados ("20212023": la extracción quita el guion), códigos
#: de norma (ISO 14001), números de actos legales (2020/852, L. 123) y años
#: lejanos que el patrón no filtra (2100).
_YEAR_RANGE_RE = re.compile(r"^(?:19|20)\d{2}(?:19|20)\d{2}$")
_CODE_PREFIX_RE = re.compile(r"(?:ISO|OHSAS|FSSC|SA|IATF|EN|IEC|EMAS|BS|NF)\s?[-:]?\s?$", re.IGNORECASE)
_LEGAL_PREFIX_RE = re.compile(r"(?:\bL\.|\bR\.|\bArt(?:icle|\.)?|\bNo\.?|\bn[o°]\.?)\s?$", re.IGNORECASE)


def _is_magnitude(text: str, m: re.Match) -> bool:
    """True si una cifra de 4+ dígitos es una magnitud y no un código o un año."""
    s = m.group(0)
    before = text[max(0, m.start() - 12):m.start()]
    after = text[m.end():m.end() + 1]
    if _YEAR_RANGE_RE.match(s):
        return False
    if _CODE_PREFIX_RE.search(before) or after == ":":
        return False
    if text[max(0, m.start() - 1):m.start()] == "/" or after == "/" or _LEGAL_PREFIX_RE.search(before) \
            or (after == "-" and text[max(0, m.start() - 3):m.start()].strip().endswith(".")):
        return False
    if s in {"2100", "3000"} or re.fullmatch(r"(?:19|20)\d{2}", s):
        return False
    return True


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
        positive_words: set[str] | None = None,
        negative_words: set[str] | None = None,
    ):
        if sus_mode not in SUS_MODES:
            raise ValueError(f"sus_mode debe ser uno de {SUS_MODES}, recibido '{sus_mode}'.")

        self.keywords = keywords
        self.hedge_words = hedge_words
        # Listas Positive/Negative de L&M en sus formas originales (las genera
        # scripts/build_lexicons.py). Si no se pasan, se leen de metadata/.
        self.positive_words = positive_words if positive_words is not None else _read_lexicon("lm_positive.txt")
        self.negative_words = negative_words if negative_words is not None else _read_lexicon("lm_negative.txt")
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

    # -- Recuentos por texto (los usan el pipeline y la simulación de re-extracción)

    def sen_counts(self, raw_text: str) -> tuple[int, int]:
        """
        Palabras positivas y negativas de L&M en un texto crudo.

        Se cruzan las FORMAS de las listas de L&M con las palabras del texto en
        minúsculas, sin lematizar ni reducir a raíz: así es como está construido
        el diccionario (lista de formas flexionadas). Reducir ambos lados a su
        raíz de Porter, como hace pysentiment2, mete palabras que no están en
        L&M ("objectives" -> "object", negativa; "information" -> "inform",
        positiva). Una positiva precedida de una negación a tres palabras o menos
        no cuenta (Loughran y McDonald, 2011).
        """
        tokens = WORD_RE.findall(raw_text.lower())
        pos = neg = 0
        for i, t in enumerate(tokens):
            if t in self.positive_words:
                if not any(w in NEGATIONS for w in tokens[max(0, i - 3):i]):
                    pos += 1
            elif t in self.negative_words:
                neg += 1
        return pos, neg

    def quant_hits(self, raw_text: str) -> int:
        """Aciertos QUANT en un texto crudo (cifras grandes solo si son magnitudes)."""
        hits = 0
        for name, pat in self.quant_patterns.items():
            if name == "large_numbers":
                hits += sum(1 for m in pat.finditer(raw_text) if _is_magnitude(raw_text, m))
            else:
                hits += sum(1 for _ in pat.finditer(raw_text))
        return hits

    def hedge_counts(self, raw_text: str) -> tuple[int, int]:
        """Aciertos HEDGE y palabras de un texto crudo ("May" como mes no cuenta)."""
        tokens = WORD_RE.findall(raw_text.lower())
        hits = sum(1 for t in tokens if t in self.hedge_words)
        if "may" in self.hedge_words:
            hits -= len(MAY_MONTH_RE.findall(raw_text))
        return max(hits, 0), len(tokens)

    # -- Scores por documento

    def calculate_sen_scores(self, raw_texts: List[str]) -> np.ndarray:
        """
        Polaridad de Loughran-McDonald, (P - N) / (P + N), sobre el texto CRUDO
        (ver sen_counts). Un texto sin palabras de tono puntúa 0.
        """
        logger.info("Calculando SEN scores (Loughran-McDonald, formas exactas)...")
        scores = []
        for text in raw_texts:
            pos, neg = self.sen_counts(text)
            scores.append((pos - neg) / (pos + neg) if pos + neg else 0.0)
        return np.array(scores)

    def calculate_quant_scores(self, raw_texts: List[str]) -> np.ndarray:
        """
        Densidad de contenido cuantificable: porcentajes, cifras grandes que son
        magnitudes y unidades físicas, por token del documento.

        No cuenta nombres de marcos (GRI, CSRD, "taxonomy"), que no son cifras y
        crecen con los mandatos de reporte; ni CO2/GHG como palabras sueltas, que
        son vocabulario de tema y ya puntúan en el SUS; ni códigos de norma,
        números de actos legales o rangos de años (ver _is_magnitude).

        ATENCIÓN: recibe el texto CRUDO extraído del PDF, no el preprocesado.
        El preprocesado descarta los tokens numéricos, así que sobre él los
        patrones 'percentages' y 'large_numbers' devuelven siempre cero.
        """
        logger.info("Calculando QUANT scores (porcentajes, magnitudes y unidades físicas)...")
        return np.array([self.quant_hits(t) / max(len(t.split()), 1) for t in raw_texts])

    #: Alias retrocompatible (lo usan scripts de análisis).
    _WORD_RE = WORD_RE

    def calculate_hedge_scores(self, raw_texts: List[str]) -> np.ndarray:
        """
        Densidad de lenguaje no comprometido: proporción de palabras del
        documento en las categorías Uncertainty y WeakModal de L&M, sin
        "risk"/"risks" (nombrar un riesgo no es matizar una afirmación) y sin
        "May" como mes.

        ATENCIÓN: recibe el texto CRUDO, no el lematizado. El preprocesado quita
        las stopwords, y entre ellas están los modales que definen la categoría
        (may, might, could) y perhaps: sobre el texto lematizado esas entradas
        no podían contar nunca. El léxico conserva las formas flexionadas
        originales de L&M, que son las que aparecen en crudo.
        """
        logger.info("Calculando HEDGE scores (L&M Uncertainty + WeakModal)...")
        scores = []
        for text in raw_texts:
            hits, n_tokens = self.hedge_counts(text)
            scores.append(hits / max(n_tokens, 1))
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
