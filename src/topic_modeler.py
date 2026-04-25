from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, FrozenSet
from pathlib import Path
from loguru import logger
from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel


class LDATopicModeler:
    """
    Entrena modelos LDA con Gensim y calcula métricas de calidad ampliadas:

    Coherencia (por run):
        · C_v    — métrica de ventana deslizante basada en PMI; la más citada en
                   literatura de topic modeling (Röder et al., 2015).
        · U_mass — coherencia basada en co-ocurrencia en el corpus; más rápida
                   pero sensible al tamaño del corpus.
        · C_npmi — PMI normalizada; complementaria a C_v, menos sensible a
                   distribuciones muy asimétricas.

    Estabilidad (entre runs con semillas distintas):
        · Jaccard medio entre los top-N términos de cada tópico, alineados
          greedily entre modelos (Jaccard ∈ [0, 1]; 1 = tópicos idénticos).
        · Un Jaccard medio > 0.5 indica estabilidad aceptable del modelo.
    """

    def __init__(self, num_topics: int = 10, alpha: str | float = "auto", random_state: int | None = None):
        self.num_topics   = num_topics
        self.alpha        = alpha
        self.random_state = random_state
        self.dictionary   = None
        self.corpus       = None
        self.model        = None
        self.tokenized_texts = None
        logger.debug(f"LDATopicModeler inicializado — k={num_topics}, alpha={alpha}.")

    # ------------------------------------------------------------------
    # Preparación del corpus
    # ------------------------------------------------------------------

    def prepare_corpus(self, texts: List[str], lda_stopwords: set | None = None):
        """
        Construye el diccionario y el corpus Bag-of-Words para Gensim.

        Parameters
        ----------
        texts : List[str]
            Textos ya preprocesados (lematizados, sin stopwords básicas).
        lda_stopwords : set, optional
            Stopwords adicionales específicas para LDA (nombres de empresa,
            boilerplate de reporting, gentilicios, artefactos). Se aplican
            antes de construir el diccionario y no afectan al scoring ESGSI.
        """
        logger.info("Preparando corpus para LDA...")
        sw = lda_stopwords or set()

        self.tokenized_texts = [
            [tok for tok in t.split() if tok not in sw]
            for t in texts
        ]

        self.dictionary = corpora.Dictionary(self.tokenized_texts)
        # no_below=5  → mínimo 5 docs para incluir un término
        # no_above=0.85 → excluye tokens en >85 % de documentos (boilerplate residual)
        self.dictionary.filter_extremes(no_below=5, no_above=0.85)

        self.corpus = [self.dictionary.doc2bow(text) for text in self.tokenized_texts]
        logger.debug(
            f"Diccionario: {len(self.dictionary)} tokens únicos "
            f"(stopwords LDA aplicadas: {len(sw)})."
        )

    def update(self, num_topics: int, alpha: str | float):
        """Actualiza hiperparámetros antes de un nuevo fit()."""
        self.num_topics = num_topics
        self.alpha      = alpha
        logger.debug(f"Hiperparámetros actualizados — k={num_topics}, alpha={alpha}.")

    # ------------------------------------------------------------------
    # Entrenamiento y métricas de coherencia
    # ------------------------------------------------------------------

    def _build_model(self, random_state: int | None = None) -> LdaModel:
        """Instancia y entrena un modelo LDA con los parámetros actuales."""
        return LdaModel(
            corpus=self.corpus,
            num_topics=self.num_topics,
            alpha=self.alpha,
            id2word=self.dictionary,
            random_state=random_state if random_state is not None else self.random_state,
            passes=50,
            iterations=100,
        )

    def _compute_coherence_metrics(self, model: LdaModel) -> dict[str, float]:
        """
        Calcula C_v, U_mass y C_npmi para el modelo dado.
        Devuelve un dict con las tres métricas.
        """
        metrics = {}
        for metric in ("c_v", "u_mass", "c_npmi"):
            try:
                cm = CoherenceModel(
                    model=model,
                    texts=self.tokenized_texts,
                    dictionary=self.dictionary,
                    coherence=metric,
                )
                metrics[metric] = cm.get_coherence()
            except Exception as e:
                logger.warning(f"No se pudo calcular coherencia '{metric}': {e}")
                metrics[metric] = float("nan")
        return metrics

    def fit(self) -> dict[str, float]:
        """
        Entrena el modelo LDA principal y devuelve un diccionario con las tres
        métricas de coherencia: {"c_v": ..., "u_mass": ..., "c_npmi": ...}.
        """
        if not self.corpus:
            logger.error("Corpus no preparado. Llama a prepare_corpus() primero.")
            return {}

        logger.debug(f"Entrenando modelo LDA — k={self.num_topics}, alpha={self.alpha}...")
        self.model = self._build_model()

        metrics = self._compute_coherence_metrics(self.model)
        logger.success(
            f"Modelo entrenado — C_v={metrics['c_v']:.4f}, "
            f"U_mass={metrics['u_mass']:.4f}, "
            f"C_npmi={metrics['c_npmi']:.4f}"
        )
        return metrics

    # ------------------------------------------------------------------
    # Análisis de estabilidad entre semillas
    # ------------------------------------------------------------------

    def _top_words(self, model: LdaModel, topn: int) -> list[frozenset]:
        """Devuelve los top-N términos de cada tópico como frozensets."""
        return [
            frozenset(w for w, _ in model.show_topic(t, topn=topn))
            for t in range(self.num_topics)
        ]

    def _aligned_jaccard(self, topics_a: list[frozenset], topics_b: list[frozenset]) -> float:
        """
        Jaccard promedio con alineación greedy entre dos listas de tópicos.
        Para cada tópico de A, elige el tópico más similar de B (sin repetir)
        y promedia los scores resultantes.
        """
        used: set[int] = set()
        total = 0.0

        for words_a in topics_a:
            best_score, best_j = -1.0, -1
            for j, words_b in enumerate(topics_b):
                if j in used:
                    continue
                union = len(words_a | words_b)
                score = len(words_a & words_b) / union if union > 0 else 0.0
                if score > best_score:
                    best_score, best_j = score, j
            if best_j >= 0:
                used.add(best_j)
            total += max(best_score, 0.0)

        return total / len(topics_a) if topics_a else 0.0

    def compute_stability(self, n_seeds: int, topn: int = 10) -> dict[str, float]:
        """
        Entrena `n_seeds` modelos con semillas distintas (0, 1, …, n_seeds-1)
        y calcula la estabilidad media de los tópicos mediante Jaccard alineado.

        Devuelve:
            {
              "mean_jaccard": float,   # Media de todos los pares de modelos
              "std_jaccard":  float,   # Desviación típica
              "n_comparisons": int,    # Número de pares comparados
            }
        """
        if not self.corpus:
            logger.error("Corpus no preparado. Llama a prepare_corpus() primero.")
            return {}

        logger.info(
            f"Análisis de estabilidad — k={self.num_topics}, "
            f"alpha={self.alpha}, {n_seeds} semillas, topn={topn}..."
        )

        models_words = []
        for seed in range(n_seeds):
            m = self._build_model(random_state=seed)
            models_words.append(self._top_words(m, topn))
            logger.debug(f"  Semilla {seed} entrenada.")

        jaccard_scores = []
        for i in range(n_seeds):
            for j in range(i + 1, n_seeds):
                score = self._aligned_jaccard(models_words[i], models_words[j])
                jaccard_scores.append(score)

        result = {
            "mean_jaccard":   float(np.mean(jaccard_scores))   if jaccard_scores else 0.0,
            "std_jaccard":    float(np.std(jaccard_scores))    if jaccard_scores else 0.0,
            "n_comparisons":  len(jaccard_scores),
        }
        logger.success(
            f"Estabilidad — Jaccard medio={result['mean_jaccard']:.4f} "
            f"(±{result['std_jaccard']:.4f}) sobre {result['n_comparisons']} pares."
        )
        return result

    # ------------------------------------------------------------------
    # Guardado de resultados
    # ------------------------------------------------------------------

    def save_results(self, lda_dir: Path, doc_names: List[str]):
        """Guarda keywords por tópico y la distribución documento-tópico."""
        if not self.model:
            logger.error("No hay modelo entrenado.")
            return

        lda_dir.mkdir(parents=True, exist_ok=True)

        # 1. Top keywords por tópico
        topics_data = []
        for i in range(self.num_topics):
            words = self.model.show_topic(i, topn=20)
            topics_data.append({
                "Topic_ID": i,
                "Keywords": ", ".join(f"{w} ({round(p, 3)})" for w, p in words),
            })
        pd.DataFrame(topics_data).to_csv(lda_dir / "lda_topics_keywords.csv", index=False, sep=";")
        logger.info(f"Keywords guardadas en {lda_dir / 'lda_topics_keywords.csv'}")

        # 2. Distribución de tópicos por documento
        doc_topic_dist = []
        for i, bow in enumerate(self.corpus):
            dist = self.model.get_document_topics(bow, minimum_probability=0)
            row = {f"Topic_{t_id}": prob for t_id, prob in dist}
            row["Documento"] = doc_names[i]
            row["Dominant_Topic"] = max(dist, key=lambda x: x[1])[0]
            doc_topic_dist.append(row)

        df_docs = pd.DataFrame(doc_topic_dist)
        cols = ["Documento", "Dominant_Topic"] + [c for c in df_docs.columns if c.startswith("Topic_")]
        df_docs[cols].to_csv(lda_dir / "lda_document_distribution.csv", index=False, sep=";")
        logger.info(f"Distribución de documentos guardada en {lda_dir / 'lda_document_distribution.csv'}")

    def save_model_artifacts(self, lda_dir: Path):
        """Guarda el modelo y diccionario en formato nativo Gensim."""
        if self.dictionary is None:
            logger.error("No hay diccionario para guardar.")
            return

        model_dir = lda_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)

        self.dictionary.save(str(model_dir / "corpus.dict"))
        logger.info(f"Diccionario guardado en {model_dir / 'corpus.dict'}")

        readable_path = model_dir / "dictionary_readable.txt"
        with open(readable_path, "w", encoding="utf-8") as f:
            for word_id, word in self.dictionary.items():
                f.write(f"{word_id}\t{word}\t{self.dictionary.dfs[word_id]}\n")
        logger.debug(f"Diccionario legible guardado en {readable_path}")

        if self.model:
            self.model.save(str(model_dir / "lda_model.model"))
            logger.info(f"Modelo guardado en {model_dir / 'lda_model.model'}")
