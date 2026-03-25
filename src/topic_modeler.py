import pandas as pd
import numpy as np
from typing import List
from pathlib import Path
from loguru import logger
from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import stat 
import os

class LDATopicModeler:
    def __init__(self, num_topics=10, alpha='auto', random_state=None):
        self.num_topics = num_topics
        self.alpha = alpha
        self.random_state = random_state
        self.dictionary = None
        self.corpus = None
        self.model = None
        self.tokenized_texts = None
        logger.debug(f"LDATopicModeler inicializado para {num_topics} tópicos.")

    def prepare_corpus(self, texts: List[str]):
        """Prepara el diccionario y el corpus (Bag-of-Words) necesarios para Gensim."""
        logger.info("Preparando corpus para LDA...")
        self.tokenized_texts = [t.split() for t in texts]
        
        self.dictionary = corpora.Dictionary(self.tokenized_texts)
        self.dictionary.filter_extremes(no_below=3, no_above=0.9)
        
        self.corpus = [self.dictionary.doc2bow(text) for text in self.tokenized_texts]
        logger.debug(f"Diccionario creado con {len(self.dictionary)} tokens únicos.")

    def update(self, num_topics:int, alpha:str|float):
        """Actualiza los valores de alpha y número de tópicos del LDA"""
        self.alpha = alpha
        self.num_topics = num_topics
        logger.debug(f"Valores actualizados a alpha={alpha} y num_topics={num_topics}")

    def fit(self):
        """Entrena el modelo LDA y calcula la coherencia."""
        if not self.corpus:
            logger.error("El corpus no está preparado. Llama a prepare_corpus() primero.")
            return

        logger.debug(f"Entrenando modelo LDA con k={self.num_topics} y alpha={self.alpha}...")
        self.model = LdaModel(
            corpus=self.corpus,
            num_topics=self.num_topics,
            alpha=self.alpha,
            id2word=self.dictionary,
            random_state=self.random_state,
            passes=50,
            iterations=100
        )
        
        # Cálculo de Coherencia (C_v es la más común en literatura)
        coherence_model = CoherenceModel(
            model=self.model, 
            texts=self.tokenized_texts, 
            dictionary=self.dictionary, 
            coherence='c_v'
        )
        coherence_score = coherence_model.get_coherence()
        logger.success(f"Modelo entrenado. Coherencia (C_v): {coherence_score:.4f}")
        return coherence_score

    def save_results(self, lda_dir: Path, doc_names: List[str]):
        """Guarda las keywords de los tópicos y la matriz documento-tópico."""
        if not self.model:
            logger.error("No hay modelo entrenado para guardar resultados.")
            return

        lda_dir.mkdir(parents=True, exist_ok=True)

        # 1. Guardar Top Keywords por Tópico
        topics_data = []
        for i in range(self.num_topics):
            words = self.model.show_topic(i, topn=20)
            topics_data.append({
                "Topic_ID": i,
                "Keywords": ", ".join([f"{word} ({round(prob, 3)})" for word, prob in words])
            })
        
        pd.DataFrame(topics_data).to_csv(lda_dir / "lda_topics_keywords.csv", index=False)
        logger.info(f"Keywords de tópicos guardadas en {lda_dir}/lda_topics_keywords.csv")

        # 2. Guardar Distribución de Tópicos por Documento
        doc_topic_dist = []
        for i, bow in enumerate(self.corpus):
            # Obtiene la distribución de tópicos para el documento i
            dist = self.model.get_document_topics(bow, minimum_probability=0)
            row = {f"Topic_{t_id}": prob for t_id, prob in dist}
            row["Documento"] = doc_names[i]
            # Identificar el tópico dominante para análisis rápido
            row["Dominant_Topic"] = max(dist, key=lambda x: x[1])[0]
            doc_topic_dist.append(row)

        df_docs = pd.DataFrame(doc_topic_dist)
        # Reordenar columnas para que Documento y Dominant_Topic estén al principio
        cols = ["Documento", "Dominant_Topic"] + [c for c in df_docs.columns if c.startswith("Topic_")]
        df_docs[cols].to_csv(lda_dir / "lda_document_distribution.csv", index=False)
        
        logger.info(f"Distribución de documentos guardada en {lda_dir}/lda_document_distribution.csv")

    def save_model_artifacts(self, lda_dir: Path):
        """Guarda el diccionario y el modelo en formato nativo de Gensim."""
        if self.dictionary is None:
            logger.error("No hay diccionario para guardar.")
            return

        # Crear subcarpeta para artefactos binarios
        model_dir = lda_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)

        # 1. Guardar diccionario en formato binario (para recargar en Python)
        dict_path = model_dir / "corpus.dict"
        self.dictionary.save(str(dict_path))
        logger.info(f"Diccionario binario guardado en {dict_path}")

        # 2. Guardar versión legible del diccionario (para inspección humana)
        # Esto guarda: ID | Palabra | Frecuencia en documentos
        readable_dict_path = model_dir / "dictionary_readable.txt"
        with open(readable_dict_path, "w", encoding="utf-8") as f:
            for word_id, word in self.dictionary.items():
                f.write(f"{word_id}\t{word}\t{self.dictionary.dfs[word_id]}\n")
        logger.debug(f"Diccionario legible guardado en {readable_dict_path}")

        # 3. Opcional: Guardar el modelo LDA completo
        if self.model:
            model_path = model_dir / "lda_model.model"
            self.model.save(str(model_path))
            logger.info(f"Modelo LDA guardado en {model_path}")