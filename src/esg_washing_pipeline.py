# import re
# import fitz  # install via PyMuPDF
# import numpy as np
# import pandas as pd
# from typing import List
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from textblob import TextBlob
# import nltk

# from pathlib import Path

# from config import KEYWORDS

# # Descarga de recursos necesarios de NLTK
# nltk.download('punkt', quiet=True)
# nltk.download('stopwords', quiet=True)
# nltk.download('wordnet', quiet=True)

# class ESGWashingPipeline:
#     def __init__(self):
#         """
#         Inicializa el pipeline con palabras clave específicas de ESG.
#         """
#         self.lemmatizer = WordNetLemmatizer()
#         self.stop_words = set(stopwords.words('english'))
#         # Vocabulario basado en el artículo (GRI/SASB)
#         self.keywords = KEYWORDS

#     def preprocess_text(self, text: str) -> str:
#         """Limpia, normaliza y lematiza el texto."""
#         # URLs y minúsculas
#         text = re.sub(r'http\S+', '', text.lower())
#         # Solo letras
#         text = re.sub(r'[^a-zA-Z\s]', '', text)
#         # Tokenización, filtrado de stopwords y lematización
#         tokens = [
#             self.lemmatizer.lemmatize(word) 
#             for word in text.split() 
#             if word not in self.stop_words and len(word) > 2
#         ]
#         return " ".join(tokens)

#     def extract_from_pdf(self, pdf_path: Path) -> str:
#         """Extrae texto de un archivo PDF y lo preprocesa."""
#         try:
#             doc = fitz.open(pdf_path)
#             full_text = "".join([page.get_text() for page in doc])
#             doc.close()
#             return self.preprocess_text(full_text)
#         except Exception as e:
#             print(f"Error al leer {pdf_path}: {e}")
#             return ""
        
#     def save_results(self, df: pd.DataFrame, res_path: Path, file_name: str) -> None:
#         """Guarda el df resultante en la carpeta de resultados"""

#         df.to_csv(res_path / file_name, index=False)

#     def run_analysis(self, pdf_paths: List[Path], res_path: Path, file_name: str) -> None:
#         """Ejecuta el pipeline completo para una lista de PDFs."""
        
#         # 1. Extracción y Limpieza
#         print("Cargando y preprocesando documentos...")
#         processed_texts = [self.extract_from_pdf(path) for path in pdf_paths]
#         # Filtrar textos vacíos si hubo errores
#         valid_texts = [txt for txt in processed_texts if txt]

#         # 2. Cálculo de Sostenibilidad (SUS) mediante TF-IDF
#         vectorizer = TfidfVectorizer(vocabulary=self.keywords, binary=False)
#         tfidf_matrix = vectorizer.fit_transform(valid_texts)
#         sus_scores = tfidf_matrix.toarray().sum(axis=1)

#         # 3. Cálculo de Sentimiento (SEN)
#         sen_scores = [TextBlob(txt).sentiment.polarity for txt in valid_texts]

#         # 4. Normalización Z-score y Cálculo de ESGSI
#         def z_score(data):
#             arr = np.array(data)
#             return (arr - np.mean(arr)) / np.std(arr) if np.std(arr) != 0 else np.zeros_like(arr)

#         norm_sen = z_score(sen_scores)
#         norm_sus = z_score(sus_scores)
#         esgsi_scores = norm_sen - norm_sus

#         # 5. Construcción de resultados
#         results = []
#         for i, score in enumerate(esgsi_scores):
#             results.append({
#                 "Documento": pdf_paths[i].name,
#                 "SUS_Score": sus_scores[i],
#                 "SEN_Score": sen_scores[i],
#                 "ESGSI": score,
#                 "Etiqueta": "Potential ESG-washing" if score > 0 else "Likely Genuine"
#             })

#         df_res = pd.DataFrame(results)

#         self.save_results(df_res, res_path, file_name)


import re
import fitz  # install via PyMuPDF
import numpy as np
import pandas as pd
from typing import List
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from loguru import logger
import nltk
from pathlib import Path

from config import KEYWORDS

# Descarga de recursos necesarios de NLTK
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class ESGWashingPipeline:
    def __init__(self):
        """
        Inicializa el pipeline con palabras clave específicas de ESG.
        """
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.keywords = KEYWORDS
        logger.debug(f"Pipeline inicializado con {len(self.keywords)} palabras clave.")

    def preprocess_text(self, text: str) -> str:
        """Limpia, normaliza y lematiza el texto."""
        # URLs y minúsculas
        text = re.sub(r'http\S+', '', text.lower())
        # Solo letras
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Tokenización, filtrado de stopwords y lematización
        tokens = [
            self.lemmatizer.lemmatize(word) 
            for word in text.split() 
            if word not in self.stop_words and len(word) > 2
        ]
        processed = " ".join(tokens)
        return processed

    def extract_from_pdf(self, pdf_path: Path) -> str:
        """Extrae texto de un archivo PDF y lo preprocesa."""
        logger.info(f"Procesando archivo: {pdf_path.name}")
        try:
            doc = fitz.open(pdf_path)
            full_text = "".join([page.get_text() for page in doc])
            doc.close()
            
            if not full_text.strip():
                logger.warning(f"El archivo {pdf_path.name} parece estar vacío o ser una imagen.")
                return ""
                
            clean_text = self.preprocess_text(full_text)
            logger.debug(f"Extracción completada para {pdf_path.name} ({len(clean_text)} caracteres tras limpieza).")
            return clean_text
            
        except Exception as e:
            logger.error(f"Error al leer el PDF {pdf_path.name}: {e}")
            return ""
        
    def save_results(self, df: pd.DataFrame, res_path: Path, file_name: str) -> None:
        """Guarda el df resultante en la carpeta de resultados."""
        try:
            # Crear directorio de resultados si no existe
            res_path.mkdir(parents=True, exist_ok=True)
            output_file = res_path / file_name
            df.to_csv(output_file, index=False)
            logger.info(f"Resultados exportados exitosamente a: {output_file}")
        except Exception as e:
            logger.error(f"No se pudo guardar el archivo CSV: {e}")

    def run_analysis(self, pdf_paths: List[Path], res_path: Path, file_name: str) -> None:
        """Ejecuta el pipeline completo para una lista de PDFs."""
        
        logger.info(f"Iniciando análisis de {len(pdf_paths)} documentos.")
        
        # 1. Extracción y Limpieza
        processed_texts = [self.extract_from_pdf(path) for path in pdf_paths]
        
        # Mapeo de textos válidos para no perder la referencia del nombre del archivo
        valid_data = [(path.name, txt) for path, txt in zip(pdf_paths, processed_texts) if txt]
        
        if not valid_data:
            logger.warning("No se obtuvo texto válido de ningún documento. Abortando análisis.")
            return