import fitz
import re
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel
from loguru import logger  # <-- Importamos loguru
import json

import warnings
warnings.filterwarnings("ignore") # No vamos a actualizar versiones de VertexAI ni GCP

class SemanticDocumentFilter:
    """
    Clase para procesar PDFs de informes anuales, generar chunks 
    y calcular la similitud semántica contra Queries Faro (Anchor Queries).
    """
    def __init__(self, project_id: str, location: str, embedding_model_name: str="text-embedding-004", run_chunker: bool=True):
        logger.info(f"Inicializando pipeline en GCP Project: {project_id}...")
        
        # Cargar el modelo de embeddings una sola vez para la sesión
        if run_chunker: 
            aiplatform.init(project=project_id, location=location)
            self.embedding_model = TextEmbeddingModel.from_pretrained(embedding_model_name)
            self.anchor_embeddings = None
            logger.success("Modelo de embeddings cargado correctamente.")
        
    def fit_anchors(self, anchor_queries: list[str]):
        """
        Pre-calcula y guarda los embeddings de las Queries Faro de referencia.
        """
        logger.info(f"Generando embeddings para {len(anchor_queries)} Anchor Queries (Vectores Faro)...")
        self.anchor_embeddings = self.get_embeddings(anchor_queries)
        logger.success("Vectores Faro (Anchors) listos para la comparación.")

    def get_clean_text(self, pdf_path: Path) -> str:
        """Extrae el texto y realiza una limpieza estructural preservando la semántica."""
        if not pdf_path.exists():
            logger.error(f"No se encontró el archivo PDF: {pdf_path}")
            raise FileNotFoundError(f"No se encontró el archivo: {pdf_path}")
            
        full_text = []
        logger.info(f"Abriendo PDF para extracción de texto: {pdf_path.name}")
        
        with fitz.open(pdf_path) as doc:
            for page in doc:

                text = page.get_text("text")
                text = re.sub(r'https?://\S+|www\.\S+', '', text)
                text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
                text = text.encode("ascii", "ignore").decode("ascii") 
                text = re.sub(r'\n', ' ', text)
                
                full_text.append(text)
        
        combined_text = " ".join(full_text)
        cleaned_text = re.sub(r'\s+', ' ', combined_text).strip()
        
        logger.debug(f"Texto extraído y limpiado del PDF ({len(cleaned_text)} caracteres).")
        return cleaned_text

    def split_into_chunks(self, text: str, pdf_path: Path, chunk_size: int = 1200, overlap: int = 300) -> dict:
        """Divide el texto en bloques con solape y adjunta metadatos."""
        chunks = list()
        start = 0
        text_len = len(text)
        indices_list = list()
        
        while start < text_len:
            end = start + chunk_size
            content = text[start:end].strip()
            indices_list.append([start, end])

            if content:
                chunks.append(content)
            
            start += (chunk_size - overlap)
            
            if chunk_size <= overlap:
                logger.warning("El tamaño del chunk es menor o igual que el solape. Rompiendo bucle para evitar bucle infinito.")
                break

        logger.info(f"Se han generado {len(chunks)} chunks de {chunk_size} caracteres para {pdf_path.name}.")
        
        return {
            "file": pdf_path.name,
            "path": pdf_path,
            "chunks": chunks,
            "index": indices_list
        }

    def get_embeddings(self, textos: list[str], batch_size: int = 25) -> np.ndarray:
        """
        Convierte una lista de textos en una matriz de vectores numéricos,
        procesando en lotes pequeños para respetar límites de la API de Google.
        """
        all_embeddings = []
        total_textos = len(textos)

        for i in range(0, total_textos, batch_size):
            batch = textos[i : i + batch_size]
            
            # Llamada a la API usando el modelo cargado en la clase
            response = self.embedding_model.get_embeddings(batch)
            
            batch_vectors = [e.values for e in response]
            all_embeddings.extend(batch_vectors) 
            logger.debug(f"Lote procesado: {len(all_embeddings)}/{total_textos} embeddings calculados...")

        return np.array(all_embeddings)

    def process_document(self, pdf_path: Path, chunk_size: int = 1200, overlap: int = 300) -> dict:
        """
        Ejecuta el pipeline completo para un documento PDF: 
        Lectura -> Chunking -> Embeddings -> Similitud Coseno.
        Requiere haber ejecutado primero fit_anchors().
        """
        if self.anchor_embeddings is None:
            logger.critical("Se intentó procesar un documento sin haber calculado los Anchor Embeddings primero.")
            raise ValueError("Debes ejecutar pipeline.fit_anchors(anchor_queries) antes de procesar un documento.")

        # 1. Extraer y limpiar
        text = self.get_clean_text(pdf_path)
        
        # 2. Segmentar en Chunks
        results = self.split_into_chunks(text, pdf_path, chunk_size, overlap)
        
        # 3. Vectorizar los chunks
        if not results['chunks']:
            logger.warning(f"El documento {pdf_path.name} no contiene texto extraíble. Abortando vectorización.")
            results['similarity'] = np.array([])
            return results

        logger.info(f"Vectorizando los chunks de {pdf_path.name} en lotes...")
        chunk_embeddings = self.get_embeddings(results['chunks'])
        
        # 4. Calcular Similitud Coseno
        max_sim = cosine_similarity(chunk_embeddings, self.anchor_embeddings).max(axis=1)
        results['similarity'] = max_sim
        
        results['full_text'] = text

        logger.success(f"Procesamiento completo y similitud coseno calculada para {pdf_path.name}.")
        return results

    def save_results(self, pdf_path: Path, results: dict):
        """
        Guarda el resultado en formato JSON sustituyendo la carpeta 'pdf' por 'chunks' 
        en la estructura de directorios y cambiando la extensión del archivo a .json.
        """
        # 1. Convertimos la ruta a String para el reemplazo de carpetas intermedias
        path_str = str(pdf_path)
        
        # 2. Reemplazamos la carpeta 'pdf' por 'chunks'
        # Nota: if 'pdf' no está en la ruta, se quedará igual, así que añade un fallback si lo necesitas
        nuevo_path_str = path_str.replace('pdf', 'chunks')
        
        # 3. Lo volvemos a pasar a Path y cambiamos la extensión a .json
        res_path = Path(nuevo_path_str).with_suffix('.json')

        # 4. Creamos las carpetas padre si no existen (los subdirectorios!)
        res_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Serializamos a JSON asegurando que las variables NumPy sean flotantes nativos de Python
            cleaned_results = {
                "file": results.get("file"),
                "path": str(results.get("path")),
                "full_text": str(results.get("full_text")),
                "chunks": results.get("chunks", []),
                "index": [list(idx) for idx in results.get("index", [])],
                "similarity": [float(s) for s in results.get("similarity", [])]
            }

            with open(res_path, 'w', encoding='utf-8') as f:
                import json
                json.dump(cleaned_results, f, ensure_ascii=False, indent=4)
                
            logger.success(f"Guardado JSON exitoso en: {res_path}")

        except Exception as e:
            logger.error(f"No se pudo guardar el archivo JSON para {pdf_path.name}: {e}")


    def process_folder(self, root_folder: Path, chunk_size: int = 1200, overlap: int = 300):
        """
        Busca recursivamente todos los PDFs en root_folder y sus subcarpetas,
        los procesa uno por uno y guarda los resultados.
        """
        if not root_folder.exists():
            logger.critical(f"La carpeta raíz no existe: {root_folder}")
            raise FileNotFoundError(f"Carpeta no encontrada: {root_folder}")

        # .rglob('*.pdf') busca en carpetas y subcarpetas recursivamente
        # Usamos lower() para capturar tanto .pdf como .PDF por si acaso
        pdf_files = [p for p in root_folder.rglob('*') if p.suffix.lower() == '.pdf']
        
        total_files = len(pdf_files)
        logger.info(f"Se encontraron {total_files} archivos PDF para procesar en '{root_folder.name}'.")

        for idx, pdf_path in enumerate(pdf_files, start=1):
            logger.info(f"[{idx}/{total_files}] Iniciando procesamiento de: {pdf_path}")
            
            try:
                # 1. Procesamos el documento
                document_results = self.process_document(pdf_path, chunk_size, overlap)
                
                # 2. Si tiene chunks válidos, lo guardamos
                if len(document_results.get("chunks", [])) > 0:
                    self.save_results(pdf_path, document_results)
                else:
                    logger.warning(f"Se omitió el guardado de {pdf_path.name} porque no arrojó chunks.")

            except Exception as e:
                # El logger.exception imprime el traceback entero en rojo para que sepas dónde petó exactamente
                logger.exception(f"Error crítico procesando el archivo {pdf_path.name}: {e}")
                # El loop CONTINÚA con el siguiente PDF gracias al try/except!
                continue

        logger.success(f"Proceso de la carpeta '{root_folder.name}' finalizado. Se intentaron procesar {total_files} PDFs.")

    def merge_chunks(self, chunks: list[str], indices: list[list[int]], full_text: str):

        merged_intervals = list()

        if len(chunks) > 1:

            for i in range(len(chunks)-1):
                curr_start, curr_end = indices[i]
                next_start, next_end = indices[i+1]
                
                if next_start <= curr_end:
                    
                    curr_end = max(curr_end, next_end)
                else:
                    merged_intervals.append((curr_start, curr_end))

            merged_intervals.append((curr_start, curr_end))

            final_parts = []
            for i, (start, end) in enumerate(merged_intervals):
                part_text = full_text[start:end]
                final_parts.append(part_text)

            return "\n\n\n\n".join(final_parts)

        elif len(chunks) == 1:
            return chunks[0]
        
        else:
            return ""

    def filter_relevant_chunks(self, json_folder: Path, similarity_threshold: float):

        logger.info(f"Filtrando los chunks en la carpeta {json_folder} por similitud coseno...")
        json_paths = [p for p in json_folder.rglob('*') if p.suffix.lower() == '.json']

        num_of_empty_docs = 0
        total_docs = len(json_paths)

        for i, path in enumerate(json_paths, 1):

            if i%50==0:
                logger.info(f"Filtrados {i}/{total_docs} documentos")

            with open(path, mode="r", encoding="utf-8") as f:
                aux_data = json.load(f)
            
                relevant_chunk_list = [x[0] for x in zip(aux_data['chunks'], aux_data['similarity']) if x[1] > similarity_threshold]
                relevant_index_list = [x[0] for x in zip(aux_data['index'], aux_data['similarity']) if x[1] > similarity_threshold]
                full_text = aux_data['full_text']

                merged_revelant_chunks = self.merge_chunks(relevant_chunk_list, relevant_index_list, full_text)

                aux_data.pop('similarity')
                aux_data.pop('full_text')
                aux_data['chunks'] = relevant_chunk_list
                aux_data['index'] = relevant_index_list
                aux_data['relevant_text'] = merged_revelant_chunks
            
            if len(relevant_chunk_list) == 0:
                logger.warning(f"El documento {path} no tiene chunks con una relevancia superior o igual a {similarity_threshold}")
                num_of_empty_docs += 1

            path_str = str(path)
            filt_path_str = path_str.replace('chunks', 'filtered')
            filt_path = Path(filt_path_str).with_suffix('.json')

            filt_path.parent.mkdir(parents=True, exist_ok=True)

            with open(filt_path, mode='w', encoding="utf-8") as f:

                json.dump(aux_data, f, indent=4)

            logger.debug(f"Filtrado con éxito el fichero {path} y guardado en {filt_path}")
        
        logger.success(f"Filtrados todos los documentos: {num_of_empty_docs} de {total_docs} no tienen chunks relevantes para un umbral de similitud de {similarity_threshold}")
