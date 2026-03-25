import sys
import re
import pandas as pd
from pathlib import Path
from datetime import datetime
from loguru import logger
from topic_modeler import LDATopicModeler

# Importación de las nuevas clases y configuración
from text_processor import TextProcessor
from esgsi_analyzer import ESGSIAnalyzer
from config import *

# Configuración de logs
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>", level="INFO")
logger.add("logs/pipeline_{time:YYYY-MM-DD}.log", rotation="10 MB", retention="10 days", level="DEBUG")

def main(
        data_dir: Path, 
        results_dir: Path, 
        lda_dir: Path, 
        processed_dir: Path,
        run_preproc: bool, 
        run_esgsi_analysis: bool,
        run_lda: bool
):
    
    logger.info(f"Iniciando Pipeline de detección de ESG-Washing (Versión Modular) - Preprocesado: {run_preproc}, ESGSI: {run_esgsi_analysis}, LDA: {run_lda}")
    
    if not data_dir.exists():
        logger.error(f"El directorio de datos no existe: {data_dir}")
        return

    try:
        # 1. Inicialización de componentes
        if run_preproc: processor = TextProcessor(extra_sw=PERSONAL_SW, spacy_model=SPACY_MODEL)
        if run_esgsi_analysis: analyzer = ESGSIAnalyzer(keywords=ESG_KEYWORDS)
        
        if run_preproc:
            logger.info("Ejecutando la limpieza de datos")
            # 2. Escaneo de ficheros
            logger.info(f"Escaneando ficheros en: {data_dir}")
            informes_paths = [f for f in data_dir.rglob("*.pdf") if f.is_file()]
            logger.info(f"Se han encontrado {len(informes_paths)} PDFs para analizar")

            if not informes_paths:
                logger.warning("No hay archivos PDF para procesar.")
                return

            # 3. Fase de Procesamiento (Extracción + Limpieza)
            # Guardamos metadatos junto al texto procesado
            corpus_data = []
            year_pattern = r"20\d{2}"

            for i, path in enumerate(informes_paths):
                logger.debug(f"Extrayendo y procesando el texto del documento {path.name}.")
                logger.info(f'Procesando documento {i+1}/{len(informes_paths)}')
                raw_text = processor.extract_from_pdf(path)
                clean_text = processor.preprocess(raw_text)
                
                if clean_text:
                    # Extraer metadatos según tu estructura de carpetas
                    year_match = re.search(year_pattern, path.name)
                    corpus_data.append({
                        "Documento": path.name,
                        "País": path.parent.name,
                        "Compañía": path.parent.parent.name,
                        "Año": year_match.group() if year_match else "N/A",
                        "clean_text": clean_text
                    })
            
            if not corpus_data:
                logger.error("No se pudo extraer texto válido de ningún documento.")
                return
            
            # Guardamos los informes procesados
            processed_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(corpus_data).to_csv(processed_dir / 'processed_texts.csv', index=False)
        
        else:
            # Si no ejecutamos el preproc, leemos los documentos ya limpios
            df_clean = pd.read_csv(CLEAN_DATA_DIR / 'processed_texts.csv')
            corpus_data = list()

            for _, row in df_clean.iterrows():
                corpus_data.append(row.to_dict())

        texts_list = [d["clean_text"] for d in corpus_data]
        doc_names = [d["Documento"] for d in corpus_data]

        if run_esgsi_analysis:
            logger.info("Ejecutando el analisis ESGSI")
            
            # 4. Fase de Análisis (Cálculos Estadísticos)
            sus_scores = analyzer.calculate_sus_scores(texts_list)
            sen_scores = analyzer.calculate_sen_scores(texts_list)
            esgsi_scores = analyzer.compute_index(sus_scores, sen_scores)

            # 5. Consolidación de resultados
            results = []
            for i, data in enumerate(corpus_data):
                score = esgsi_scores[i]
                results.append({
                    "Documento": data["Documento"],
                    "País": data["País"],
                    "Compañía": data["Compañía"],
                    "Año": data["Año"],
                    "SUS_Score": round(sus_scores[i], 4),
                    "SEN_Score": round(sen_scores[i], 4),
                    "ESGSI": round(score, 4),
                    "Etiqueta": "Potential ESG-washing" if score > 0 else "Likely Genuine"
                })

            # 6. Guardado de resultados
            df_res = pd.DataFrame(results)
            results_dir.mkdir(parents=True, exist_ok=True)
            filename = f"results.csv"
            output_path = results_dir / filename
            
            df_res.to_csv(output_path, index=False)
            
            washing_count = len(df_res[df_res["ESGSI"] > 0])
            logger.success(f"Proceso finalizado. Casos de riesgo: {washing_count}/{len(df_res)}")
            logger.info(f"Resultados guardados en: {output_path}")

        if run_lda:
            logger.info("Ejecutando el Topic Modeling con LDA")

            # 7. Fase de Análisis Cualitativo (Topic Modeling - LDA)
            logger.info("Iniciando modelado de temas (LDA)...")
            lda_modeler = LDATopicModeler()
            lda_modeler.prepare_corpus(texts_list)

            lda_dir.mkdir(parents=True, exist_ok=True)

            coherence_list = list()

            total_models = len(K_TOPICS_LIST) * len(ALPHA_LIST) * K_ITERS
            n_model = 1

            for num_topics in K_TOPICS_LIST:
                for alpha in ALPHA_LIST:
                    for k in range(K_ITERS):
        
                        logger.info(f"Entrenando modelo #{n_model}/{total_models}")
                        n_model += 1

                        lda_modeler.update(num_topics, alpha)
                        coherence = lda_modeler.fit()
                        coherence_list.append([num_topics, alpha, k, coherence])
                        logger.info(f"Coherencia del modelo LDA - alpha_{lda_modeler.alpha}_n_topics_{lda_modeler.num_topics}_k_{k}: {coherence:.4f}")

                        # Los resultados de LDA se guardan en una subcarpeta dentro de results_dir
                        aux_dir = lda_dir / f"alpha_{lda_modeler.alpha}_n_topics_{lda_modeler.num_topics}_k_{k}"
                        lda_modeler.save_results(aux_dir, doc_names)
                        # Guardar diccionario y modelo (Artefactos binarios)
                        lda_modeler.save_model_artifacts(aux_dir)

                        logger.info(f"Resultados, diccionario y modelo del LDA guardados en: {aux_dir}")
            
            # Save coherence list
            df_coherence = pd.DataFrame(coherence_list, columns=['Num. topics', 'Alpha', 'Iter', 'Coherence'])
            df_coherence.to_csv(lda_dir / "coherence.csv", index=False)

    except Exception as e:
        logger.exception(f"Error crítico durante la ejecución del pipeline: {e}")

if __name__ == "__main__":

    # Elige las partes del pipeline a ejecutar
    run_preproc = False
    run_esgsi_analysis = False
    run_lda = True

    main(
        PDF_DATA_DIR / "España" / "BBVA",
        METRICS_RESULTS_DIR,
        LDA_RESULTS_DIR,
        CLEAN_DATA_DIR,
        run_preproc=run_preproc,
        run_esgsi_analysis=run_esgsi_analysis,
        run_lda=run_lda
    )