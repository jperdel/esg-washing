# from esg_washing_pipeline import ESGWashingPipeline
# from config import RESULTS_DIR, PDF_DATA_DIR

# from pathlib import Path
# from datetime import datetime

# import sys
# from loguru import logger

# logger.remove()
# logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>", level="INFO")
# logger.add("logs/pipeline_{time:YYYY-MM-DD}.log", rotation="10 MB", retention="10 days", level="DEBUG")

# def main(data_dir: Path, results_dir: Path):
#     logger.info("Iniciando Pipeline de detección de ESG-Washing")
    
#     # Verificación de directorios
#     if not data_dir.exists():
#         logger.error(f"El directorio de datos no existe: {data_dir}")
#         return

#     try:
#         pipeline = ESGWashingPipeline()
        
#         logger.info(f"Escaneando ficheros en: {data_dir}")
#         # informes_list = [f for f in data_dir.iterdir() if f.is_file()]
#         informes_list = [f for f in data_dir.rglob("*") if f.is_file()]
#         logger.info(f"Se han encontrado {len(informes_list)} ficheros para analizar")

#         filename = f"results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
        
#         # Ejecución del análisis
#         pipeline.run_analysis(informes_list, results_dir, filename)
        
#         logger.success(f"Proceso finalizado correctamente. Resultados guardados en {results_dir / filename}")

#     except Exception as e:
#         logger.exception(f"Error crítico durante la ejecución del pipeline: {e}")

# if __name__ == "__main__":

#     main(PDF_DATA_DIR, RESULTS_DIR)

import sys
import re
import pandas as pd
from pathlib import Path
from datetime import datetime
from loguru import logger

# Importación de las nuevas clases y configuración
from text_processor import TextProcessor
from esgsi_analyzer import ESGSIAnalyzer
from config import RESULTS_DIR, PDF_DATA_DIR, KEYWORDS

# Configuración de logs
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>", level="INFO")
logger.add("logs/pipeline_{time:YYYY-MM-DD}.log", rotation="10 MB", retention="10 days", level="DEBUG")

def main(data_dir: Path, results_dir: Path):
    logger.info("Iniciando Pipeline de detección de ESG-Washing (Versión Modular)")
    
    if not data_dir.exists():
        logger.error(f"El directorio de datos no existe: {data_dir}")
        return

    try:
        # 1. Inicialización de componentes
        processor = TextProcessor()
        analyzer = ESGSIAnalyzer(keywords=KEYWORDS)
        
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

        # 4. Fase de Análisis (Cálculos Estadísticos)
        texts_list = [d["clean_text"] for d in corpus_data]
        
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
        filename = f"results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
        output_path = results_dir / filename
        
        df_res.to_csv(output_path, index=False)
        
        washing_count = len(df_res[df_res["ESGSI"] > 0])
        logger.success(f"Proceso finalizado. Casos de riesgo: {washing_count}/{len(df_res)}")
        logger.info(f"Resultados guardados en: {output_path}")

    except Exception as e:
        logger.exception(f"Error crítico durante la ejecución del pipeline: {e}")

if __name__ == "__main__":
    main(PDF_DATA_DIR, RESULTS_DIR)