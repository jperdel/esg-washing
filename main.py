from esg_washing_pipeline import ESGWashingPipeline
from config import RESULTS_DIR, PDF_DATA_DIR

from pathlib import Path
from datetime import datetime

import sys
from loguru import logger

logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>", level="INFO")
logger.add("logs/pipeline_{time:YYYY-MM-DD}.log", rotation="10 MB", retention="10 days", level="DEBUG")

def main(data_dir: Path, results_dir: Path):
    logger.info("Iniciando Pipeline de detección de ESG-Washing")
    
    # Verificación de directorios
    if not data_dir.exists():
        logger.error(f"El directorio de datos no existe: {data_dir}")
        return

    try:
        pipeline = ESGWashingPipeline()
        
        logger.info(f"Escaneando ficheros en: {data_dir}")
        # informes_list = [f for f in data_dir.iterdir() if f.is_file()]
        informes_list = [f for f in data_dir.rglob("*") if f.is_file()]
        logger.info(f"Se han encontrado {len(informes_list)} ficheros para analizar")

        filename = f"results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
        
        # Ejecución del análisis
        pipeline.run_analysis(informes_list, results_dir, filename)
        
        logger.success(f"Proceso finalizado correctamente. Resultados guardados en {results_dir / filename}")

    except Exception as e:
        logger.exception(f"Error crítico durante la ejecución del pipeline: {e}")

if __name__ == "__main__":

    main(PDF_DATA_DIR, RESULTS_DIR)