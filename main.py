from __future__ import annotations
import sys
import re
import pandas as pd
from pathlib import Path
from loguru import logger

# Asegurar que se usan los módulos de ESTA copia del proyecto,
# no los del paquete instalado globalmente con pip install -e
sys.path.insert(0, str(Path(__file__).parent / "src"))

from topic_modeler import LDATopicModeler
from text_processor import TextProcessor
from esgsi_analyzer import ESGSIAnalyzer
from lexical_document_filter import LexicalDocumentFilter
from metadata_loader import load_document_metadata, get_doc_type

from config import (
    METADATA_EXCEL,
    SPACY_MODEL, ESG_KEYWORDS, HEDGE_KEYWORDS, QUANT_PATTERNS, ESGSI_EXT_WEIGHTS, PERSONAL_SW,
    K_TOPICS_LIST, ALPHA_LIST, K_ITERS, LDA_STOPWORDS,
    PDF_DATA_DIR, LEXICAL_DATA_DIR, CLEAN_DATA_DIR, METRICS_RESULTS_DIR, LDA_RESULTS_DIR,
    LEXICAL_KW_THRESHOLD, LEXICAL_CONTEXT_PARAS, LEXICAL_MIN_ZONE_LEN,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="INFO",
)
logger.add("logs/pipeline_{time:YYYY-MM-DD}.log", rotation="10 MB", retention="10 days", level="DEBUG")


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def main(
    pdf_data_dir:       Path,
    lexical_data_dir:   Path,
    results_dir:        Path,
    lda_dir:            Path,
    processed_dir:      Path,
    run_extraction:     bool,
    run_preproc:        bool,
    run_esgsi_analysis: bool,
    run_lda:            bool,
):
    logger.info(
        f"Iniciando Pipeline ESG-Washing (LexicalFilter):\n"
        f"  Extracción={run_extraction} | "
        f"Preprocesado={run_preproc} | ESGSI={run_esgsi_analysis} | LDA={run_lda}"
    )

    if not pdf_data_dir.exists():
        logger.error(f"Directorio de datos no encontrado: {pdf_data_dir}")
        return

    # -- Metadatos de tipo de documento ----------------------------------
    doc_metadata = {}
    if METADATA_EXCEL.exists():
        doc_metadata = load_document_metadata(METADATA_EXCEL)
    else:
        logger.warning(f"No se encontró el fichero de metadatos: {METADATA_EXCEL}. "
                       "El campo 'TipoDocumento' se rellenará como 'Unknown'.")

    try:
        # -- Instanciación de módulos ------------------------------------
        if run_extraction:
            extractor = LexicalDocumentFilter(
                kw_threshold=LEXICAL_KW_THRESHOLD,
                context_paras=LEXICAL_CONTEXT_PARAS,
                min_zone_len=LEXICAL_MIN_ZONE_LEN,
            )
        if run_preproc:
            processor = TextProcessor(extra_sw=PERSONAL_SW, spacy_model=SPACY_MODEL)

        if run_esgsi_analysis:
            analyzer = ESGSIAnalyzer(
                keywords=ESG_KEYWORDS,
                hedge_words=HEDGE_KEYWORDS,
                quant_patterns=QUANT_PATTERNS,
                ext_weights=ESGSI_EXT_WEIGHTS,
            )

        # -- Extracción lexical (reemplaza chunking + filtrado semántico) ----
        if run_extraction:
            logger.info(f"Extracción lexical de PDFs en: {pdf_data_dir}")
            extractor.process_folder(root_folder=pdf_data_dir)

        # -- Preprocesado ------------------------------------------------
        if run_preproc:
            logger.info("Ejecutando preprocesado de texto...")
            informes_paths = [f for f in lexical_data_dir.rglob("*.json") if f.is_file()]
            logger.info(f"Encontrados {len(informes_paths)} JSONs para procesar")

            if not informes_paths:
                logger.warning("No hay archivos JSON para procesar.")
                return

            corpus_data  = []
            year_4digit  = re.compile(r"20\d{2}")
            year_2digit  = re.compile(r"(?:ar|report)(\d{2})\b", re.IGNORECASE)

            for i, path in enumerate(informes_paths, 1):
                if i % 50 == 0:
                    logger.info(f"  Procesando {i}/{len(informes_paths)}...")
                logger.debug(f"  Procesando: {path}")

                raw_text   = processor.extract_from_json(path)
                clean_text = processor.preprocess(raw_text)

                if clean_text:
                    m4 = year_4digit.search(path.name)
                    if m4:
                        year = m4.group()
                    else:
                        m2 = year_2digit.search(path.name)
                        year = ("20" + m2.group(1)) if m2 else "N/A"
                    country    = path.parent.parent.name
                    company    = path.parent.name

                    doc_type = get_doc_type(doc_metadata, country, company, year)

                    corpus_data.append({
                        "Documento":      path.name,
                        "País":           country,
                        "Compañía":       company,
                        "Año":            year,
                        "TipoDocumento":  doc_type,
                        "clean_text":     clean_text,
                    })

            if not corpus_data:
                logger.error("No se pudo extraer texto válido de ningún documento.")
                return

            processed_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(corpus_data).to_csv(
                processed_dir / "processed_texts.csv", index=False, sep=";"
            )
            logger.success(f"Corpus preprocesado guardado ({len(corpus_data)} documentos).")

        else:
            # Cargar corpus ya procesado
            csv_path = CLEAN_DATA_DIR / "processed_texts.csv"
            if not csv_path.exists():
                logger.error(f"No existe {csv_path}. Ejecuta con run_preproc=True primero.")
                return

            df_clean    = pd.read_csv(csv_path, sep=";")
            corpus_data = df_clean.to_dict(orient="records")

            # Retrocompatibilidad: añadir TipoDocumento si falta la columna
            # (también se recalcula si ya existe para aplicar el mapa corregido)
            year_4digit = re.compile(r"20\d{2}")
            year_2digit = re.compile(r"(?:ar|report)(\d{2})\b", re.IGNORECASE)

            needs_recalc = "TipoDocumento" not in df_clean.columns
            if not needs_recalc:
                logger.info("Recalculando TipoDocumento con mapa de nombres actualizado...")
            else:
                logger.warning("Columna 'TipoDocumento' no encontrada. Calculando desde metadatos...")

            for row in corpus_data:
                doc_name = str(row.get("Documento", ""))
                m4 = year_4digit.search(doc_name)
                if m4:
                    year = m4.group()
                else:
                    m2 = year_2digit.search(doc_name)
                    year = ("20" + m2.group(1)) if m2 else row.get("Año", "N/A")

                row["TipoDocumento"] = get_doc_type(
                    doc_metadata,
                    str(row.get("Pa\u00eds", row.get("Pais", ""))),
                    str(row.get("Compa\u00f1\u00eda", row.get("Compania", ""))),
                    year,
                )

        texts_list = [d["clean_text"] for d in corpus_data]
        doc_names  = [d["Documento"]  for d in corpus_data]

        # -- Análisis ESGSI ----------------------------------------------
        if run_esgsi_analysis:
            logger.info("Ejecutando análisis ESGSI...")

            sus_scores   = analyzer.calculate_sus_scores(texts_list)
            sen_scores   = analyzer.calculate_sen_scores(texts_list)
            quant_scores = analyzer.calculate_quant_scores(texts_list)
            hedge_scores = analyzer.calculate_hedge_scores(texts_list)

            esgsi_scores     = analyzer.compute_index(sus_scores, sen_scores)
            esgsi_ext_scores = analyzer.compute_extended_index(
                sus_scores, sen_scores, quant_scores, hedge_scores
            )

            results = []
            for i, data in enumerate(corpus_data):
                results.append({
                    "Documento":      data["Documento"],
                    "País":           data["País"],
                    "Compañía":       data["Compañía"],
                    "Año":            data["Año"],
                    "TipoDocumento":  data.get("TipoDocumento", "Unknown"),
                    "SUS_Score":      round(float(sus_scores[i]),       4),
                    "SEN_Score":      round(float(sen_scores[i]),       4),
                    "QUANT_Score":    round(float(quant_scores[i]),     4),
                    "HEDGE_Score":    round(float(hedge_scores[i]),     4),
                    "ESGSI":          round(float(esgsi_scores[i]),     4),
                    "ESGSI_ext":      round(float(esgsi_ext_scores[i]), 4),
                    "Etiqueta":       "Potential ESG-washing" if esgsi_scores[i]     > 0 else "Likely Genuine",
                    "Etiqueta_ext":   "Potential ESG-washing" if esgsi_ext_scores[i] > 0 else "Likely Genuine",
                })

            df_res = pd.DataFrame(results)
            results_dir.mkdir(parents=True, exist_ok=True)
            output_path = results_dir / "results.csv"
            df_res.to_csv(output_path, index=False, sep=";")

            n_wash     = (df_res["ESGSI"]     > 0).sum()
            n_wash_ext = (df_res["ESGSI_ext"] > 0).sum()
            logger.success(
                f"ESGSI finalizado — riesgo original: {n_wash}/{len(df_res)} | "
                f"riesgo extendido: {n_wash_ext}/{len(df_res)}"
            )
            logger.info(f"Resultados guardados en: {output_path}")

        # -- Topic Modeling LDA ------------------------------------------
        if run_lda:
            logger.info("Ejecutando Topic Modeling (LDA) — modo exploración...")

            lda_modeler = LDATopicModeler()
            lda_modeler.prepare_corpus(texts_list, lda_stopwords=LDA_STOPWORDS)
            lda_dir.mkdir(parents=True, exist_ok=True)

            coherence_rows = []
            total_models   = len(K_TOPICS_LIST) * len(ALPHA_LIST) * K_ITERS
            n_model        = 1

            for num_topics in K_TOPICS_LIST:
                for alpha in ALPHA_LIST:
                    lda_modeler.update(num_topics, alpha)

                    for k in range(K_ITERS):
                        logger.info(f"Entrenando modelo #{n_model}/{total_models} "
                                    f"(k={num_topics}, alpha={alpha}, iter={k})...")
                        n_model += 1

                        metrics = lda_modeler.fit()

                        coherence_rows.append({
                            "Num_topics": num_topics,
                            "Alpha":      alpha,
                            "Iter":       k,
                            "C_v":        round(metrics.get("c_v",    float("nan")), 4),
                            "U_mass":     round(metrics.get("u_mass", float("nan")), 4),
                            "C_npmi":     round(metrics.get("c_npmi", float("nan")), 4),
                        })
                        logger.info(
                            f"  C_v={metrics.get('c_v', float('nan')):.4f} | "
                            f"U_mass={metrics.get('u_mass', float('nan')):.4f} | "
                            f"C_npmi={metrics.get('c_npmi', float('nan')):.4f}"
                        )

                        run_dir = lda_dir / f"alpha_{alpha}_n_topics_{num_topics}_k_{k}"
                        lda_modeler.save_results(run_dir, doc_names)
                        lda_modeler.save_model_artifacts(run_dir)
                        logger.info(f"Artefactos guardados en: {run_dir}")

            df_coh = pd.DataFrame(coherence_rows)
            df_coh.to_csv(lda_dir / "coherence.csv", index=False, sep=";")
            logger.success(f"Coherencia guardada en {lda_dir / 'coherence.csv'}")

    except Exception as e:
        logger.exception(f"Error crítico en el pipeline: {e}")


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    run_extraction     = True
    run_preproc        = True
    run_esgsi_analysis = True
    run_lda            = False

    main(
        pdf_data_dir      = PDF_DATA_DIR,
        lexical_data_dir  = LEXICAL_DATA_DIR,
        results_dir       = METRICS_RESULTS_DIR,
        lda_dir           = LDA_RESULTS_DIR,
        processed_dir     = CLEAN_DATA_DIR,
        run_extraction    = run_extraction,
        run_preproc       = run_preproc,
        run_esgsi_analysis= run_esgsi_analysis,
        run_lda           = run_lda,
    )
