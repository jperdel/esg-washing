#!/bin/sh
# Ejecucion completa de la especificacion adoptada (vocabulario v3 con
# sectoriales): pipeline y todas las cifras, tablas y figuras del paper.
#
#   sh scripts/run_all_v3.sh            # todo (unas 2,5 h)
#   sh scripts/run_all_v3.sh --analysis # solo la capa de analisis (sin main.py)
#
# PY apunta al interprete del entorno del proyecto (Python 3.11).
set -e
cd "$(dirname "$0")/.."
PY="${PY:-C:/Users/Jorge/anaconda3/envs/esgwashing/python.exe}"
export ESG_RUN_TAG=v3 ESG_INCLUDE_SECTORAL=1 PYTHONIOENCODING=utf-8

if [ "$1" != "--analysis" ]; then
    "$PY" main.py                                  # extraccion, preprocesado y scores
fi
"$PY" scripts/corpus_composition.py                # composicion del corpus (seccion de datos)
"$PY" scripts/analysis_v3.py                       # summary.json, tablas y csv
"$PY" scripts/validity_v3.py                       # validez convergente SUS-QUANT
"$PY" scripts/stability_v3.py                      # estabilidad con extraccion fija
"$PY" scripts/stability_v3_e2e.py --rebuild        # estabilidad con re-extraccion simulada
"$PY" scripts/robustness_v3.py                     # inferencia de las perturbaciones
"$PY" scripts/revision_v3.py                       # diagnosticos de la revision (issue #3)
"$PY" scripts/extraction_stats_v3.py               # densidad en la extraccion
"$PY" scripts/vocab_v3_stats.py                    # composicion del vocabulario
"$PY" scripts/figures_v3.py                        # figuras del paper
