# Plan: adaptar el paper a las directrices (2026-09-26)

Fuente de verdad de las reglas: `paper/_research/directrices.md` (manda sobre
todo lo demás). Este fichero es el estado del trabajo: cada sesión o
relanzamiento lo lee PRIMERO, sigue por la primera tarea no marcada y marca
[x] lo que termine, con una línea en el registro del final.

Entorno: Python `C:/Users/Jorge/anaconda3/envs/esgwashing/python.exe`,
`PYTHONIOENCODING=utf-8 ESG_RUN_TAG=v3 ESG_INCLUDE_SECTORAL=1`. No ejecutar
`main.py` (los resultados del pipeline son finales), no tocar `data/`, no
hacer commit. Compilar en `paper/` con `latexmk -pdf -interaction=nonstopmode main.tex`.

Página de estado para el autor (móvil): https://claude.ai/artifact/FqAvgF2qNGY42ArDTXVhJ6
Su fuente es `C:/Users/Jorge/AppData/Local/Temp/claude/c--Users-Jorge-Desktop-Proyectos-ESGwashing/8bc63ee6-9277-45ff-b62f-29801463cdf6/scratchpad/estado_paper.html`.
Cada vez que se marque una tarea, actualizar ese HTML (estado de la tarea y
una línea en el registro, con hora) y republicarlo con la herramienta Artifact
(mismo file_path, o `url` de arriba desde otra conversación).

## Estructura objetivo del paper

1. Introducción y alcance (`01_scope.tex`): paper propio; Lagasio como antecedente.
2. Datos (`03_data.tex`): corpus, sectores, países, tipos de documento.
3. Extracción (`04_extraction.tex`).
4. Vocabulario (`05_lexicon.tex`): fuentes, criterios, composición, frases
   protegidas, léxico HEDGE. Sin auditoría KWIC, sin historial.
5. Especificación (`06_index_specification.tex`): ESGSI (Lagasio con nuestras
   mejoras) y Extended ESGSI (métrica nueva); por qué densidad y no TF-IDF L2,
   por qué z-score y sin umbral (absorbe lo esencial de §2, §7, §8, §9 en una
   subsección de justificación y una de validez convergente con QUANT).
6. Robustez (`10_robustness.tex`): especificación de SUS, pesos del extended,
   estabilidad del vocabulario v3 (Spearman, solapamiento de deciles extremos,
   pendiente).
7. Resultados (`11_temporal_results.tex`): tiempo, sectores, países, pilares,
   extended ESGSI. Sin etiquetas, sin nombres de empresa.
8. Limitaciones (`12_limitations.tex`).
Apéndice: reproducibilidad (`13_reproducibility.tex`).
`02_original_specification.tex`, `07_measurement_critique.tex`,
`08_normalisation_artefact.tex`, `09_convergent_validity.tex` salen de
`main.tex` una vez absorbido su contenido útil (se borran los ficheros).

## Tareas

### Fase A — análisis (scripts)
- [x] A1. `analysis_v3.py`, `figures_v3.py`: quitar etiquetas (señalados,
  transiciones, años señalada, flips) y nombres de empresa (posiciones
  anónimas por industria); añadir Spearman y solapamiento de deciles extremos
  donde se comparen scores; rejilla de pesos del extended ESGSI
  (w_q, w_h ∈ {0, .25, .5, .75, 1}: Spearman con ESGSI y con el de 0,5,
  solapamiento de deciles, pendiente); ESGSI_ext como métrica propia en todos
  los cortes (año, sector, país, pilar). Figuras sin etiquetas ni empresas.
- [x] A2. `stability_v3.py`, `stability_v3_e2e.py`, `robustness_v3.py`: quitar
  las variantes de vocabularios antiguos (154/289); medidas = Spearman,
  solapamiento de deciles extremos, pendiente (+ r si ya está); sin "cambios"
  de etiqueta ni escalados basados en ellos. Re-ejecutar (e2e ~1 h).
- [x] A3. `validity_v3.py`, `critique_v3.py`: quitar flips/señalados y el
  factorial min–max basado en % señalado; conservar lo que justifica la
  densidad (invariancia L2, dilución, correlaciones con BREADTH y QUANT,
  bases disjuntas, IC por bootstrap de empresa). `vocab_v3_stats.py`: solo
  composición, frases protegidas, HEDGE; sin auditoría ni antes/después.
- [x] A4. Re-ejecutar todo el bloque de análisis y figuras; comprobar salidas.

### Fase B — texto (tras la fase A)
- [x] B1. §1 + §6 (con justificación y validez absorbidas de §2/§7/§8/§9);
  quitar 02/07/08/09 de `main.tex` y borrar los ficheros.
- [x] B2. §5 (vocabulario sin auditoría ni historial) y §4 (ajustes).
- [x] B3. §10 robustez con las nuevas medidas.
- [x] B4. §11 resultados (sin etiquetas ni empresas; extended ESGSI como
  métrica nueva).
- [x] B5. §3, §12, §13 coherentes con todo lo anterior.

### Fase C — revisión
- [x] C1. Revisión adversaria por sección (cifras contra JSON, código contra
  texto, cumplimiento de directrices: grep de "flag", "label", "washing" como
  clasificación, 154/289, nombres de empresa, KWIC/precision).
- [x] C2. Pasada de coherencia entre secciones + compilación limpia.
- [x] C3. Actualizar `pendiente_secciones_no_tocadas.md` con lo que quede para
  el autor y ESCRIBIR UN RESUMEN FINAL en este fichero. Borrar el
  relanzamiento programado (CronList → CronDelete).

## Registro
- 2026-09-26: plan creado; directrices fijadas; relanzamiento cada 2 h.
- 2026-09-26 16:20: A1–A3 cortados por límite de sesión (~11:00); reanudados con su contexto.
- 2026-09-26: A3 hecha. validity_v3/critique_v3 sin etiquetas (índice por spec: r, ρ, deciles extremos; normalización z vs min–max sin umbral; SUS×BREADTH añadido; sin nombres de empresa). vocab_v3_stats solo vocabulario vigente (composición, V=399, frases protegidas, masa v3, board 59.749/59.912, HEDGE). Borrados figures_vocab_v3.py y vocab_precision_mass.pdf (quitar su \includegraphics de §5 en B2).
- 2026-09-26: A1 hecha. analysis_v3/figures_v3 sin etiquetas ni nombres (ids anonimos <Industria>-NN; claves en *_internal.*); ESGSI_ext en todos los cortes; rejilla de pesos (summary `w_pesos_ext`, tex/ext_weights_grid.tex, fig_ext_weights.pdf); fuera flagged_by_group, label_transitions, run_comparison (bloques h/i eliminados; estabilidad se cita de results/stability_v3); extraction_by_doc -> extraction_by_doc_internal.csv.
- 2026-09-26 16:50: B1, B2 y B4 en curso (agentes); B3 espera a A2 (e2e en ejecución).
- 2026-09-26: B2 hecha. §5 reescrita: fuentes (marcos/normas, literatura, revisión estructurada de 17 informes con marca [Author: identify the reviewers of the candidate lists.]), criterios de admisión, pilares, un vocabulario, frases protegidas (11 frases, 13 términos), composición + V=399 por pilar, tabla nueva tab:lexicon-mass (top 10), board 59.749/59.912, HEDGE; fuera KWIC, precisión, consenso, retirados, fig:lexicon-precision, eq:lexicon-precision, tab:lexicon-procedure/-context/-retire/-audit. §4 sin auditoría, pre-v3 ni nombres; sin refs a sec:measurement/sec:artefact. Errores de compilación restantes solo en 06 (B1 en curso: align l.222, eq:esgsi, eq:esgsiext, sec:justification).
- 2026-09-26: B1 hecha. §1 reescrita como introducción de paper propio (score continuo, sin umbral ni clasificación; Lagasio como antecedente con 6 divergencias; sin nombres de empresa). §6: ESGSI principal (SUS dens) y ESGSI_ext como métrica nueva (pesos 0,5 + resumen de la rejilla); fuera la regla de etiquetas; nueva subsección final (5.8 en el PDF) "Why density, z-scores and no threshold" (lo que Lagasio declara/omite, invariancia L2 con amplificación y dilución, BREADTH r 0,9546, idf, validez con QUANT con IC bootstrap por empresa y bases disjuntas, z vs min–max como distribuciones) que define sec:original, sec:measurement, sec:validity(-disjoint, -adopted), sec:artefact(-which). 02/07/08/09 fuera de main.tex y borrados (git rm). Título nuevo. Compila sin errores ni referencias indefinidas.
- 2026-09-26: A2 hecha. stability_v3/_e2e/robustness_v3 solo perturbaciones del v3 (sin 154/289/_prev, sin cambios/señalados/escalado/umbral); medidas rho, r, top10/bot10 (k=34), pendiente MCO y EF (+SE cluster y p wild en robustness_v3.json), para ESGSI y ESGSI_ext (también en re-extracción: QUANT/HEDGE por párrafo añadidos a la caché); nulos igualados con percentil de rho/top10/bot10; variante nueva sin 'board director'. fig_stability: (a) rho, (b) solapamiento de deciles, (c) pendiente EF vs % borrado. analysis_v3 ya no lee stability (A1).
- 2026-09-26: B4 hecha. §11 reescrita como Resultados (550 líneas): ESGSI principal y ESGSI_ext en todos los cortes; sin etiquetas/umbral/señalados/transiciones ni nombres de empresa (heterogeneidad por industria: tab:temporal-firms nueva, dispersión, ICC, persistencia Spearman); pendientes de grupo por IC de pendientes de empresa; confusión mandato/cronología con tab:temporal-frameworks (rho, deciles extremos, pendientes de ambos índices) como no resuelta; sin comparaciones con vocabularios previos; pesos y estabilidad del vocabulario remitidos a §10 sin cifras. Fuera tab:temporal-threats y tab:temporal-pillars (cifras en texto + fig_pillars). Figuras revisadas: sin etiquetas ni nombres. Etiquetas conservadas: sec:temporal(-mandate/-firms/-sectors/-vocabulary), nueva sec:temporal-pillars. Compila sin errores ni referencias indefinidas.
- 2026-09-26: B3 hecha. §10 reescrita (400 líneas) solo con medidas de score (rho, r, deciles extremos 34/343, pendiente EF con SE cluster y p wild), ESGSI y ESGSI_ext: (1) especificación de SUS (tab:robustness-components, tab:robustness-index con rho/r/deciles, fig:robustness-specs; pendiente bajo L2 -0,143; tfidf-l sin pendiente en los scripts); (2) rejilla de pesos (fig:robustness-weights = fig_ext_weights.pdf, tab:robustness-weights nueva con solapamientos; ranking depende de w_h, pendiente casi solo de w_q); (3) estabilidad v3: diseño congelado vs re-extracción simulada y su fidelidad, tab:robustness-deletions (sectoriales, 6 con aviso descritos sin precisión, pilares, board director; ambos modos; percentil en nulo igualado), tab:robustness-random (10–50 %, ambos modos, ambos índices), fig:robustness-stability descrita (bandas de pendiente anchas), quitar-uno; (4) sec:robustness-limits (corpus propio, suelo estructural, estabilidad != validez). Fuera: flips, señalados, umbral, escalado flips-(1-r), sec:robustness-compare, tab:robustness-hierarchy (sin referencias externas). Compila sin errores ni referencias indefinidas. Página de estado HTML no actualizada por este agente.
- 2026-09-26 17:10: B5 hecha. §3 sin nombres de empresa (duplicado = un informe de 2019 presente dos veces; la empresa neerlandesa discutible, anónima), composición de corpus_composition.csv/summary.json, confusión país/tipo, qué identifica y qué no el sector. §12 reescrita contra §10 final (B3 ya [x]): 17 limitaciones ordenadas (score relativo al corpus, sin criterio externo, mandato sin resolver, vocabulario sobre este corpus sin precisión publicada y revisores pendientes de §5, acoplamiento, magnitud de la pendiente dependiente de E/TRANS, pesos del extendido con la rejilla, país/tipo, 49 clústeres, recall, extracción, QUANT, HEDGE, SEN, frases protegidas, capa de texto, no commit); tab:limitations-origin sustituida por tab:limitations-summary (sin referencias externas). §13: tabla de scripts actual (sin figures_vocab_v3; tiempos de los JSON), ficheros *_internal y demás con nombres, semillas sin la del KWIC, parámetros sin fila de etiquetado, Etiqueta/Etiqueta_ext declaradas sin uso, defectos sin cifras del corpus antiguo ni nombres, procedencia al día (stability_v3.py también modificado). Nuevas etiquetas sec:repro-pipeline/-params/-schema/-provenance. Compila sin errores ni referencias indefinidas; sin overfull en 03/12/13. Página HTML de estado no actualizada por este agente.
- 2026-09-26 ~17:30: A4 hecha (9 scripts rc=0; salidas sin nombres de empresa ni etiquetas).
- 2026-09-26 22:20: C1a/C1b/C1c cortados por límite de sesión; reanudados.
- 2026-09-26: C1a hecha (01/03/04/05). Cumplimiento: sin nombres de empresa, etiquetas, umbral usado ni comparaciones con vocabularios previos; la mención del umbral de Lagasio queda solo como motivo. §4: fuera las tasas de calibración del filtro de navegación (51 %, 0,39 %, 2.963) → solo "umbrales calibrados sobre una muestra etiquetada que no está en el repositorio"; frase de "error por año" sin alusión a auditoría. §5: "no precision figure" → "no error rate"; tokenización de HEDGE precisada (compuestos con guion enteros, como en esgsi_analyzer/vocab_v3_stats). ~60 cifras cotejadas con summary.json, extraction_stats.json, extraction_by_year.csv, corpus_composition.csv, industry_x_country_firms.csv, vocab_stats.json: todas cuadran. Procedimiento de §4/§5 cotejado con lexical_document_filter.py, text_processor.py, build_lexicons.py, config.py, main.py, metadata_loader.py: correcto. Marca [Author: …] de §5 intacta. Compila sin errores ni referencias indefinidas; en 01–05 solo el overfull de 0,46 pt. Aviso: results/analysis_v3/corpus_composition.csv (no *_internal) lleva filas "firm" con nombres de empresa.
- 2026-09-26: C1c hecha (§11–§13). >60 cifras de §11 verificadas contra summary.json/CSV (descriptivos, serie, tendencias y Alemania, drivers, marcos, empresas, grupos con IC de pendientes de empresa, varianza, pilares, masa, lift) y varias recalculadas de doc_level.csv; sin discrepancias. Directrices: sin etiquetas, umbral, nombres (texto, tablas y 5 figuras renderizadas) ni comparaciones con vocabularios previos. Arreglos: §11 el panel equilibrado no fija la mezcla de tipos de documento (6 empresas con más de un tipo); pie de fig:temporal-countries (pendiente impresa, no línea). §12: −0,117 marcado "extracción fija"; añadido que el pilar E cae más lejos de su nulo igualado (ρ bajo el 99,8 % de borrados aleatorios). §13: tiempos de critique (854 s) y robustness (33 s) de los JSON actuales; vocab_v3_stats también importa de analysis_v3; fuera la frase sobre figures_vocab_v3 (historial); corpus_composition_internal.csv en la lista de *_internal y en la tabla de scripts. Esquema de results.csv, semillas, recuentos de ficheros y parámetros coinciden con el código; Etiqueta/Etiqueta_ext solo en main.py. §13: tasas del filtro de navegación "no reportadas" (coherente con C1a). Compila sin errores ni referencias indefinidas; único overfull restante en 05 (0,46 pt); ninguno en 11–13.
- 2026-09-26: C1b hecha: §6 y §10 revisadas. >60 cifras cotejadas con validity_v3/critique_v3/summary/ext_weights_grid/robustness_v3/e2e_summary/stability summary/quitar_uno: todas cuadran salvo 'ρ ≥ 0,974' (real 0,97396 → 0,973, en §6 y §10) y Spearman SUS_L2–BREADTH 0,9514 (§6, critique) vs 0,9515 (§10, results.csv) → §6 a 0,951. Definiciones contra código OK (idf suavizado, CountVectorizer n_max, L2+media, W_d/R_d/L_d, ε=1e-6, CR1 con G-1 gl, WCR Rademacher, pilares); raw_text también quita NUL (añadido). Cambios: ESGSI_ext presentado con la dirección de QUANT/HEDGE como elección y HEDGE ≈ riesgo; 'neither can dominate' suavizado; resumen de pesos (ranking ~w_h, pendiente ~w_q) en §6; 'precision' estadística reformulada; 'every other designed deletion' aclarado (salvo E y TRANS); 23/27 y 28/29 explícitos; 'normalisation' cualificada en §10; \texorpdfstring en títulos con $z$. Sin frases de 'borrados como aleatorios' para pilares (E fuera del nulo, correcto). Overfull 6,25pt de §6 corregido; compila sin errores ni referencias indefinidas, sin overfull en 06/10. Pendiente para C2: uso de 'v3' como nombre del vocabulario (§5/§6/§10/§13).
- 2026-09-26: C1 completa (a, b, c). corpus_composition.py: filas por empresa a corpus_composition_internal.csv. C2 en curso.
- 2026-09-26 22:35: C2 hecha. «v3» como nombre del vocabulario fuera de la prosa (§6 ×4, §10 ×2, §11, §13 ×5; se mantiene en rutas/scripts/ESG_RUN_TAG); §13 sin «earlier vocabulary». §12: «$\rho$» corrompido (CR en lugar de \r) reparado; 0,974 → 0,973; «precision» → «error rate» (como §5). Duplicados sustituidos por remisiones: §6 sin cifras de la rejilla de pesos ni de ESGSI–ESGSI_ext (quedan en §10/§11; §6 conserva el resumen cualitativo), §6 sin recuentos de HEDGE (dueño §5), §6 idf remite a tab:robustness-index, §11 sensibilidad de SUS remite a tab:robustness-index/§10 (fuera r/ρ/deciles/p repetidos); §10 cita tab:temporal-trend para las pendientes de referencia. ESGSI con macro en la prosa de §1/§6; «sector» → «industry» donde los resultados son por industria (§1, §6). Parámetros de §6: tokenización de HEDGE alineada con su definición. tab:limitations-summary movida al inicio de §12 (antes flotaba al apéndice). Cifras clave (V=399, 434, 343/49, −0,181/−0,260 y p, 0,865/0,973, ICC, varianzas, borrados por pilar, HEDGE, rendimiento) idénticas en todas las secciones y cotejadas con summary.json. Compila: 0 errores, 0 referencias/citas indefinidas, único overfull 0,46 pt (§5); 55 págs. HTML de estado actualizado.
- 2026-09-26: C3 hecha. Pendientes para el autor en pendiente_secciones_no_tocadas.md. Relanzamiento cancelado.

## Resumen final

Todas las tareas están hechas. El paper (55 páginas, compila sin errores ni
referencias sin resolver) sigue las directrices: score continuo sin etiquetas,
solo el vocabulario actual, sin auditoría ni historial, sin nombres de empresa,
Lagasio como antecedente (§2, §7, §8 y §9 eliminadas y su contenido útil
absorbido en §6), ESGSI como índice principal y Extended ESGSI como métrica
nueva con su rejilla de pesos. Scripts de análisis y estabilidad reescritos con
medidas de score (Spearman, deciles extremos, pendiente). Nada commiteado.
Lo que queda para el autor está en `pendiente_secciones_no_tocadas.md`.
