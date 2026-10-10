# Tareas pendientes del autor

Lista de lo que solo puedes hacer o decidir tú para que pueda seguir con el paper. Está ordenada por urgencia. Cada tarea indica de dónde viene (issue o sección).

Los identificadores anónimos (`Financials-04`, etc.) son los del análisis **anterior** a las correcciones: se recalculan con cada ejecución y pueden cambiar. La correspondencia con las empresas está en `results/_pre_fix_20261010/analysis_v3/firm_ids_internal.csv`, que no se versiona. Este fichero se publica en el repositorio, así que no lleva nombres de empresa.

---

## 1. Bloquea la reejecución definitiva

- [x] **Sustituir el informe 2024 de Financials-04**, que era de otra empresa. Hecho el 10-10-2026: el informe anual 2024 correcto (352 páginas) ya está en el corpus y la ejecución completa con los 343 informes está hecha.

## 2. Datos que faltan en el texto (marcas `[Author: …]` del paper)

- [x] **Muestra** (§2, `paper/sections/03_data.tex`):
  - [x] Índice: EURO STOXX 50, composición de 2024. El día exacto no hace falta.
  - [x] Empresa excluida: la única constituyente sin informes de 2018 ni de 2019.
  - [x] Empresa surgida de la fusión de 2021: se mantienen las dos predecesoras (opción a). Se explica en §2.
  - [x] Idioma: no hace falta precisar qué documentos son traducciones.
- [ ] **Revisores del vocabulario** (§3.2 y §6):
  - Quiénes son, cuántos y con qué perfil.
  - Si alguna fase (lectura de informes o evaluación de candidatos) la hizo un modelo de lenguaje. Si fue así, el κ se presenta como consistencia entre ejecuciones, no como fiabilidad entre personas.
  - Si la admisión fue por consenso o por valoraciones independientes. Hay un κ de Fleiss de 0,81 sobre 532 candidatos calculado de las rondas de revisión.
- [x] **Precisión del ~82 %** (§3.2): opción (a). La frase lleva ahora la muestra (3.401 concordancias de 430 entradas) y el intervalo al 95 % (80–84 %). El cálculo está versionado en `scripts/vocab_v3_stats.py`.
- [ ] **Disponibilidad de datos** (§6): URL del repositorio o del depósito anonimizado (OSF o Zenodo con DOI). Decide también qué se redistribuye: URL y hashes de los PDF, recuentos y scores.

## 3. Decisiones de método (issue #3, síntesis)

- [x] **Confirmar las tres medidas corregidas.** Adoptadas, reejecutadas y llevadas al paper en la rama `fix/corpus-and-measures`, con la ruptura de 2024 explicada en §4.1 y §4.2:
  - SEN por forma exacta de Loughran–McDonald;
  - QUANT V2b;
  - HEDGE con Uncertainty + Weak Modal, sin *risk/risks* y sin *May* como mes.
- [ ] **QUANT V2b, detalles abiertos del issue #1:**
  - ¿Cuentan las celdas `0%` de las plantillas de la Taxonomía? Ahora sí cuentan.
  - ¿Deben contar las cifras con separador de miles (`14,918`)? Ahora no cuentan.
- [ ] **Validación con codificación humana:** ¿se hace? Serían unos 300 pasajes, 2 codificadores, 4–5 días-persona y como mucho una página de apéndice.
- [ ] **Diccionario ESG externo** como validez convergente de SUS: ¿se incluye? Roza la directriz 2.
- [x] **Límite de páginas:** el límite de 15 páginas del núcleo técnico es blando. Ahora ocupa unas 15,5, y no hay que recortar mientras no crezca mucho más.
- [ ] **Título:** ¿se acota el alcance? Por ejemplo, grandes emisores cotizados, o "densidad de contenido ESG" en lugar de "sustancia".
- [ ] **ClimateBERT:** después de la reejecución, ¿entra la media página de validez convergente en §5? El borrador está en la rama `issue-2-climatebert`. Antes de citarlo, conviene leer completo Bingler et al. (2022, 2024) para confirmar la definición del índice de *cheap talk*.
- [ ] **Sesgo del extractor** hacia `taxonomy`, CO2 y GHG (issue #1, pregunta 5): ¿se menciona en limitaciones?

## 4. Corpus: anomalías que quedan por decidir

Los cuatro informes de 2024 ya están corregidos. Quedan tres casos de 2018 con un documento atípico (issue #2, trabajo de base). ¿Se sustituyen?

- [ ] Utilities-01, 2018: las cuentas consolidadas con informe de auditoría (830 páginas).
- [ ] Health Care-03, 2018: solo la DPEF (28 páginas).
- [ ] Financials-08, 2018: 179 páginas, frente a una mediana de 546.

## 5. Consecuencias financieras (issue #2)

- [ ] ¿Tiene la universidad acceso a **WRDS** o a **LSEG**? Si lo tiene, H1 (I/B/E/S) y H2 (RepRisk) se pueden contrastar. Los ISIN y LEI ya están preparados.
- [ ] Definición de **controversia** (recuento, índice de RepRisk o binaria), **horizonte** de H2 (t+1, t+2) y año final de los datos.
- [ ] **Dispersión de analistas:** sobre la media o sobre el precio, y en qué fecha se mide.
- [ ] ¿Merece la pena pagar **Violation Tracker Global** como criterio complementario?
- [ ] ¿Pedir la descarga del **Sabin Center**? Es gratis, pero requiere un formulario.

## 6. Secciones que escribes tú

- [ ] **Introducción:** pregunta económica y por qué la medida es útil aunque baje cuando una empresa cumple un mandato. Hay sugerencias en las respuestas a R1.20 y R2.22 del issue #3.
- [ ] **Literatura:** tres líneas.
  - Tono anormal en finanzas.
  - Efectos de la divulgación ESG obligatoria.
  - Medidas textuales de greenwashing o *cheap talk*.

  Hay 36 referencias verificadas en `results/issue2_lit/refs.bib` de la rama `issue-2-lit-review`. Faltan Huang, Teoh y Zhang (2014), Davis, Piger y Sedor (2012), Brown y Tucker (2011), Lang y Stice-Lawrence (2015), Dyer et al. (2017) y Hope et al. (2016).
- [ ] **Discusión** y **conclusión**.
- [ ] **Resumen.**

## 7. Repositorio

- [ ] **Commits y push.** Todo está solo en local:
  - la rama `fix/corpus-and-measures`, con los cambios de la Tabla 3 y estas correcciones;
  - las ramas `issue-1-quant-trend`, `issue-2-*` y `issue-2-climatebert`.

  ¿Hago commit y subo alguna?
- [ ] **Nombres de empresa en un repo público.** `metadata/empresas_supersector.csv` (ya publicado) y `metadata/empresas_tickers.csv` y `metadata/mandates.csv` (en ramas locales) llevan nombres. ¿Se mantienen públicos o pasan a un fichero no versionado?
- [ ] **Carpeta antigua `data/chunks_lexical/`:** una ejecución mal configurada sobrescribió 23 de sus JSON. El paper no la usa. ¿Se regenera o se borra?
