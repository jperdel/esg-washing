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
- [x] **QUANT V2c** (issue #6, rama `quant-v2c`), decidido el 12-10-2026:
  - Las celdas `0%` de las plantillas de la Taxonomía ya no cuentan.
  - Las unidades solo cuentan si llevan una cifra delante. Se añaden los plurales (*tonnes*, *litres*…).
  - Las cifras con separador de miles (`14,918`) siguen sin contar: son sobre todo tablas financieras y de remuneración (solo un 30 % son métricas ESG).
  - Análisis completo reejecutado y cifras del paper actualizadas (§3–§5, Tablas 3–5 y figuras). Solo cambian QUANT y el ESGSI extendido, y poco: pendiente del extendido de −0,255 a −0,244.
- [x] **UNBACK sustituye a HEDGE** (issue #7, rama `unback`): compromisos ESG sin cantidad, plazo ni año base, con reglas. Integrado en el pipeline y en el paper (§3.3 con tabla de ejemplos, §4, §5, §6). La última versión con HEDGE queda en la etiqueta `paper-hedge`.
- [ ] **Precisión de UNBACK (marca `[Author: …]` en §3.3):** revisar una muestra de unas 100 frases de compromiso detectadas (¿es un compromiso ESG? ¿está bien marcado el respaldo?) para dar una precisión con intervalo. Te preparo la hoja cuando quieras.
- [ ] **Extensión:** con UNBACK el paper pasa a 19 páginas; las limitaciones empiezan en la 17. Decidir si se recorta.
- [ ] **Validación con codificación humana:** ¿se hace? Serían unos 300 pasajes, 2 codificadores, 4–5 días-persona y como mucho una página de apéndice.
- [ ] **Diccionario ESG externo** como validez convergente de SUS: ¿se incluye? Roza la directriz 2.
- [x] **Límite de páginas:** el límite de 15 páginas del núcleo técnico es blando. Con ClimateBERT y los modelos del #5 ocupa unas 16,5 (limitaciones en la 16, bibliografía en la 17). No hay que recortar mientras no crezca mucho más.
- [ ] **Título:** ¿se acota el alcance? Por ejemplo, grandes emisores cotizados, o "densidad de contenido ESG" en lugar de "sustancia".
- [x] **ClimateBERT:** integrado en §5, con el contraste de 2024 en §4.1 y el matiz sobre HEDGE en §3.3 (issue #4, rama `climatebert`).
- [x] **Otros modelos ESG** (FinBERT, ESGBERT, NetZero): integrados en §5, con el recall de la extracción en §3.1, la lectura de 2024 matizada en §4.1 y el ajuste de §6 (issue #5, rama `esg-models`).
- [ ] **Lectura de 2024:** ahora el paper dice que la caída de SEN es vocabulario de las tablas ESRS, pero que FinBERT y ClimateBERT ven además un texto narrativo menos positivo que SEN no capta. Revisa que la introducción y las conclusiones que escribas no digan que el tono no cambió.
- [ ] **Licencia de FinBERT:** los modelos de FinBERT no declaran licencia en Hugging Face (el código es Apache-2.0 y la ficha pide citar a Huang et al., 2023). ¿Se menciona en la disponibilidad de datos y código?
- [ ] **Lecturas antes del envío** (issues #4 y #5):
  - Bingler et al. (2024) completo. El índice de *cheap talk* es de ese artículo, no del de 2022 (ya corregido en §5); la definición y el α = 0,17 están comprobados por un agente en el texto completo (issue #7).
  - Bax, Paterlini y Valentini (2026, *Economics Letters*, 10.1016/j.econlet.2026.112835), que encuentra el mismo salto del *cheap talk* con la CSRD en el EURO STOXX 50. Solo leído el resumen; si se confirma, conviene citarlo.
  - Huang, Wang y Yang (2023, FinBERT) y Schimanski et al. (2024, ESGBERT), citados en §3.1 y §5.
- [x] **Sesgo del extractor** (issue #6): nueva limitación en §6 ("Selection by the scoring vocabulary") y una frase en §3.1. Alrededor de una décima parte de la subida del texto extraído (31,6 % → 49,6 %) se debe a las entradas transversales: sin ellas la subida sería 30,2 % → 46,4 %.

## 4. Corpus: anomalías que quedan por decidir

Los cuatro informes de 2024 ya están corregidos. Quedan tres casos de 2018 con un documento atípico (issue #2, trabajo de base). ¿Se sustituyen?

- [ ] Utilities-01, 2018: las cuentas consolidadas con informe de auditoría (830 páginas).
- [ ] Health Care-03, 2018: solo la DPEF (28 páginas).
- [ ] Financials-08, 2018: 179 páginas, frente a una mediana de 546.

## 5. Consecuencias financieras (issue #2): aplazado

Cerrado el 11-10-2026. Con datos gratuitos y 49 empresas no se puede contrastar H1 ni H2, así que queda para trabajos futuros. Si se retoma, el punto de partida está en los comentarios del issue #2 y en las ramas locales `issue-2-*`: ISIN y LEI, calendario de mandatos, pipeline validado con datos simulados y pilotos nulos con su efecto mínimo detectable.

Preguntas para entonces:
- ¿Hay acceso a **WRDS** o **LSEG** (I/B/E/S, RepRisk)?
- ¿Cómo se define la **controversia**, con qué **horizonte** y cómo se mide la **dispersión de analistas**?
- ¿Merece la pena una fuente de pago (Violation Tracker Global) o pedir la descarga del Sabin Center?

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

- [x] **Commits y push** de `fix/corpus-and-measures` y `climatebert`: hecho el 10-10-2026.
- [x] **Ramas de los issues** (`issue-1-quant-trend` y las `issue-2-*`): subidas a origin el 11-10-2026, sin merge. `issue-2-climatebert` se borró el 11-10-2026 porque la sustituye `climatebert`. Su material es exploratorio; lo que llegó al paper ya está en `main`.
- [x] **Merge a `main`:** hecho el 10-10-2026 por fast-forward de `main` hasta `climatebert`, que incluye `fix/corpus-and-measures`.
- [x] **Merge de `esg-models` a `main`:** hecho el 11-10-2026 por fast-forward (issue #5).
- [x] **Nombres de empresa en un repo público:** pueden quedarse en los metadatos y datos del repositorio. Lo que se evita es centrar el análisis o el paper en empresas concretas.
- [ ] **Carpeta antigua `data/chunks_lexical/`:** una ejecución mal configurada sobrescribió 23 de sus JSON. El paper no la usa. ¿Se regenera o se borra?
