# Issue #2 (H2): panel gratuito de controversias ESG

Informe de trabajo (octubre de 2026). Es un estudio de viabilidad y un piloto para sustituir RepRisk en H2 con
fuentes gratuitas.

- **H2.** Una mayor divergencia tono-sustancia en t anticipa controversias ESG en t+1 o t+2.

Se siguen las directrices de `paper/_research/directrices.md`:

- el ESGSI, el ESGSI extendido, Z(SEN) y Z(SUS) se usan como scores continuos, sin umbrales ni etiquetas;
- ninguna empresa aparece por su nombre: solo los identificadores anónimos `<Industria>-NN` de `analysis_v3`;
- los términos de búsqueda con nombres y marcas, las descargas en bruto y los eventos con titulares están en
  `data/issue2_controversies/*_internal*`, que no se versiona.

## 0. Resumen

| Pregunta | Respuesta corta |
|---|---|
| ¿Hay un panel gratuito de controversias 2018-2026 para las 49 empresas? | **Sí, pero de calidad intermedia.** La única fuente gratuita con volumen, fecha diaria y cobertura amplia es el **Business & Human Rights Resource Centre (BHRRC)**. Da 1.244 alegaciones fechadas en 2018-2025 sobre 42 de las 49 empresas. Las demás fuentes que funcionan aportan poco: OCDE (quejas ante los PNC), autorregulación publicitaria (JDP en Francia y RCC en Países Bajos) y SBTi. **GDELT** funciona gratis y con fecha diaria. Su API limita tanto (en la práctica, una consulta útil cada 5-10 minutos) que solo se completó la serie de atención mediática total; la de noticias negativas ESG quedó en 2 de 49 empresas. El script es reanudable y, dejado una noche, completaría el panel. |
| ¿Frecuencia? | BHRRC: **3,2 alegaciones por empresa-año** (mediana 1) y el **55 %** de los empresa-año con al menos una. Con solo las alegaciones "focales" (la empresa figura en el titular), 0,9 y el 28 %. Las de severidad alta (acción legal o regulatoria), 0,64 y el 24 %. La distribución está muy concentrada: energía, consumo discrecional y salud concentran la mayoría, y tecnología y servicios financieros casi no tienen. La serie crece mucho con el tiempo (1,3 en 2018 frente a 5,1 en 2024), por la mayor cobertura de la fuente. |
| ¿Potencia con las tasas observadas? | Con la heterogeneidad y la sobredispersión observadas, el MDE es una **IRR de 1,45-1,8 por DT** del ESGSI (Poisson con EF de año) y de **1,28-1,65 con EF de empresa**. El mejor resultado en potencia son los meses con alguna alegación BHRRC (`m_bhrrc`): IRR 1,45 agrupada y 1,28 intra-empresa en t+1. El supuesto teórico de la rama anterior (1,13-1,52) era optimista. Las fuentes raras (publicidad, OCDE) solo detectarían IRR ≥ 2. |
| ¿Modelos H2? | **No hay asociación robusta.** Con el recuento BHRRC en t+1 la IRR por DT del ESGSI es 1,02 (p 0,86) en la especificación agrupada, 1,10 (p 0,34) en la binomial negativa y 1,01 (p 0,93) con EF de empresa. En t+2: 1,03, 1,10 y 0,81 (p 0,75, 0,34 y 0,13). La única pista en la dirección de H2 aparece con el ESGSI extendido y las alegaciones de severidad alta (acción legal o regulatoria) en t+2: IRR 1,42 (p 0,05) agrupada y 1,58 (p 0,005) NB, y 1,50-1,63 en la ventana posterior a la publicación. Procede de una menor densidad de contenido ESG (Z(SUS)). No se mantiene con EF de empresa (1,14; p 0,58). En el **placebo** (t-1) hay más significaciones que en t+1 y t+2, todas con IRR < 1: 11 de 120, frente a 1 y 7. |
| ¿Sirve para el paper? | **Como análisis exploratorio o de robustez, sí; como contraste principal de H2, no.** Véase §6. |

## 1. Fuentes evaluadas

Criterios: acceso gratuito sin registro ni credenciales, respeto de `robots.txt` y de los límites de uso, y ninguna
elusión de protecciones anti-bot. Los retos de Cloudflare se registran como "no accesible".

| Fuente | Contenido | Acceso | Cobertura de las 49 | Años | Fecha del evento | Severidad | Estado |
|---|---|---|---|---|---|---|---|
| **BHRRC** (business-humanrights.org) | noticias, alegaciones, demandas y respuestas de empresas sobre derechos humanos y medio ambiente, etiquetadas por empresa | gratis; la propia web usa una API JSON pública sin token; Crawl-Delay 5 s respetado | 48 con id (1 sin ficha); 13.289 ítems; **42 con alegaciones en 2018-2025** | 2001-2026; útil desde 2018 | día de publicación o del incidente (`backdate`) | marca `contains_allegations`, tipo (Lawsuit...) y palabras clave del titular | **usada: fuente principal** |
| **OECD Watch** (base de quejas ante los PNC de la OCDE) | quejas por incumplimiento de las Líneas Directrices | gratis, HTML con sitemap | 495 quejas en total; **12 en 2018-2026 sobre 10 empresas** | 2000-2026 | día de presentación | aceptada o con acuerdo frente a solo presentada | usada; muy escasa |
| Base oficial de la OCDE (mneguidelines.oecd.org) | igual | reto de Cloudflare | — | — | — | — | no accesible |
| **JDP** (ARPP, Francia) | resoluciones del Jury de Déontologie Publicitaire | gratis, listado HTML | 1.104 resoluciones (2008-2026); **19 sobre 9 empresas o marcas**, todas desde 2022 | 2008-2026 | día de la resolución | queja estimada o desestimada | usada |
| **RCC** (Reclame Code Commissie, Países Bajos) | resoluciones de publicidad | gratis, buscador HTML | 696 resoluciones con términos ambientales desde 2017; **37 sobre 5 empresas o marcas** (concentradas en una sola empresa) | 2017-2026 | día de la resolución | Aanbeveling (estimada) o Afwijzing | usada |
| Autocontrol (ES), IAP (IT), Werberat (DE), JEP (BE), consejo finlandés | resoluciones publicitarias | IAP y Werberat accesibles; JEP con captcha; Autocontrol con buscador sin campo de anunciante | — | — | — | — | no automatizadas: volumen esperado muy bajo |
| **SBTi Target Dashboard** | estado de compromisos climáticos | descarga Excel gratuita | 39 de 49 presentes; 3 con compromiso a corto plazo retirado o caducado y 3 con el de cero neto | solo foto actual | **sin fecha del cambio de estado** | — | solo descriptivo (`sbti_estado.csv`) |
| **GDELT DOC 2.0 API** | nº diario de artículos que casan con «empresa AND términos de incidente ESG» | gratis, sin registro; límite de 1 consulta cada 5 s, y en la práctica HTTP 429 muy frecuentes | ALL (atención total): 48 de 49 (falta 1 por un término demasiado corto, ya corregido). NEG (incidentes ESG): **2 de 49** | 2017-2026 (resolución diaria en una sola consulta) | día | ninguna (volumen) | **funciona, pero no se completó**: unos 10 minutos por consulta útil, con unas 300 consultas para 6 temas × 49 empresas. Se usa ALL como atención mediática y en la validación. Además, la cobertura del índice cae a una cuarta parte entre 2017 y 2025 (mediana 23.000 frente a 5.700 artículos por empresa-año), así que exige EF de año |
| GDELT GKG en bruto (data.gdeltproject.org) | ficheros cada 15 minutos con organizaciones y temas | gratis | probado: unos 8 MB por fichero, unos 26 artículos por fichero que mencionan alguna de las 49 | 2015- | día | temas | **inviable completo** (unos 245.000 ficheros y 2 TB en 2018-2024); con un fichero al día serían unos 27 GB y una muestra del 1 % de los artículos |
| Sabin Center / Climate Policy Radar (climatecasechart.com) | litigios climáticos | la descarga completa exige un formulario; la API de la web requiere token y la exportación por búsqueda está "temporalmente no disponible" | — | — | — | — | no usada (pedir la descarga por formulario) |
| Violation Tracker Global | sanciones por matriz | Cloudflare; descarga de pago | — | — | — | — | no accesible |
| Reguladores: AMF, BaFin, CNMV, Consob, AFM, FSMA, BCE (MUS), DGCCRF, AGCM, ACM; multas de competencia de la UE | sanciones | Consob y AFM accesibles; CNMV y DGCCRF con 403 o Cloudflare; AGCM y ACM sin conexión; la lista del BCE se carga por JavaScript; el portal de competencia de la UE es una SPA | — | — | — | — | no automatizados: pocas sanciones a estos emisores; codificación manual factible (p. ej. unas 40 decisiones del BCE sobre los 7 bancos) |
| Wikipedia / Wikidata | secciones de controversias | gratis | — | — | fechas imprecisas, sesgo de edición | — | descartada |
| Banktrack, InfluenceMap, Global Witness | perfiles o puntuaciones | gratis | solo bancos o puntuación actual | — | sin panel fechado | — | descartadas para H2 |

### Emparejamiento y definición de evento

- **BHRRC**:
  - Ids de empresa revisados a mano: matriz y filiales consolidadas en 2018-2024. Se excluyen homónimos,
    concesionarios y negocios escindidos.
  - Es **evento** un ítem con `contains_allegations`, de tipo demanda o cuyo titular o resumen describe un incidente.
    Se excluyen respuestas de la empresa, informes, benchmarks y fichas de divulgación.
  - Variantes:
    - `n_bhrrc`: todas las alegaciones.
    - `m_bhrrc`: meses distintos con alguna alegación. Es robusta a la cobertura repetida de un mismo incidente.
    - `n_bhrrcf`: alegaciones **focales**, en las que la empresa o una de sus marcas figura en el titular. Excluye los
      informes que citan a decenas de empresas, que son la principal fuente de ruido.
    - `n_bhrrc_sev2`: **severidad alta**, es decir, demanda, denuncia, multa, condena, sanción o tribunal.
    - `n_bhrrc_E`, `_S`, `_G` y `_GW`: categorías E, S, G y greenwashing, por palabras clave.
- **Revisión manual de una muestra**: la precisión es aceptable en las alegaciones focales y baja en las no
  focales, porque muchas son informes sobre sectores enteros. Las categorías por palabras clave son
  orientativas, porque un ítem puede tener varias.
- **Pool**: `n_pool` = BHRRC + OCDE + publicidad, y `n_poolf` = BHRRC focal + OCDE + publicidad. Las quejas paralelas
  presentadas el mismo día ante varios PNC cuentan como un solo evento.

## 2. Frecuencia (2018-2025, 49 empresas × 8 años = 392 empresa-año)

| Recuento | Media por empresa-año | Mediana | % con ≥ 1 | Varianza / media | Empresas con ≥ 1 | Parte de la varianza entre empresas |
|---|---|---|---|---|---|---|
| `n_bhrrc` (todas las alegaciones) | 3,17 | 1 | 54,8 % | 17,4 | 42 | 0,61 |
| `m_bhrrc` (meses con alegación) | 1,77 | 1 | 54,8 % | 3,5 | 42 | 0,69 |
| `n_bhrrcf` (focales) | 0,89 | 0 | 28,1 % | 6,9 | 30 | 0,50 |
| `n_bhrrc_sev2` (acción legal o regulatoria) | 0,64 | 0 | 24,0 % | 5,3 | 32 | 0,65 |
| `n_bhrrc_E` (ambientales) | 0,92 | 0 | 32,1 % | 5,5 | 32 | 0,63 |
| `n_bhrrc_GW` (greenwashing) | 0,08 | 0 | 4,3 % | 2,6 | 10 | 0,38 |
| `n_oecd` (quejas PNC) | 0,03 | 0 | 2,8 % | 1,1 | 10 | 0,12 |
| `n_ads` (JDP + RCC) | 0,12 | 0 | 6,6 % | 2,6 | 11 | 0,59 |
| `n_pool_GW` (greenwashing, todas las fuentes) | 0,15 | 0 | 8,7 % | 2,5 | 15 | 0,43 |
| `n_pool` | 3,33 | 1 | 56,1 % | 16,9 | 42 | 0,62 |
| `n_poolf` (focales + OCDE + publicidad) | 1,04 | 0 | 31,6 % | 6,6 | 33 | 0,50 |
| `gd_all` (GDELT, artículos que mencionan la empresa) | 32.835 | 11.858 | 100 % | — | 48 | 0,67 |

**Por industria** (media de `n_bhrrc` por empresa-año; entre paréntesis, la de focales):

| Industria | Empresas | `n_bhrrc` (focales) | Severidad alta | Publicidad | % de empresa-año con alguna alegación |
|---|---|---|---|---|---|
| Energy | 2 | 16,7 (5,8) | 5,9 | 0,19 | 100 % |
| Consumer Discretionary | 10 | 7,0 (1,7) | 0,7 | 0,31 | 68 % |
| Health Care | 3 | 2,8 (1,7) | 1,3 | 0,04 | 42 % |
| Basic Materials | 2 | 2,1 (0,3) | 0,3 | 0 | 50 % |
| Consumer Staples | 5 | 1,6 (0,4) | 0,2 | 0,65 | 60 % |
| Financials | 11 | 1,6 (0,3) | 0,4 | 0 | 53 % |
| Telecommunications | 2 | 1,1 (0,3) | 0,1 | 0 | 63 % |
| Utilities | 2 | 1,1 (0,4) | 0,4 | 0 | 63 % |
| Industrials | 9 | 1,0 (0,2) | 0,3 | 0,01 | 44 % |
| Technology | 3 | 0,4 (0,0) | 0,1 | 0 | 17 % |

**Por país**: España 7,6 (lo explica una sola empresa, por alegaciones en su cadena de suministro),
Alemania 3,7, Francia 3,2, Italia 1,6, Países Bajos 1,6, Finlandia 0,9 y Bélgica 0,8. Las resoluciones publicitarias se
concentran en Países Bajos (0,65 por empresa-año) por el buscador de la RCC.

**Por año**:

| Recuento | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|
| `n_bhrrc` | 1,27 | 1,41 | 1,92 | 2,45 | 3,55 | 4,76 | 5,10 | 4,94 |
| `n_bhrrcf` | 0,41 | 0,33 | 0,41 | 0,63 | 0,96 | 1,71 | 1,20 | 1,45 |
| `n_bhrrc_sev2` | 0,39 | 0,39 | 0,47 | 0,41 | 0,41 | 1,12 | 1,00 | 0,94 |

La serie se multiplica por cuatro: refleja sobre todo la mayor actividad del BHRRC y no más controversias. Los
**efectos fijos de año son imprescindibles**.

**Validez convergente** (`correlaciones_fuentes.csv`):

- Entre empresas, el total de alegaciones BHRRC 2018-2025 correlaciona con:
  - la atención mediática total en GDELT: Spearman 0,55, y 0,62 con las focales;
  - las resoluciones publicitarias: 0,47;
  - las quejas ante la OCDE: 0,34.
- Dentro de la empresa y el año (log1p sin EF de empresa ni de año), las correlaciones entre fuentes son casi nulas
  (≤ 0,18).

Las fuentes coinciden en *qué* empresas son controvertidas, pero apenas en *cuándo*. Es justo la variación que
necesita la especificación intra-empresa.

## 3. Potencia con las tasas observadas

`ctrv_power.py` reutiliza la lógica de `fin_power.py`: el ESGSI es el real y solo se simula el resultado. Cada
recuento se calibra a sus momentos observados:

- la media λ;
- la heterogeneidad entre empresas σ_u, por momentos;
- la sobredispersión intra-empresa α (NB2).

Se usan 200 réplicas por punto, SE agrupados por empresa, un contraste bilateral al 5 % y una potencia del 80 %.
Las especificaciones son (A) Poisson con EF de año y (B) Poisson con EF de empresa y año.

| Recuento | λ (media) | σ_u | α intra | MDE IRR, t+1 (A / B) | MDE IRR, t+2 (A / B) |
|---|---|---|---|---|---|
| `n_bhrrc` | 3,17 | 1,19 | 0,48 | 1,64 / 1,45 | 1,68 / 1,62 |
| `m_bhrrc` (meses con alegación) | 1,77 | 0,92 | 0,06 | **1,45 / 1,28** | 1,46 / 1,40 |
| `n_bhrrcf` (focales) | 0,89 | 1,21 | 0,68 | 1,67 / 1,65 | 1,83 / 2,00 |
| `n_bhrrc_sev2` (acción legal) | 0,64 | 1,34 | 0,26 | 1,78 / 1,59 | 1,84 / 1,88 |
| `n_ads` (publicidad) | 0,14 | 1,53 | 0,05 | 2,03 / 2,09 | 2,17 / no se alcanza |
| `n_oecd` (PNC) | 0,03 | ≈ 0 | 0,86 | ≈ 2 (con unos 10 eventos: poco fiable) / no se alcanza | — |

**Lectura.**

- La heterogeneidad observada entre empresas es grande: σ_u ≈ 1,2, es decir, alrededor del 60 % de la varianza
  entre empresas.
- Con la sobredispersión observada, el MDE pasa del 1,13-1,52 del supuesto teórico (media 2, rama anterior) a una
  **IRR de 1,45-1,8 por DT** del ESGSI. Equivale a un 45-80 % más de controversias por cada DT de divergencia, un efecto
  grande.
- El resultado más potente es `m_bhrrc`, los meses con alguna alegación. Recorta la sobredispersión que generan los
  incidentes muy cubiertos, y con EF de empresa detecta una **IRR de 1,28** en t+1.
- Las fuentes raras (publicidad, OCDE) solo detectan efectos de IRR ≥ 2: no sirven como resultado por separado.

## 4. Modelos exploratorios de H2

Diseño (`ctrv_models.py`, resultados completos en `modelos_h2.csv`):

- **Muestra**: el informe del ejercicio t (2018-2024) frente al recuento del año natural t+h, solo con años
  completos (≤ 2025). Con h = +1 hay 343 informes; con h = +2, 294; con el placebo h = −1, 343 (recuentos 2017-2023).
- **Especificaciones**:
  - **A**: Poisson (cuasi-verosimilitud) con EF de año, industria y país. Controles: log del tamaño (capitalización
    aproximada al cierre de t, de la rama anterior) y log(1 + recuento en t).
  - **NB**: binomial negativa (NB2) con las mismas variables.
  - **B**: Poisson con EF de empresa y año y control de tamaño. Excluye las empresas sin ningún evento en la muestra.
  - Las celdas de efectos fijos sin eventos se eliminan (separación).
- **Inferencia**: SE agrupados por empresa (CR1) con t de 48 grados de libertad. La IRR es el efecto de 1 DT del
  regresor.

| Resultado | Regresor | h | A: IRR (p) | NB: IRR (p) | B, EF de empresa: IRR (p) |
|---|---|---|---|---|---|
| `n_bhrrc` | ESGSI | +1 | 1,02 (0,86) | 1,10 (0,34) | 1,01 (0,93) |
| | | +2 | 1,03 (0,75) | 1,10 (0,34) | 0,81 (0,13) |
| | | −1 (placebo) | 0,95 (0,58) | 0,90 (0,30) | 1,03 (0,83) |
| | ESGSI ext. | +1 | 1,04 (0,76) | 1,15 (0,20) | 1,02 (0,86) |
| | | +2 | 1,00 (0,98) | 1,08 (0,57) | 0,85 (0,22) |
| | | −1 | 1,07 (0,49) | 1,01 (0,94) | 1,25 (0,27) |
| `n_bhrrcf` (focales) | ESGSI | +1 | 0,95 (0,65) | 0,90 (0,50) | 0,94 (0,76) |
| | | +2 | 1,04 (0,78) | 1,01 (0,94) | 0,81 (0,50) |
| | | −1 | 0,84 (0,23) | **0,68 (0,03)** | 1,10 (0,69) |
| `n_bhrrc_sev2` (acción legal) | ESGSI | +1 | 1,06 (0,71) | 1,09 (0,58) | 0,90 (0,68) |
| | | +2 | 1,18 (0,30) | 1,25 (0,15) | 0,96 (0,88) |
| | ESGSI ext. | +1 | 1,25 (0,16) | 1,32 (0,10) | 1,05 (0,84) |
| | | +2 | **1,42 (0,05)** | **1,58 (0,005)** | 1,14 (0,58) |
| | | −1 | 1,03 (0,86) | 1,05 (0,81) | 1,29 (0,33) |
| `n_pool` | ESGSI | +1 | 1,01 (0,89) | 1,08 (0,41) | 1,03 (0,78) |
| | | +2 | 1,00 (0,98) | 1,07 (0,51) | 0,82 (0,13) |
| `n_bhrrc` | Z(SEN) / Z(SUS) | +1 | 1,03 / 1,01 | 1,06 / 0,92 | 1,09 / 1,08 |
| | | +2 | 1,03 / 0,99 | 1,01 / 0,87 | 0,90 / 1,21 |
| `n_bhrrc_w` (ventana abr-(t+1) a mar-(t+2)) | ESGSI | +1 | 1,07 (0,45) | 1,17 (0,09) | 1,05 (0,73) |
| `n_bhrrc_sev2_w` (ídem, acción legal) | ESGSI ext. | +1 | 1,33 (0,08) | **1,43 (0,03)** | 1,12 (0,68) |
| | | +2 | **1,50 (0,02)** | **1,63 (0,005)** | 1,15 (0,62) |
| | Z(SUS) | +2 | **0,70 (0,04)** | **0,66 (0,01)** | 0,93 (0,82) |
| | Z(SEN) | −1 (placebo) | **0,77 (0,05)** | **0,74 (0,05)** | 0,79 (0,36) |
| `m_bhrrc` (meses con alegación) | ESGSI | +1 | 1,07 (0,35) | 1,08 (0,34) | 1,00 (0,97) |
| | ESGSI ext. | +1 | 1,13 (0,20) | 1,13 (0,20) | 1,04 (0,64) |

**Lectura.**

- Ninguna asociación entre el score de t y las controversias de t+1 o t+2 es robusta. Las estimaciones puntuales
  rondan 1 (0,8-1,15 por DT) y los intervalos al 95 % son amplios (p. ej. `n_bhrrc`, ESGSI, t+1, A: 0,85-1,22).
- **Recuento de significaciones nominales**: de 120 contrastes por horizonte (10 recuentos × 3 especificaciones × 4 regresores), hay 1 con
  p < 0,05 en t+1, 7 en t+2 y 11 en el **placebo t-1**. Por azar se esperarían unas 6 en cada horizonte. Que el placebo dé
  más significaciones (todas con IRR < 1) indica componentes persistentes de la empresa o ruido de muestra
  pequeña, no un efecto de la divergencia.
- La única pista en la dirección de H2 es la del **ESGSI extendido con las alegaciones de severidad alta** (demandas,
  denuncias, multas): IRR 1,42-1,58 por DT en t+2, y 1,50-1,63 en la ventana posterior a la publicación. Viene de
  Z(SUS): los informes con menos densidad de contenido ESG anticipan más acciones legales (IRR 0,66-0,70 por DT). Desaparece con EF de empresa (1,14; p 0,58), así que viene de
  diferencias entre empresas, no de cambios dentro de cada una. Además no sobrevive a ninguna corrección por
  comparaciones múltiples. Conviene tratarla como hipótesis para una fuente mejor (RepRisk) y no como hallazgo.
- Fuera de esas acciones legales, los componentes no dan señal: ni el tono (Z(SEN)) ni la sustancia (Z(SUS))
  anticipan de forma estable el total de alegaciones. El tono aparece sobre todo en el placebo, con IRR < 1.
- El recuento de severidad alta y la ventana `*_w` se añadieron durante la exploración y no se fijaron de antemano.
  Cualquier uso en el paper debería prerregistrarse con datos nuevos (2026-2027) o con RepRisk.

## 5. Limitaciones

1. **Ruido de medida.** El BHRRC no deduplica incidentes ni mide la novedad ni el alcance, como sí hace RepRisk. Las
   alegaciones no focales suelen ser informes sobre muchas empresas. La categoría E/S/G/GW sale de palabras clave.
2. **Sesgo de cobertura.** El BHRRC se centra en derechos humanos y en los sectores con cadenas de suministro o
   proyectos extractivos: energía, textil, automóvil y alimentación. Infra-representa los problemas de gobierno
   corporativo (G: 0,11 por empresa-año) y a las financieras y tecnológicas. Su actividad se cuadruplica en
   2018-2025.
3. **Greenwashing propiamente dicho.** Es raro en todas las fuentes gratuitas: 0,15 por empresa-año y el 9 % de los
   empresa-año, sumando BHRRC, JDP y RCC. Ninguna serie gratuita permite un contraste específico de greenwashing.
4. **La publicidad depende del país.** Solo hay archivos rastreables en Francia (JDP) y Países Bajos (RCC). El
   emparejamiento por marcas añade falsos positivos (revisión parcial).
5. **Fechas.** BHRRC y GDELT tienen fecha diaria; la OCDE, fecha de presentación; las resoluciones, fecha de
   resolución, que llega con retraso respecto a la campaña. Se usa el año natural y, como variante (`*_w`), la ventana
   posterior a la publicación del informe (1-abr-(t+1) a 31-mar-(t+2)), como en `issue2_finance`.
6. **GDELT incompleto.** Solo está la atención total. La serie de incidentes ESG por noticias, que sería el
   análogo más cercano a RepRisk, requiere unas 8 horas más de descarga con el límite actual de la API.

## 6. ¿Es creíble para el paper?

- **Como contraste principal de H2, no.** La fuente útil es una sola (BHRRC). Es temática, tiene una tendencia de
  cobertura fuerte y no está deduplicada. Además, las fuentes apenas coinciden dentro de cada empresa, que es la
  variación que identifica H2. Un evaluador pediría RepRisk o una base equivalente.
- **Como análisis exploratorio o de validez externa, sí**, con tres condiciones:
  1. definir ex ante un único resultado principal (propuesta: `n_bhrrc_sev2` o `m_bhrrc`, con EF de año);
  2. publicar el placebo t-1;
  3. presentarlo como límite inferior de lo que permitiría una fuente comercial.
- **Mensaje honesto con los datos actuales**: no hay evidencia de que la divergencia anticipe controversias, con un
  MDE en torno a una IRR de 1,3-1,8 por DT. El resultado es compatible con efectos moderados que este panel no puede detectar.
- **Siguiente paso recomendado**:
  1. RepRisk vía WRDS, que permite incidentes deduplicados con severidad y novedad;
  2. mientras tanto, pedir la descarga completa del Sabin Center (formulario gratuito) para añadir los litigios
     climáticos;
  3. codificar a mano las sanciones del BCE (bancos) y de las autoridades de consumo (AGCM, ACM y DGCCRF) sobre
     alegaciones ambientales.

## 7. Reproducción y ficheros

```
python -I scripts/ctrv_bhrrc.py        # BHRRC (unos 40 min, pausa de 5 s, con caché)
python -I scripts/ctrv_oecdwatch.py    # quejas PNC (OECD Watch)
python -I scripts/ctrv_adrulings.py    # JDP + RCC
python -I scripts/ctrv_sbti.py         # estado SBTi (descriptivo)
python    scripts/ctrv_gdelt.py --topics ALL,NEG   # GDELT DOC API (horas por los HTTP 429; reanudable)
python -I scripts/ctrv_panel.py        # panel empresa-año y frecuencias
python -I scripts/ctrv_power.py        # potencia con tasas observadas
python -I scripts/ctrv_models.py       # modelos H2 de recuento y placebo
```

Requiere el fichero interno `data/issue2_controversies/firm_terms_internal.csv`, que no se versiona. Sus columnas son
`firm_id; gdelt; regex; name; bhrrc_ids; brands`, con los términos de búsqueda de cada empresa. También requiere
`data/issue2_controversies/panel_mercado_internal.csv`, el tamaño de la rama `issue-2-financial-consequences`.

| Fichero (anónimo) | Contenido |
|---|---|
| `panel_controversias.csv` | panel empresa × año (2017-2026) con todos los recuentos, la industria y el país; 2026 es parcial (`meses_obs`) |
| `frecuencias_fuente.csv`, `frecuencias_sector.csv`, `frecuencias_pais.csv`, `frecuencias_ano.csv` | descriptivos |
| `correlaciones_fuentes.csv` | validez convergente entre fuentes |
| `sbti_estado.csv` | estado actual de los compromisos SBTi |
| `potencia_recuento_observado.csv`, `potencia_resumen.json` | potencia con tasas observadas |
| `modelos_h2.csv`, `modelos_h2_resumen.json` | todos los coeficientes (IRR, IC, p, n) |
