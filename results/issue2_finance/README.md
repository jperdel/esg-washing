# Issue #2: consecuencias financieras de la divergencia tono-sustancia

Informe de trabajo (octubre de 2026). Explora si la divergencia entre tono y sustancia,
medida con el ESGSI y el ESGSI extendido como scores continuos, se asocia con resultados
financieros externos. Cubre la primera limitación de §6 del paper: hasta ahora los scores
no se contrastan con ningún criterio externo.

- **H1.** Una mayor divergencia se asocia con una mayor dispersión de las previsiones de los analistas.
- **H2.** Una mayor divergencia en t anticipa controversias ESG en t+1 o t+2.

Se siguen las directrices de `paper/_research/directrices.md`: el score se usa como variable
continua, sin umbrales ni etiquetas; solo el vocabulario actual; ninguna empresa aparece
por su nombre (identificadores anónimos `<Industria>-NN`, resultados por sector y país).

## 0. Resumen

| Pregunta | Respuesta corta |
|---|---|
| ¿Hay datos gratuitos para H1 y H2 en 2018-2024? | **No.** Las fuentes gratuitas probadas dan solo la foto actual del consenso (Yahoo) o no devuelven nada para estas empresas (ESG de Yahoo, histórico de GDELT). Para H1 y H2 hacen falta I/B/E/S y RepRisk (o LSEG), normalmente a través de una licencia institucional (WRDS o LSEG Workspace). |
| ¿Cobertura de las 49 empresas? | Las 49 cotizan y tienen precios diarios 2017-2026. Todas tienen consenso de analistas hoy: mediana de 21 analistas, con un rango de 8 a 35. Por su tamaño cabe esperar que I/B/E/S y RepRisk las cubran todas, pero falta comprobarlo con acceso. |
| ¿Potencia? | El efecto mínimo detectable (MDE) de H1 está en torno a 0,21-0,35 desviaciones típicas (DT) del resultado por cada DT del score. En H2, con logit agrupado y una tasa base del 10-20 %, el MDE es una odds ratio de 1,5-2,0. Con efectos fijos de empresa, H2 binaria no alcanza el 80 % de potencia ni siquiera con OR = 2,7. |
| ¿Mandatos? | La Taxonomía (calendario distinto para financieras y no financieras) y la CSRD en el ejercicio 2024 (solo en los países que la habían transpuesto) dan variación de exposición. Con ella se definen 4 regímenes, y efectos fijos de régimen × año absorben cualquier choque común de mandato. |
| ¿Modelo exploratorio? | Se estimó con proxies de mercado gratuitos: volatilidad idiosincrática y asimetría negativa de los rendimientos. No son las variables de H1 y H2. Ninguna asociación con el ESGSI se distingue de cero. La mayor es la del ESGSI extendido con la volatilidad idiosincrática (+0,16 DT; p agrupado 0,08, p wild 0,047 en EF de empresa y año), que no es estable entre especificaciones y queda por debajo del MDE. |
| ¿Pipeline listo? | Sí. Hay un esquema de entrada para los datos comerciales y una validación simulada: el estimador de H1 no tiene sesgo y su cobertura está entre el 90 y el 97 %. |

## 1. Universo y panel

- **49 empresas**, todas cotizadas en bolsas de la zona euro y de gran capitalización. Están en 7 países: Francia 17,
  Alemania 14, Países Bajos 6, Italia 5, España 4, Finlandia 2 y Bélgica 1. Cubren 10 industrias ICB y 16 supersectores.
- **Panel equilibrado de 343 informes** (2018-2024, uno por empresa y año).
- **Identificadores de mercado.** `metadata/empresas_tickers.csv` recoge el ticker de Yahoo, la bolsa, la marca de empresa
  financiera a efectos del art. 8 de la Taxonomía y el primer ejercicio cuyo informe corresponde al emisor del
  ticker. Es un fichero de metadatos interno, como `empresas_supersector.csv`. **No hay ISIN en
  `metadata/`**: hará falta añadirlos para cruzar con WRDS o LSEG (decisión abierta 1).
- **Casos especiales.** Una empresa (automóvil, Países Bajos) procede de una fusión, y sus informes de 2018 y 2019
  son del predecesor que no cotiza con ese ticker; esos 2 informes se excluyen de los resultados de mercado.
  Otra (Industrials, Países Bajos) salió a bolsa a mediados de 2018, así que su ventana de control de 2018 está
  incompleta. Muestra efectiva de los proxies: 341 informes con resultado y 340 con controles.
- **Variación disponible del regresor.** Con las cifras de este panel, el 66,7 % de la varianza del ESGSI está entre empresas.
  Tras quitar efectos fijos de empresa y año solo queda el **26 %** (el 22 % en el ESGSI extendido). Las
  especificaciones intra-empresa trabajan con una cuarta parte de la variación.

## 2. Fuentes de datos

### 2.1 Dispersión de analistas (H1)

| Fuente | Variable | Histórico | Acceso | Estado |
|---|---|---|---|---|
| **I/B/E/S** (LSEG), vía WRDS o LSEG Workspace | Resumen mensual: STDEV, MEANEST, NUMEST del BPA de t+1 | desde los años 80; internacional | licencia institucional (de pago) | estándar en la literatura; **opción recomendada** |
| FactSet Estimates | igual | sí | licencia (de pago) | alternativa |
| Bloomberg BEst | igual | sí | terminal (de pago) | alternativa si la facultad lo tiene |
| Yahoo Finance (`yfinance`) | consenso actual: n.º de analistas, máx./mín./media del BPA, recomendaciones | **no hay histórico**: solo la foto de hoy; la sorpresa de BPA cubre solo los últimos 4 trimestres | gratis | **probado: inservible para 2018-2024** |

Prueba sobre Yahoo con las 49 empresas, sin nombres: 49 de 49 tienen consenso vigente, con una mediana
de 21 analistas (8-35) y un rango relativo del BPA del ejercicio en curso de 0,13 de mediana. Solo 5 de las 49 tienen
histórico de cambios de recomendación. Detalle en `cobertura_yfinance.csv` y `cobertura_yfinance.json`.

### 2.2 Controversias ESG (H2)

| Fuente | Contenido | Histórico | Acceso | Estado |
|---|---|---|---|---|
| **RepRisk** (vía WRDS) | incidentes diarios con severidad, alcance y novedad; índice RRI; 28 temas | desde 2007 | licencia académica vía WRDS (de pago para la institución) | **opción recomendada**: fecha del incidente y severidad, lo que permite definir t+1 y t+2 con precisión |
| LSEG ESG (antes Refinitiv) | score de controversias (23 temas) y ESGC | desde 2002 | LSEG Workspace o Datastream (de pago) | alternativa; anual, con decalaje y revisiones retroactivas |
| MSCI ESG Controversies | banderas y casos | sí | licencia (de pago) | alternativa |
| Sustainalytics (Morningstar) | nivel de controversia 0-5 | sí | licencia (de pago) | alternativa |
| Yahoo ESG (Sustainalytics) | riesgo ESG y controversia máxima | — | gratis | **probado: vacío para las 49** |
| GDELT DOC 2.0 API | volumen de noticias por consulta | desde 2017 en teoría | gratis | **probado**: funciona para ventanas recientes (≤ 12 meses); las consultas 2018-2024 devuelven HTTP 429. Haría falta el GKG masivo (BigQuery, con cuenta) |
| Violation Tracker Global (Good Jobs First) | sanciones regulatorias por matriz: año, materia, jurisdicción e importe | desde 2010 | consulta gratis; la descarga es de pago (5 USD por 50 registros o suscripción desde 25 USD/mes) | **probado**: una matriz del panel muestra 51 registros 2010-2024. Cloudflare bloquea el acceso automatizado y no se ha extraído. Una sanción no es una controversia: llega con retraso y la UE tiene lagunas en medio ambiente y empleo |

### 2.3 Lo que sí es gratis y cubre el panel

Las **cotizaciones diarias** de las 49 empresas y del EURO STOXX 50 (Yahoo, 2017-2026). Con ellas se construyen
proxies de mercado de la misma familia que H1 y H2, aunque **no son H1 ni H2**:

- **Volatilidad idiosincrática** (log, anualizada): incertidumbre informativa, sustituto de la dispersión de analistas.
- **NCSKEW, DUVOL y semana de caída extrema**: acumulación y revelación de malas noticias (Chen, Hong y Stein, 2001;
  Kim, Li y Zhang, 2014). Es el análogo de mercado de una controversia. La tasa de años con caída extrema es
  del 21 %.

## 3. Cobertura y potencia

### 3.1 Observaciones utilizables

- **H1**: 343 informes si se mide la dispersión del BPA de t+1 en el mes siguiente a la publicación del informe de t.
  El informe de 2024 se publica en 2025 y sus previsiones ya existen.
- **H2**: depende de hasta cuándo lleguen las controversias.
  - t+1: 294 informes si llegan a 2024, 343 si llegan a 2025.
  - t+2: 245 informes si llegan a 2024, 294 si llegan a 2025.
- **Frecuencia de controversias**: no se puede medir sin RepRisk o LSEG. En empresas de este tamaño casi todos los
  años tienen algún incidente, así que "al menos un incidente" sería una constante inútil. El resultado debe fijarse ex ante
  sobre la severidad o el alcance de la fuente, por ejemplo incidentes de severidad media o alta. Las tasas base plausibles van del 10 al 40 %.

### 3.2 Simulación de potencia

`fin_power.py` usa la estructura real del panel y los valores reales del ESGSI y del ESGSI extendido; solo
simula el resultado. Se hacen 1000 réplicas por punto, con SE agrupados por empresa (CR1), t con 48 g.l., un contraste bilateral al 5 %
y una potencia objetivo del 80 %.

**H1** (resultado continuo, efecto en DT del resultado por DT del score; ρ = parte de la varianza del resultado entre empresas):

| Regresor | ρ | MDE agrupada (EF año, industria y país) | MDE con EF de empresa y año |
|---|---|---|---|
| ESGSI | 0,3 | 0,29 | 0,28 |
| ESGSI | 0,6 | 0,32 | 0,21 |
| ESGSI ext. | 0,3 | 0,31 | 0,30 |
| ESGSI ext. | 0,6 | 0,35 | 0,23 |

Solo se detectan con fiabilidad efectos moderados, en torno a 0,2-0,35 DT. Como referencia, en
I/B/E/S la dispersión es muy persistente dentro de cada empresa (ρ alto), lo que favorece la especificación intra-empresa.

**H2** (odds ratio mínima por DT del score, ESGSI; σ_u = heterogeneidad latente entre empresas):

| Tasa base | N = 294, σ_u = 0 | N = 294, σ_u = 1 | N = 245, σ_u = 0 | N = 245, σ_u = 1 |
|---|---|---|---|---|
| 5 % | 1,95 | 2,26 | 2,04 | 2,38 |
| 10 % | 1,73 | 2,01 | 1,80 | 2,10 |
| 20 % | 1,53 | 1,84 | 1,60 | 1,93 |
| 40 % | 1,45 | 1,75 | 1,49 | 1,79 |

- Con logit agrupado (EF de año), una tasa base del 10-20 % y heterogeneidad entre empresas, hace falta una **OR ≈ 1,8-2,0**: unos
  6-8 puntos porcentuales por DT.
- Con **efectos fijos de empresa** (modelo lineal de probabilidad) la potencia de H2 binaria no llega al 80 % en
  la rejilla (OR ≤ 2,7), salvo con una tasa del 40 % sin heterogeneidad (OR 2,56): **H2 no se puede identificar solo con la variación intra-empresa**.
- Un **recuento de incidentes** (Poisson con EF de año) es más potente. Con una media de 2 incidentes por año el MDE es una
  IRR de 1,13 sin heterogeneidad y de 1,52 con ella; con una media de 0,5, de 1,27 y 1,59. Es otro argumento a favor de RepRisk, que da incidentes fechados y no solo un score anual.
- Los resultados con el ESGSI extendido son muy parecidos (`potencia_h2.csv`).

## 4. Diseño: controles y efectos fijos

- **Controles** (medidos en t, antes del resultado):
  - tamaño: log de la capitalización a cierre de t;
  - cobertura de analistas: log del n.º de previsiones;
  - rating ESG previo: score del proveedor a cierre de t-1, estandarizado;
  - en H1, el retardo del resultado; en H2, controversias previas (t-1, t).
  - Con datos de mercado se añaden el rendimiento, la volatilidad y la asimetría previas.
- **Efectos fijos**:
  - siempre EF de año;
  - en la especificación agrupada (A), EF de industria ICB y de país. El tipo de documento casi coincide con el país (§3 del paper), así que no se añade aparte;
  - en la especificación intra-empresa (B), EF de empresa, que absorben industria, país y tipo de documento.
- **Inferencia**: SE agrupados por empresa (49 clústeres) y wild cluster bootstrap restringido (Rademacher, 9999 réplicas),
  como en el paper. Con 7 años no se agrupa por año.
- **Escala**: score estandarizado en la muestra de estimación. Se informa el efecto por DT y, en H1, también en DT del resultado.
- **Tono frente a sustancia** (especificación E): Z(SEN) y Z(SUS) por separado, para ver si una posible asociación
  viene del tono o de la densidad de contenido ESG.

## 5. Separar la divergencia de los mandatos de reporte (§4.2)

El problema de §4.2 es que el mandato es un choque común, colineal con el año, y que cumplirlo baja ambos scores. Para H1 y H2
lo que importa es otra cosa: que el mandato mueva a la vez el score y el resultado, por ejemplo con más información prescrita,
menos dispersión y menos controversias inesperadas. La exposición **no** es idéntica para las 49 empresas:

| Mandato | Ejercicio de aplicación | Variación en el panel |
|---|---|---|
| NFRD (estado no financiero) | 2017-2018 | ninguna: todas obligadas desde el inicio |
| Taxonomía, art. 8: elegibilidad | 2021 | ninguna |
| Taxonomía, art. 8: alineamiento | no financieras 2022; **financieras 2023** | por sector regulatorio (12 financieras, 37 no financieras) |
| CSRD/ESRS, primera ola | **2024, solo donde estaba transpuesta** | Francia, Italia, Finlandia y Bélgica sí (25 empresas); Alemania, España y Países Bajos no (24). **A verificar**: Bélgica (transposición en diciembre de 2024) y Países Bajos |

Con esa variación se proponen cinco especificaciones, todas implementadas en `fin_models.py`:

- **A. Agrupada** con EF de año, industria y país. Es la referencia, y usa también la variación entre empresas.
- **B. EF de empresa y año.** Identifica con las desviaciones de cada empresa respecto a su media y al año. Elimina
  el componente común del mandato.
- **C. EF de empresa y de régimen × año.** Hay cuatro regímenes: financiera o no × CSRD en vigor en 2024 o no. Tienen 17, 20, 7 y 5 empresas.
  Absorben cualquier choque de mandato propio de cada régimen y año, incluido el escalón de la Taxonomía en 2022-2023 y
  el de la CSRD en 2024. Si β se mantiene de B a C, el mandato no es lo que la explica.
- **D. Interacción** score × mandato prescrito (alineamiento de Taxonomía o CSRD en vigor). Contrasta si la asociación cambia
  cuando el contenido pasa a ser obligatorio. Si la divergencia recoge discrecionalidad, debería debilitarse.
- **E. Componentes** (tono frente a sustancia), como arriba.

Hay tres complementos que conviene añadir antes de llevarlo al paper:

1. **Score depurado de régimen.** Se recalcula el ESGSI sin el pilar transversal, que contiene los nombres de regímenes y
   estándares, como en el ejercicio de §4.2, y se usa como regresor alternativo. Hoy esas puntuaciones por informe no se
   guardan: hay que hacer que `robustness_v3.py` las exporte.
2. **Primera etapa en el propio score.** Se compara el score de las empresas sujetas a la CSRD en 2024 con el de las que no
   lo están, antes y después: una diferencia en diferencias con un solo año posterior, 25 frente a 24 empresas. Mide cuánto mueve el mandato el score.
3. **Placebo temporal.** El score de t frente al resultado de t-1, que no puede causar. Si aparece "asociación", indica
   una característica persistente de la empresa y no un efecto de la divergencia.

## 6. Modelo exploratorio con proxies de mercado (datos reales)

La ventana del resultado del informe del ejercicio t va del 1 de abril de t+1 al 31 de marzo de t+2, después de la
publicación. Se usan rendimientos semanales y un modelo de mercado con el índice (retardo, contemporáneo y adelanto). Los controles se miden en el año natural t. Hay 340 informes,
49 empresas y SE agrupados por empresa; p wild con 9999 réplicas. El coeficiente es el efecto de 1 DT del score.

| Resultado | Score | A agrupada | B EF emp. + año | C EF emp. + régimen × año | D: score × mandato |
|---|---|---|---|---|---|
| log vol. idiosincrática | ESGSI | 0,020 (p 0,13) | 0,032 (p 0,12; wild 0,10) | 0,036 (p 0,12; wild 0,08) | 0,020 (p 0,43) |
| log vol. idiosincrática | ESGSI ext. | 0,034 (p 0,05) | 0,046 (p 0,08; wild 0,047) | 0,040 (p 0,12; wild 0,06) | 0,013 (p 0,61) |
| NCSKEW | ESGSI | −0,018 (p 0,69) | 0,079 (p 0,26) | 0,074 (p 0,29) | −0,089 (p 0,21) |
| NCSKEW | ESGSI ext. | −0,020 (p 0,66) | 0,054 (p 0,41) | 0,051 (p 0,49) | −0,008 (p 0,92) |
| DUVOL | ESGSI | −0,011 (p 0,71) | 0,026 (p 0,59) | 0,031 (p 0,51) | −0,057 (p 0,30) |
| Caída extrema (0/1) | ESGSI | −0,004 log-odds (p 0,98) | 0,022, es decir +2,2 pp (p 0,56) | 0,023 (p 0,59) | 0,001 (p 0,98) |

Tabla completa, con controles y componentes, en `modelos_proxy_mercado.csv`. Descriptivos en `descriptivos_proxy_mercado.csv`.

**Lectura.**

- Ninguna asociación se distingue de cero de forma robusta.
- La única cercana al 5 % es la del ESGSI extendido con la volatilidad idiosincrática:
  0,046 en log, es decir +0,16 DT del resultado por DT del score. Su p agrupado es 0,08 y su p wild 0,047, pero cae en C y en D, y es una entre
  unas 50 pruebas. No es evidencia.
- Las estimaciones puntuales (0,05-0,16 DT) quedan por debajo del MDE de §3 (≈ 0,2-0,35 DT): con 49 empresas, efectos de ese
  tamaño no son detectables aunque existan.
- En los componentes (E) el signo es el esperado para la sustancia (más densidad ESG, menos volatilidad: −0,11 DT), pero p = 0,19.
- **Estos proxies no contrastan H1 ni H2**: miden incertidumbre y malas noticias en el mercado, no dispersión de
  analistas ni controversias ESG.

## 7. Validación del pipeline con datos simulados

`fin_models.py` genera datos con el esquema de `metadata/issue2_esquema_outcomes.csv` sobre el panel real de scores. Incluye
efectos conocidos y un choque de mandato que afecta directamente al resultado (alineamiento de Taxonomía y CSRD). Se hacen 300 réplicas:

| Resultado | Especificación | Verdad | Media estimada | Cobertura IC 95 % | Potencia |
|---|---|---|---|---|---|
| dispersión (H1) | A | 0,30 | 0,296 | 0,90 | 0,82 |
| dispersión (H1) | B | 0,30 | 0,298 | 0,97 | 0,95 |
| dispersión (H1) | C | 0,30 | 0,303 | 0,96 | 0,94 |
| controversia t+1 (H2) | A logit | 0,588 (OR 1,8) | 0,528 | 0,91 | 0,71 |
| controversia t+1 (H2) | B / C, modelo lineal de probabilidad | — | 0,074 / 0,073 | — | 0,27 / 0,21 |

- El estimador de H1 no tiene sesgo en ninguna especificación. La cobertura de la agrupada (0,90) es algo baja, como es habitual con
  49 clústeres; por eso se informa también el p wild.
- El logit agrupado de H2 estima un efecto medio poblacional, atenuado respecto al condicional cuando hay heterogeneidad
  entre empresas (σ_u = 1). Es lo esperado, no un error.
- La baja potencia del modelo lineal de probabilidad con EF de empresa confirma lo de §3.

## 8. Decisiones abiertas para el autor

1. **Acceso a datos.** ¿Tiene la universidad WRDS (I/B/E/S y RepRisk) o LSEG Workspace? Sin ellos no se pueden contrastar H1 y H2.
   Si lo tiene, hay que añadir los ISIN a `metadata/` y rellenar el esquema de `metadata/issue2_esquema_outcomes.csv` en
   `data/issue2_finance/outcomes_externos.csv`. Después basta con volver a ejecutar `fin_models.py`.
2. **Definición de controversia**:
   - ¿binaria (severidad media o alta), recuento de incidentes o variación del RRI?
   - Por potencia se recomienda el recuento o el RRI, con la binaria como robustez.
   - Hay que fijar el umbral de severidad ex ante sobre la fuente. Ese umbral es de la variable externa, no del score.
3. **Horizonte de H2**: t+1, t+2 o ambos, y hasta qué año llegan los datos. Determina N, que va de 245 a 343.
4. **Medida de dispersión**: desviación típica entre |media| o entre precio. El precio es más estable con BPA cerca de cero, frecuente en bancos y en 2020.
   También hay que fijar la fecha de medida respecto a la publicación de cada informe. Las fechas exactas de publicación no están en el corpus; hoy se usa el 1 de abril de t+1.
5. **Exposición a la CSRD en 2024**: verificar la situación de Bélgica y de los Países Bajos, y confirmar qué empresas
   financieras aplican los KPI de la Taxonomía como entidad financiera. El caso dudoso es una empresa de pagos con licencia bancaria clasificada en Industrials.
6. **Score depurado de régimen**: decidir si se exporta el ESGSI sin el pilar transversal por informe para usarlo como
   regresor alternativo.
7. **Alcance en el paper**:
   - Con 49 clústeres, la potencia solo alcanza efectos moderados (≈ 0,2-0,35 DT en H1; OR ≈ 1,8-2 en H2 binaria).
   - Un resultado nulo no sería informativo salvo que se presente con su MDE.
   - Hay que valorar si ampliar el panel (más empresas, que exigiría extraer más informes) o presentar esta parte como validación externa
     exploratoria.
   - ¿Se incluyen los proxies de mercado como análisis complementario o se dejan fuera?
8. **Violation Tracker Global**: ¿merece pagar la descarga (suscripción mínima de 25 USD/mes) como criterio externo
   complementario de conducta (sanciones), distinto de las controversias?

## 9. Reproducción

```bash
# entorno: conda env 'esgwashing' (Python 3.11) + yfinance instalado aparte
python scripts/fin_fetch_market.py --results-dir results   # precios + sondeo de cobertura (red)
python scripts/fin_power.py        --results-dir results --reps 1000
python scripts/fin_models.py       --results-dir results --sim-reps 300 --n-boot 9999
```

- **Scripts**:
  - `scripts/fin_econ.py`: estimadores con SE agrupados, doble desviación y wild bootstrap.
  - `scripts/fin_fetch_market.py`: datos de mercado y cobertura de Yahoo.
  - `scripts/fin_power.py`: simulación de potencia.
  - `scripts/fin_models.py`: proxies de mercado, datos externos y validación.
- **Ficheros internos** (no versionados, en `data/issue2_finance/`): precios, mapa firm_id-ticker y panel de mercado.
  En `results/issue2_finance/` solo hay identificadores anónimos, coeficientes y agregados.
- Semilla: 20261009.

Fuentes consultadas sobre acceso:
- RepRisk en WRDS: https://www.reprisk.com/research-insights/news-and-media-coverage/2015-10-20-reprisk-risk-data-now-available-to-academic
- Violation Tracker Global: https://violationtrackerglobal.goodjobsfirst.org/
- API DOC 2.0 de GDELT: https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/
- Metodología ESG de LSEG: https://www.lseg.com/content/dam/data-analytics/en_us/documents/methodology/lseg-esg-scores-methodology.pdf
- Seguimiento de la transposición de la CSRD: https://www.gibsondunn.com/wp-content/uploads/2026/09/CSRD_Transposition-Tracker_Draft_September_2026.pdf
