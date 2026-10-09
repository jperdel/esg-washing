# Issue #2: trabajo preparatorio (mandatos, identificadores y datos de analistas)

Informe de trabajo (octubre de 2026). Prepara tres piezas que necesita el diseño de
`results/issue2_finance/README.md` (rama `issue-2-financial-consequences`) antes de pedir datos
comerciales:

1. los hechos de mandato (NFRD, Taxonomía y CSRD) con fuentes oficiales, en un panel empresa × año;
2. los ISIN y LEI de las 49 empresas, para cruzar con WRDS o LSEG;
3. si alguna vía gratuita da datos históricos de dispersión de analistas (2018-2025).

Se siguen las directrices de `paper/_research/directrices.md`: ninguna empresa aparece por su nombre
(solo `<Industria>-NN`), el ESGSI es un score continuo y no se usan etiquetas.

## 0. Resumen

| Pregunta | Respuesta corta |
|---|---|
| ¿CSRD en el ejercicio 2024? | **Sí** en Francia, Italia, Finlandia y Bélgica. **No** en Alemania, España y Países Bajos, que a octubre de 2026 siguen sin transponerla. Bélgica sí la aplica: ley de 2-12-2024 (M.B. 20-12-2024), con la ola 1 en el ejercicio 2024. La stop-the-clock (Dir. 2025/794) solo aplaza las olas 2 y 3. |
| ¿Cuántas empresas tenían la CSRD en su informe de 2024? | **23 de 49**, no 25. Lo que cuenta es el Estado de la **sede legal**, no el de cotización. Dos empresas de las muestras francesa e italiana son sociedades neerlandesas y por tanto no tenían CSRD en 2024. |
| ¿Taxonomía? | En los informes publicados en 2022, solo elegibilidad. Los KPI de alineamiento se exigen desde el 1-1-2023 a las no financieras y desde el 1-1-2024 a las financieras. Manda la fecha de publicación, así que las tres empresas con ejercicio no natural (julio-junio y octubre-septiembre) se retrasan un año. |
| ¿Empresa de pagos? | Informa como **entidad de crédito**, por su licencia bancaria neerlandesa. Usa las plantillas de entidades de crédito del Reglamento Delegado 2021/2178: elegibilidad en 2022 y GAR desde 2023. Va con las financieras. Al revés, la financiera de servicios de mercado (Financials-01) informa como **no financiera** y alinea desde 2022. |
| ¿NFRD? | Las 49 estaban obligadas desde el informe de 2018: son EIP grandes con más de 500 empleados y ninguna es una filial exenta. El caso más ajustado es la empresa de pagos en 2018, con unos 770 ETC. |
| ¿ISIN y LEI? | **49/49 ISIN** verificados con OpenFIGI: son acciones y el ticker coincide con el de Yahoo. **49/49 LEI** en GLEIF: 47 por el mapa ISIN-LEI y 2 por nombre legal. También se obtuvieron los LEI de los dos predecesores. |
| ¿Datos históricos de analistas gratis? | **No sirven como fuente principal.** Yahoo y MarketScreener archivados no dan el rango del BPA. Lo único útil son los **consensos que compilan las propias empresas** (PDF/XLS de relación con inversores) archivados en la Wayback Machine. Cubren **39 de 343 informes (11 %) en 9 empresas**. En el piloto, 26 informes tienen BPA del ejercicio t+1 con mínimo, máximo y número de analistas. Pueden servir para validar I/B/E/S en una submuestra, no para sustituirlo. |
| Hallazgos laterales | (1) **El PDF de 2024 de Financials-04 es el informe anual de otra empresa**, una cooperativa no cotizada de otro país con un nombre parecido. Ese informe del panel no es válido. (2) La etiqueta 2024 de Consumer Staples-04 corresponde al ejercicio cerrado en junio de 2025, y falta el cerrado en junio de 2024. (3) En 2024, **22 de las 26 empresas sin CSRD legal ya incluyen un estado ESRS**, así que la CSRD legal apenas separa el contenido real (§3). |

## 1. Hechos de mandato verificados

Las fuentes, con URL y fecha, están en `metadata/issue2_mandates_fuentes.csv`. Se indica cuáles se leyeron
en esta sesión y cuáles son referencias estables de diario oficial.

### 1.1 CSRD (Dir. (UE) 2022/2464), ola 1: grandes EIP con más de 500 empleados, ejercicios iniciados desde el 1-1-2024

| País | Norma de transposición | ¿Informes del ejercicio 2024 bajo ESRS por ley? |
|---|---|---|
| FR | Ordonnance 2023-1142 (6-12-2023, JORF 7-12-2023) y Décret 2023-1394 (30-12-2023) | **Sí** (ejercicios abiertos desde el 1-1-2024) |
| IT | D.Lgs. 6-09-2024 n. 125 (GU 10-09-2024; en vigor el 25-09-2024) | **Sí** |
| FI | Laki kirjanpitolain muuttamisesta 1249/2023 (publicada el 29-12-2023; HE 20/2023) | **Sí** |
| BE | Loi du 2 décembre 2024 (M.B. 20-12-2024). Según su art. 116, la ola 1 es el ejercicio 2024 y publica en 2025 (nota técnica del IBR-IRE de 18-02-2025) | **Sí**. Llegó tarde, pero antes del cierre del ejercicio 2024 y de la publicación de sus informes |
| NL | **Sin transponer.** El proyecto 36.678 se presentó el 13-01-2025 y sigue en la Tweede Kamer (último documento: 13-05-2026). El 9-02-2026 el Gobierno anunció una "cláusula de reparación" para dar por cumplidos, con efecto retroactivo, los informes 2024-2025 elaborados con ESRS | **No**: sigue aplicándose el Besluit niet-financiële informatie (NFRD). Queda pendiente esa convalidación retroactiva |
| DE | **Sin transponer.** El proyecto de 2024 decayó. El nuevo (BT-Drs. 21/1857, 3-09-2025; enmienda de 31-03-2026) está sin aprobar | **No**: se aplican los §§ 289b ss. HGB. El IDW considera inconstitucional aplicarlo retroactivamente a 2024 |
| ES | **Sin transponer.** El Proyecto de Ley de Información Empresarial sobre Sostenibilidad (29-10-2024) sigue en tramitación | **No**: se aplica la Ley 11/2018. La CNMV y el ICAC recomiendan usar las ESRS como marco |

- **Stop-the-clock** (Dir. (UE) 2025/794, 14-04-2025): su art. 1 solo cambia las letras b) y c) del art. 5(2) de
  la CSRD, es decir, las olas 2 y 3 (a 2027 y 2028). La ola 1 (letra a) no se toca. **Confirmado** con el texto oficial.
  El Ómnibus I (Dir. (UE) 2026/470) cambia el alcance futuro, pero no los informes de 2024.
- **Sede legal frente a país de la muestra.** La obligación la fija el Estado cuyo derecho rige la sociedad
  (GLEIF, dirección legal). En la muestra, una empresa del grupo francés (Industrials-07) y otra del italiano
  (Consumer Discretionary-10) son **sociedades neerlandesas**, así que en 2024 **no** tenían CSRD. Con eso el reparto pasa de 25/24
  a **23/26**: FR 16, IT 4, FI 2 y BE 1 con CSRD; DE 14, ES 4 y NL 8 sin ella.
- **Regímenes de la especificación C** (financiera según la plantilla del art. 8 × CSRD en 2024): no financiera con CSRD 18,
  no financiera sin CSRD 20, financiera con CSRD 5 y financiera sin CSRD 6. Antes eran 17/20/7/5.
- **Ejercicios no naturales.** Dos empresas alemanas cierran en septiembre: su ejercicio 2024 empezó el 1-10-2023,
  así que no entrarían en la ola 1 aunque Alemania hubiera transpuesto. Una francesa cierra en junio: su
  etiqueta 2024 es el ejercicio julio de 2024-junio de 2025, que sí está bajo la CSRD.

### 1.2 Taxonomía, art. 8 (Reg. (UE) 2020/852 y Reg. Delegado (UE) 2021/2178, art. 10, leído en el texto oficial)

- Del 1-1-2022 al 31-12-2022, las no financieras publican **solo elegibilidad**: informes del ejercicio 2021.
- Del 1-1-2022 al 31-12-2023, las financieras publican **solo elegibilidad** más exposiciones: informes de los ejercicios 2021 y 2022.
- KPI completos (alineamiento) **desde el 1-1-2023 para las no financieras** (ejercicio 2022) y **desde el 1-1-2024 para
  las financieras** (ejercicio 2023: GAR de las entidades de crédito, KPI de inversiones y de suscripción de las
  aseguradoras). Los KPI de cartera de negociación y comisiones de las entidades de crédito llegan el 1-1-2026.
- El plazo depende de la **fecha de publicación**. Así lo confirman los propios informes: las dos alemanas que cierran
  en septiembre publican en diciembre de 2022 solo elegibilidad y alinean desde la etiqueta 2023. La que cierra en junio
  se comporta igual.
- El acto delegado medioambiental (2023/2486) añade los objetivos 3 a 6: elegibilidad en el ejercicio 2023 y alineamiento en 2024
  (no financieras). Es un escalón más dentro de la Taxonomía, común a todo el panel.
- **Plantilla del art. 8 en el panel**: 8 entidades de crédito (incluida la empresa de pagos), 3 aseguradoras y 38
  no financieras (incluida Financials-01).
- **Empresa de pagos (Industrials-06).** Su informe de 2022 dice que, "como entidad de crédito", del 1-1-2022 al
  31-12-2023 publica la proporción de exposiciones elegibles. Usa las plantillas de entidades de crédito del
  Reglamento Delegado con datos FinRep. El de 2023 publica el **GAR** con esas mismas plantillas. Encaja con su licencia
  de entidad de crédito de DNB, que ya la convierte en EIP desde 2017.
- **Financials-01** declara en 2024 que aplica "los requisitos para empresas no financieras" (la supervisión no la
  considera sociedad financiera de cartera). Ya publicó alineamiento de ventas, CapEx y OpEx en el informe de 2022.

### 1.3 NFRD (Dir. 2014/95/UE)

- Se aplica a los ejercicios iniciados desde el 1-1-2017, a EIP grandes con más de 500 empleados de media. Las normas
  nacionales son: CSR-RUG (DE), Ord. 2017-1180 (FR), D.Lgs. 254/2016 (IT), RDL 18/2017 y Ley 11/2018 (ES), Besluit
  bekendmaking niet-financiële informatie (NL), Loi du 3-09-2017 (BE) y KPL 1376/2016 (FI).
- **Las 49 están sujetas desde el informe de 2018.** Todas son emisoras cotizadas en un mercado regulado de la UE
  (EIP), son grandes y superan holgadamente los 500 empleados. Ninguna es filial de una matriz de la UE que la consolide,
  así que la exención de filial no aplica.
- Casos revisados:
  - **Empresa de pagos, 2018.** Es EIP desde abril de 2017 por su licencia bancaria. Tenía 668 ETC a cierre de 2017 y
    873 a cierre de 2018 (unos 770 de media), por encima de 500. Su informe de 2018 no rotula un "estado no
    financiero", pero la obligación existe.
  - **Financials-07, 2018.** Se redomicilió en Finlandia el 1-10-2018, así que su informe de 2018 ya se rige por la
    ley finlandesa. Antes había estado sujeta a la ley sueca, que también transpone la NFRD.
  - **Consumer Discretionary-09, 2018-2019.** Esos informes son del predecesor, una SA francesa sujeta a la DPEF.
- En 2024, las 23 empresas con CSRD pasan del régimen NFRD al régimen ESRS: `nfrd = 0` y `csrd = 1`.

### 1.4 `metadata/issue2_mandates.csv`

Tiene 343 filas (empresa × año, 2018-2024) con estas columnas:

- `pais_muestra`, `sede_legal` (GLEIF), `plantilla_art8`;
- `inicio_ejercicio` y `publicacion_aprox`;
- marcas `nfrd`, `taxonomy_eligibility`, `taxonomy_alignment` y `csrd`, más `regimen`;
- `nota` y `sources`, con códigos de `issue2_mandates_fuentes.csv`.

Totales por año:

| Año | NFRD | Elegibilidad | Alineamiento | CSRD |
|---|---|---|---|---|
| 2018-2020 | 49 | 0 | 0 | 0 |
| 2021 | 49 | 46 | 0 | 0 |
| 2022 | 49 | 49 | 35 | 0 |
| 2023 | 49 | 49 | 49 | 0 |
| 2024 | 26 | 49 | 49 | 23 |

## 2. Identificadores (ISIN y LEI)

- `scripts/gw_identifiers.py` parte de ISIN candidatos y los verifica con dos fuentes gratuitas:
  - **OpenFIGI** (sin clave, ≤ 25 peticiones por minuto): tipo de valor y tickers. Las 49 son "Common Stock"
    (la preferente de Consumer Discretionary-04 sale como "Preference", como debe) y en las 49 el ticker coincide con el de Yahoo.
  - **GLEIF**: LEI, nombre legal, país de la dirección legal y estado.
- Resultado:
  - 47 LEI salen por el mapa ISIN-LEI y 2 (las finlandesas) por nombre legal exacto.
  - Todos están en estado ISSUED y el país del LEI coincide con la sede legal.
  - Predecesores: el LEI retirado de la matriz sueca de Financials-07 y el de la SA francesa de Consumer Discretionary-09.
- **Ficheros**:
  - `metadata/empresas_identificadores_internal.csv` y `metadata/empresas_isin_internal.csv`: nombres, ISIN, LEI, FIGI y tickers.
    **No se versionan**: `.gitignore` incluye ahora `metadata/*_internal.csv`.
  - `metadata/issue2_identificadores_anon.csv` (versionado): solo `firm_id` y marcas de verificación. **No lleva el ISIN ni el LEI**,
    porque cualquiera de los dos identifica a la empresa.
- Para WRDS:
  - El ISIN de Consumer Discretionary-09 es el de la sociedad absorbente desde 2021. Para los informes de 2018 y 2019 hay que
    usar el ISIN del predecesor francés (está en el fichero interno).
  - I/B/E/S se cruza por CUSIP/SEDOL o por su propio ticker. El ISIN sirve para pasar por la tabla de enlace de WRDS.

## 3. Evidencia documental de los regímenes (`mandate_evidence_by_report.csv`)

`scripts/gw_mandate_evidence.py` cuenta marcadores en el texto de los 343 PDF del corpus (solo lectura):

- ESRS y "sustainability statement";
- estado no financiero;
- CSRD;
- elegibilidad, alineamiento y GAR de la Taxonomía;
- referencia al Reglamento Delegado 2021/2178.

Resultados:

- **ESRS de hecho en 2024.** Hay 43 informes con al menos 20 menciones a ESRS: 21 de las 23 empresas con CSRD legal y **22 de las 26 sin
  ella**. Las empresas neerlandesas, alemanas y españolas usan las ESRS como marco voluntario (recomendado por CNMV/ICAC; en
  los Países Bajos, en espera de la convalidación). Las dos con CSRD pero sin marcadores tienen otro tipo de documento en el
  corpus (§5). Las cuatro sin CSRD y sin ESRS son las dos que cierran en septiembre, el PDF erróneo (§5) y un informe
  integrado que no incluye el estado no financiero.
- **Consecuencia para el diseño (§5 del README anterior).** La variación "CSRD legal en 2024" casi no separa el contenido
  real de los informes. Una diferencia en diferencias sobre la CSRD legal tendrá una primera etapa débil. Conviene
  usar como exposición alternativa la adopción efectiva de ESRS (marcador documental) o tratar 2024 como un choque común,
  absorbido por los EF de año, y apoyarse en el escalón de la Taxonomía (2022 frente a 2023), que sí es obligatorio y
  distinto por plantilla.
- **Taxonomía.** El alineamiento aparece donde y cuando el calendario legal lo exige (45 o 46 informes de 49 en 2023 y 2024).
  Algunas empresas ya hablan de alineamiento voluntariamente en 2021-2022. El conteo de palabras es grueso: sirve para
  detectar el régimen, no para medir su contenido.

## 4. Datos históricos de analistas gratuitos

Pregunta: ¿se puede obtener gratis, por empresa y año, el BPA medio, máximo y mínimo y el número de analistas en torno a la
publicación del informe? Ventana usada: previsión del BPA de t+1 recogida entre marzo y julio de t+1 (estricta) o entre
enero y julio de t+1 (laxa, que incluye el consenso previo al cuarto trimestre). Todo se consultó con la API CDX de la Wayback Machine,
sin registro, con una petición cada 5 s y pausas largas cuando el archivo estaba saturado.

| Fuente | Qué hay en el archivo | ¿Sirve? |
|---|---|---|
| **Yahoo Finance "analysis"** (`wayback_yahoo_analysis.csv`) | 386 capturas para 48 de 49 empresas, de 0 a 87 por año (unas 7 por empresa en total). En las tres capturas abiertas no hay datos: en 2019-2021 el bloque `earningsTrend` está vacío o con ceros y fechas de 2018, y desde 2023 la página se pinta en el navegador sin datos en el HTML | **No** |
| **MarketScreener**, consenso (piloto con 5 empresas, `wayback_marketscreener_piloto.csv`) | Entre 35 y 42 capturas desde 2020 en 4 de las 5 empresas (1 en la quinta) (casi nada antes). La página da recomendación media, n.º de analistas y precio objetivo medio, máximo y mínimo, **no el rango del BPA** | Solo como proxy (dispersión del precio objetivo), y con huecos: cubre entre 0 y 4 de los 7 años |
| **WSJ research-ratings** | 1 captura (2024) en la empresa probada | No |
| **Consensos compilados por la empresa** (IR, PDF/XLS) (`wayback_ir_consensus.csv`, `wayback_ir_cobertura_docs.csv`) | 14 empresas tienen ficheros de consenso archivados. Fechados por el trimestre o la fecha del nombre del fichero (y si no, por la fecha de captura), cubren **39 informes de 343 (laxa) en 9 empresas**, 35 con la ventana estricta. Por año: 4, 4, 6, 5, 7, 6 y 7 (2018-2024) | **Sí, pero en pocas empresas** |
| Páginas de consenso con widget de terceros (Euroland, Vara/Modular Finance) | La página IR carga un iframe. El archivo guarda pocas capturas, y solo desde 2022 | No en 2018-2021 |

**Piloto** (`piloto_consenso_ir.csv`, `scripts/gw_consensus_pilot.py`): 8 empresas. Cinco tienen documentos, tres no.
Se descargó un documento por informe (32 en total) desde el archivo y se leyó como datos.

| Empresa | Informes con documento | Con BPA de t+1, mínimo/máximo y n.º de analistas | Observaciones |
|---|---|---|---|
| Consumer Staples-05 | 7/7 | 7 | Consenso propio por trimestre: consenso, mínimo, máximo y n.º de estimaciones del BPA subyacente (8-15 analistas) |
| Consumer Staples-03 | 6/7 | 6 | Mediana, mínimo, máximo y n.º de estimaciones por partida (14-17 analistas) |
| Financials-02 | 6/7 | 5 | Media, mediana, máximo y mínimo del BPA y n.º de contribuidores |
| Consumer Staples-01 | 5/7 | 4 | Tripletes del BPA (consenso, bajo y alto; la cabecera no sale como texto) y n.º de instituciones (≈ 22) |
| Telecommunications-01 | 7/7 | 3 | Hasta 2021 la hoja no incluye BPA (solo EBITDA, FCF y DPS). Desde 2022, n.º, máximo, mínimo y media del BPA ajustado: 12-13 analistas; por ejemplo, para t = 2024, media 2,04 y rango 1,94-2,16 |
| Technology-02 | 1/7 | 1 | Solo dos hojas archivadas (2023 y 2024) |
| Financials-07 | 0/7 | 0 | Widget de terceros |
| Utilities-01 | 0/7 | 0 | Sin consenso publicado |

**Estimación de cobertura de las 49 empresas × 7 años.**

- **Cota inferior**: unos 26-30 informes con BPA, rango y n (≈ 8-9 %).
- **Cota superior razonable**: 60-90 informes (≈ 20-25 %). En 22 de los 49 dominios la consulta de dominio completo
  excede el límite de 60 s del servidor CDX. Para ellos solo se exploraron rutas IR típicas, que no siempre son donde se
  guardan los ficheros; una de las empresas del piloto tenía sus hojas en otra ruta.
- En todo caso, la cobertura es selectiva:
  - publican consenso propio sobre todo empresas de consumo, telecomunicaciones y algunos bancos;
  - casi ninguna francesa ni española;
  - las medidas no son homogéneas: mediana o media, mínimo-máximo en vez de desviación típica, y conjuntos de
    contribuidores distintos.

**Conclusión**: no hay una vía gratuita viable para H1 en el panel completo. Los consensos de IR archivados sirven para:

- validar la medida de I/B/E/S en una submuestra (unos 30 informes);
- un análisis de robustez con (máximo − mínimo)/|media| como medida de dispersión.

Para H1 sigue haciendo falta I/B/E/S (WRDS o LSEG).

## 5. Incidencias del corpus detectadas de paso

Afectan al ESGSI del paper, no solo al issue #2.

1. **Financials-04, 2024: documento equivocado.** El PDF es el informe anual de 2024 de una cooperativa cárnica no
   cotizada de otro país con un nombre parecido. No contiene ninguna mención a la empresa del panel (0 autorreferencias,
   frente a cientos en los demás años). Hay que sustituirlo y recalcular los scores de ese informe.
2. **Consumer Staples-04, 2024**: la etiqueta 2024 es el ejercicio cerrado el 30-06-2025. Las etiquetas 2018-2023 son
   los ejercicios cerrados en junio del año de la etiqueta, así que **falta el ejercicio cerrado en junio de 2024** y hay un salto.
3. **Cambios de tipo de documento** (por número de páginas):
   - Health Care-01, 2024: un resumen de 23 páginas frente a unas 400 en el URD de los demás años.
   - Telecommunications-02, 2024: el Form 20-F, sin estado de sostenibilidad.
   - Utilities-01, 2018: las cuentas consolidadas con informe de auditoría (830 páginas).
   - Health Care-03, 2018: solo la DPEF (28 páginas).
   - Financials-08, 2018: 179 páginas frente a una mediana de 546.

   Puede que ya estuviera documentado como heterogeneidad del tipo de documento. Aun así, el primero y el segundo son
   cambios dentro de la misma empresa en 2024, el año del mandato.
4. Los informes del corpus de Technology-01 no contienen la Taxonomía ni el estado no financiero, que la empresa publica en un informe aparte.

## 6. Decisiones abiertas

1. **Exposición a la CSRD.** Hay tres opciones:
   - usar la marca legal (23/26);
   - usar la adopción efectiva de ESRS (43/49 en 2024);
   - o renunciar a la CSRD como fuente de variación y quedarse con el escalón de la Taxonomía por plantilla.

   Se recomienda la tercera, con la primera como robustez.
2. **Países Bajos**: si se aprueba la cláusula de reparación con efecto retroactivo, los informes neerlandeses de 2024 se
   considerarían cumplidores de la CSRD *a posteriori*. Para la identificación cuenta la obligación *ex ante*
   (`csrd = 0`), y así está codificado.
3. **Financials-04, 2024 y Consumer Staples-04, 2024-2025**: hay que corregir el corpus antes de cualquier estimación.
4. **Datos de analistas**: decidir si se completa la recogida de consensos de IR (búsqueda manual de rutas en los 22 dominios
   grandes) como submuestra de validación, o si se espera a I/B/E/S.
5. **Identificadores**: confirmar en WRDS la tabla de enlace ISIN → I/B/E/S ticker, sobre todo para la preferente
   (Consumer Discretionary-04) y el predecesor de Consumer Discretionary-09.

## 7. Reproducción

```bash
M=<clon principal>   # data/ y results/analysis_v3/ se leen de aquí (solo lectura)
python scripts/gw_mandate_evidence.py --pdf-dir $M/data/pdf --firm-ids $M/results/analysis_v3/firm_ids_internal.csv
python scripts/gw_identifiers.py --firm-ids $M/results/analysis_v3/firm_ids_internal.csv --raw-dir <scratch>/dl_ids
python scripts/gw_mandates.py   --firm-ids $M/results/analysis_v3/firm_ids_internal.csv
python scripts/gw_wayback_cdx.py --firm-ids $M/results/analysis_v3/firm_ids_internal.csv --raw-dir <scratch>/dl_cdx --pause 5
python -I scripts/gw_wayback_summary.py --raw-dir <scratch>/dl_cdx --firm-ids $M/results/analysis_v3/firm_ids_internal.csv
python -I scripts/gw_consensus_pilot.py --raw-dir <scratch>/dl_cdx --dl-dir <scratch>/dl_pilot_docs \
       --firms "Telecommunications-01,Consumer Staples-05,Consumer Staples-01,Consumer Staples-03,Financials-02,Technology-02,Financials-07,Utilities-01"
```

- **Entradas internas** (con nombres, no versionadas): `metadata/empresas_isin_internal.csv`,
  `metadata/empresas_dominios_internal.csv` y `metadata/empresas_identificadores_internal.csv`.
- **Descargas** (listados CDX, consensos, respuestas de API): siempre fuera del repositorio, en directorios
  propios, y leídas como datos con `python -I`.
- **Ficheros con nombre o con URL**: no se versiona ninguno.
- **Pendiente para reproducir el piloto completo**: los listados de dos empresas del piloto se completaron
  a mano con consultas CDX adicionales (dominio completo y ruta de descargas). La CDX no es determinista con dominios
  grandes: unas veces responde y otras supera el límite de 60 s.
