# Issue #2: revisión de literatura sobre validación externa y consecuencias financieras

Informe de trabajo (9 de octubre de 2026). Complementa `results/issue2_finance/README.md`
(rama `issue-2-financial-consequences`), que concluyó que H1 y H2 necesitan datos de pago
(I/B/E/S, RepRisk), que con 49 empresas el MDE de H1 es ≈ 0,21-0,35 DT y que la H2 binaria con EF de empresa
no tiene potencia. Aquí se responde a otra pregunta: **qué hace la literatura para validar
externamente medidas textuales de este tipo y qué debe hacer el paper con 49 empresas**.

Se respetan las directrices de `paper/_research/directrices.md`: el ESGSI es un score continuo
(ni etiquetas ni umbrales), no se nombra a ninguna empresa y Lagasio (2024) figura solo como antecedente.
No se ha tocado el paper ni `paper/references.bib`. Las referencias nuevas están en
`results/issue2_lit/refs.bib`.

## 0. Resumen

| Pregunta | Respuesta corta |
|---|---|
| ¿Cómo se validan fuera del texto las medidas de cheap talk o greenwashing? | Con cuatro familias de criterios: **(i) conducta ambiental** (emisiones, violaciones, vertidos), **(ii) controversias o noticias negativas** (RepRisk o el riesgo ESG de un proveedor), **(iii) mercado o analistas** (error o dispersión de previsiones, coste de capital, volatilidad) y **(iv) grupos conocidos o iniciativas** (apoyo a la TCFD, compromisos, sancionados frente a limpios). Casi todos los estudios usan N de **cientos a miles de empresas** y datos de pago. |
| ¿Algún estudio comparable con ≈ 50 empresas valida contra un resultado financiero? | **No se ha encontrado ninguno.** El único análogo europeo de muestra pequeña (36 empresas del DAX, 2020-2022) se publicó en una revista de datos y políticas, no de finanzas, y construye la medida con ratings ESG; no la contrasta con un resultado financiero. |
| ¿Exigen las revistas una validación externa? | Ninguna lo exige por escrito. Pero el antecedente directo (IRFA, 2024) se publicó **sin** validación externa, y las versiones con criterio externo de la misma línea de trabajo acabaron en revistas de mayor nivel (*JBF* frente a *FRL*). Los manuales de análisis textual piden pruebas de validez de constructo, que no tienen por qué ser financieras. |
| ¿Qué resultado externo es creíble y viable gratis con 49 × 7? | Ninguno de conducta ni financiero. Lo único con potencia y coste razonables es una **validez convergente y discriminante con un instrumento independiente y publicado**: los clasificadores abiertos de ClimateBERT (especificidad, compromiso y sentimiento), aplicados a los mismos informes. Ese instrumento sí tiene validación externa publicada (emisiones y noticias negativas en ≈ 14.600 informes anuales). |
| **Recomendación** | **(c) con un añadido acotado de (b).** H1 y H2 quedan como limitación y trabajo futuro, con su MDE. Se añade una comprobación breve (≈ ½ página) de validez convergente y discriminante frente a ClimateBERT, presentada como validez de constructo y **no** como validación externa de conducta. Si el autor consigue WRDS, H1 entra como un único contraste preespecificado en un apéndice, no como sección. |

## 1. Método de la revisión

- Búsqueda web y en catálogos (Crossref, OpenAlex, arXiv, Hugging Face) a partir de la lista del encargo y de las
  citas cruzadas de un survey reciente de detección de greenwashing en texto (arXiv 2502.07541).
- **Verificación.** Cada referencia de `refs.bib` se comprobó contra Crossref por DOI: título, autores, revista,
  volumen, páginas y año. La entrada de arXiv y el informe del JRC se comprobaron en su página oficial.
  Los resúmenes se leyeron en OpenAlex o en la editorial. Cuando un dato (N, proveedor) solo aparece en
  resúmenes secundarios se indica en la tabla con *(sec.)*. Cuando no se pudo confirmar, *(n.v.)*.
- **Mahmood et al.**, citado en el encargo, **no se ha podido identificar**: ninguna búsqueda devolvió un artículo
  de ese autor que proponga o valide una medida textual de greenwashing. No se incluye. Hace falta el título o el DOI.
- No se ha leído el texto completo de la mayoría de los artículos (los PDF de Elsevier y Wiley devuelven 403). Las cifras
  de N proceden de resúmenes o de versiones de trabajo. Ninguna cifra de tamaño de efecto de la literatura se usa en la
  argumentación porque no se ha podido comprobar en el texto.

## 2. Tabla resumen de la literatura

Leyenda. **Datos**: G = gratuitos, P = de pago o con licencia. **¿49 × 7?**: si el mismo diseño es realista con
este panel.

### 2.1 Medidas textuales con criterio externo

| Referencia | Medida | Muestra | Criterio externo | Datos | Identificación | Hallazgo | ¿49 × 7? |
|---|---|---|---|---|---|---|---|
| Bingler et al. (2022), *Finance Research Letters* | Cheap talk = proporción de compromisos climáticos no específicos (ClimateBERT) | Informes de empresas que apoyan la TCFD (N exacto n.v.) | Apoyo público a la TCFD (grupo conocido) | Informes G; lista TCFD G | Comparación de grupos o asociación de panel | El apoyo a la TCFD va con **más** cheap talk y con información poco material | Parcial: la lista TCFD es gratuita, pero sería casi constante dentro de empresa en 49 grandes cotizadas |
| Bingler et al. (2024), *Journal of Banking & Finance* | ClimateBERT-CTI (el mismo índice) | 14.584 informes anuales del MSCI World, 2010-2020 | Iniciativas climáticas, crecimiento de emisiones y noticias negativas | Emisiones P (Urgentem, *sec.*); noticias P (RepRisk, *sec.*) | Panel con controles y EF | Solo la implicación activa (engagement) va con menos cheap talk. El cheap talk se asocia a más noticias negativas y a mayor crecimiento de emisiones | No: necesita datos de pago y N grande. Su instrumento es abierto (§5) |
| Gorovaia y Makrominas (2025), *European Financial Management* | Longitud, tono, legibilidad y contenido ambiental (léxico) de informes RSC | Empresas con y sin violaciones ambientales (N n.v.) | Violaciones ambientales (sancionados frente a historial limpio) | Violaciones: fuente n.v. | Grupos conocidos más cambio tras la violación | Los infractores publican informes más largos, más positivos y menos legibles, y los cambian justo después de la violación | Débil: sanciones raras y con retraso en la UE. Violation Tracker Global cuesta ≥ 25 USD/mes (issue2_finance §2.2) |
| Li et al. (2024), *International Review of Financial Analysis* | Tono positivo anormal de informes ESG | Empresas chinas (N n.v.) | Riesgo ESG futuro, "de fuente pública" y "de medios" | Proveedor n.v. | Panel, t → t+1 | El tono anormal anticipa más riesgo ESG, solo ambiental y solo el de fuente pública, no el de medios | No: es la H2 del issue, con N grande y datos de pago |
| Coen et al. (2022), *Business Strategy and the Environment* | Tipos de "talk" climático (análisis textual asistido) | 725 informes de sostenibilidad | Cambio de emisiones GEI a 10 años ("walk") | Emisiones: fuente n.v. | Regresión de walk sobre talk | Resultados mixtos: el talk operativo predice mejoras y el reactivo va con peor desempeño | Parcial: con EU ETS gratis solo para las empresas emisoras (§3) |
| Lu y Jagoda (2023), *International Journal of Accounting and Finance* | Tono de informes de sostenibilidad | Extractivas de EE. UU. con informe independiente, 2010-2018 | Desempeño ambiental futuro | n.v. | 3SLS | Tono más positivo, peor desempeño ambiental futuro | No: un solo sector |
| Cho et al. (2010), *Accounting, Organizations and Society* | Optimismo y certeza (DICTION) en la divulgación ambiental del 10-K | Corte transversal de EE. UU. (N n.v.) | Desempeño ambiental (peores frente a mejores) | n.v. | Grupos conocidos | Los peores desempeños escriben con más optimismo y menos certeza | Sí en diseño (corte transversal), pero el criterio de desempeño es de pago |
| Melloni et al. (2017), *Journal of Accounting and Public Policy* | Concisión, completitud y balance (tono) de informes integrados | Primeros adoptantes del IR (N pequeño, n.v.) | Desempeño financiero y social | Ratings P | Determinantes | Con peor desempeño, informes más largos, menos legibles y más optimistas | Sí: precedente de muestra pequeña, pero con ratings de pago |
| Motz et al. (2026), *Data & Policy* | Greenwashing Tendency Score: sentimiento y alineamiento texto-rating | **36 empresas del DAX, 2020-2022** | Ninguno fuera de la medida: los ratings ESG forman parte de ella | Ratings P | Descriptivo | Gran variación entre empresas | Es el único análogo de N pequeño y **no** contrasta un resultado financiero |
| Sautner et al. (2023), *Journal of Finance* | Exposición climática en conference calls | > 10.000 empresas, 34 países, 2002-2020 | Empleo verde, patentes, precios de opciones y acciones | P | Validación por resultados reales y de mercado | La medida predice resultados reales y está en precios | No: es el estándar de validación de *JF*, con N de miles |
| **Lagasio (2024)**, *International Review of Financial Analysis* (antecedente) | ESGSI (tono menos sustancia) | 749 informes de sostenibilidad, corte transversal 2023 | **Ninguno.** ANOVA del índice por continente, sector, tamaño y cuartil del rating ESG | Rating P | Descriptivo | Diferencias por grupo con η² de 0,02-0,07 | — (precedente de publicación sin validación externa; véase `paper/_research/lagasio_source.md` §7) |

### 2.2 Brecha divulgación-desempeño sin texto (o con texto secundario)

| Referencia | Medida | Muestra | Resultado | Datos | Hallazgo | ¿49 × 7? |
|---|---|---|---|---|---|---|
| Delmas y Burbano (2011), *California Management Review* | Marco conceptual | — | — | — | Tipología y determinantes del greenwashing (comunicación positiva con desempeño pobre) | Marco teórico, citable sin datos |
| Kim y Lyon (2015), *Organization Science* | Reducciones de GEI declaradas frente a reales (greenwash y brownwash) | Eléctricas de EE. UU., programa voluntario 1605(b) del DOE (*sec.*) | Determinantes (crecimiento, desregulación, beneficios) | G (registro público de EE. UU.) | El crecimiento favorece la exageración y la desregulación con beneficios bajos la modestia indebida; el escrutinio lo atenúa | No (EE. UU.), pero muestra que la brecha tiene **dos colas**, a favor de un score continuo y no de etiquetas |
| Marquis et al. (2016), *Organization Science* | Divulgación selectiva | 4.750 empresas, 45 países, 2004-2007 | Determinantes institucionales | P | Las empresas más contaminantes, con más escrutinio, divulgan de forma menos selectiva | No |
| Yu et al. (2020), *Research in International Business and Finance* | Score relativo a pares: divulgación (Bloomberg) frente a desempeño (Asset4) | 1.925 grandes cotizadas, 47 países | Determinantes de gobierno corporativo | P | Consejeros independientes, inversores institucionales, país menos corrupto y cotización cruzada disuaden el greenwashing | No. **Arquitectura análoga al ESGSI** (diferencia de posiciones relativas), pero con ratings y no texto |
| Clarkson et al. (2008), *Accounting, Organizations and Society* | Calidad de divulgación ambiental (índice GRI) | Industrias contaminantes de EE. UU. (N n.v.) | Desempeño ambiental (TRI) | G en EE. UU. | Relación positiva, contra la teoría de la legitimidad | No |
| Hummel y Schlick (2016), *Journal of Accounting and Public Policy* | Calidad de divulgación de sostenibilidad | Empresas europeas (N n.v.) | Desempeño de sostenibilidad (rating) | P | Los buenos desempeños divulgan con calidad y los malos con cantidad de baja calidad | No |
| Duchin et al. (2025), *Journal of Finance* | Venta de plantas contaminantes | Plantas de EE. UU. (TRI) | Contaminación posterior, discurso en calls, ratings y retornos | G (TRI) y P | Las emisiones no bajan tras la venta, pero el vendedor habla más de sostenibilidad y gana rating | No. Muestra que el **rating ESG no sirve de criterio** de conducta |
| Raghunandan y Rajgopal (2022), *Review of Accounting Studies* | Discurso ESG de fondos frente a cartera | Fondos de EE. UU., 2010-2018 | Violaciones legales y emisiones de las participadas | G (Violation Tracker de EE. UU.) y P | Los scores ESG se correlacionan con la **cantidad de divulgación**, no con el cumplimiento ni con las emisiones | No. Advierte contra validar un score de divulgación con ratings ESG (circularidad) |
| Li et al. (2023), *Business Strategy and the Environment* | Greenwashing (medida no textual) | 735 empresas chinas, 2.816 obs., 2013-2017 | Desempeño financiero | P | Efecto positivo, moderado por la regulación y los medios | No |

### 2.3 Analistas, coste de capital y riesgo (H1)

| Referencia | Medida | Muestra | Resultado | Datos | Hallazgo | ¿49 × 7? |
|---|---|---|---|---|---|---|
| Lang y Lundholm (1996), *The Accounting Review* | Calidad de divulgación (ratings de analistas) | EE. UU. | Cobertura, dispersión y precisión | P (I/B/E/S) | Mejor divulgación, más cobertura y menos dispersión | Base teórica de H1 |
| Dhaliwal et al. (2012), *The Accounting Review* | Publicación de un informe RSC independiente | 31 países | Error de previsión | P (I/B/E/S) | Menos error, más en países orientados a los stakeholders | No (N grande); la variable es binaria y no textual |
| Muslu et al. (2019), *Journal of Business Ethics* | Score de narrativa del informe RSC: tono, legibilidad, longitud, cifras y horizonte | EE. UU. (N n.v.) | Precisión de analistas | P | Una narrativa de más calidad va con previsiones más precisas, solo en reportantes experimentados | El precedente textual más cercano a H1, con N grande |
| García-Sánchez et al. (2021), *Business & Society* | Decoupling RSC (divulgación frente a desempeño) | EE. UU., 7.681 obs., 2006-2015 | Error de previsión, coste de capital, acceso a financiación | P | Más brecha, más error, mayor coste de capital y menos acceso | No: es el contraste H1 publicado, con 7.681 obs. |
| Kölbel et al. (2017), *Strategic Management Journal* | Cobertura mediática de irresponsabilidad (RepRisk) | 539 empresas, 2008-2013 | Riesgo financiero (CDS) | P | Más cobertura negativa, más riesgo de crédito | No |

### 2.4 Divergencia de ratings ESG

| Referencia | Muestra | Hallazgo | Relevancia | ¿49 × 7? |
|---|---|---|---|---|
| Berg et al. (2022), *Review of Finance* | 6 agencias | La divergencia se debe a la medición (56 %), el alcance (38 %) y los pesos (6 %), con efecto evaluador | Un rating ESG no es un criterio externo fiable | No hay ratings múltiples históricos gratis |
| Christensen et al. (2022), *The Accounting Review* | EE. UU., EF de empresa y DiD con mandatos | **Más divulgación ESG, más desacuerdo**; el desacuerdo va con más volatilidad | La relación divulgación-rating no es monótona | No |
| Kimbrough et al. (2024), *European Accounting Review* | EE. UU. | El informe voluntario reduce el desacuerdo, pero un **tono más positivo lo aumenta** | El análogo más directo del componente SEN del ESGSI | No: datos de varios proveedores |
| Gibson Brandon et al. (2021), *Financial Analysts Journal* | S&P 500, 7 proveedores, 2010-2017 | El desacuerdo se asocia a mayores retornos (prima de riesgo) | Resultado de mercado | No |

### 2.5 Mandatos europeos como cuasi-experimento

| Referencia | Diseño | Hallazgo | Relevancia para el paper |
|---|---|---|---|
| Grewal et al. (2019), *Management Science* | Estudio de eventos sobre la aprobación de la Directiva 2014/95 (NFRD) | Reacción media −0,79 %, más negativa en empresas con divulgación y desempeño previos débiles | El mandato es un choque común, identificable solo con exposición heterogénea |
| Fiechter et al. (2022), *Journal of Accounting Research* | DiD: empresas dentro o fuera del alcance de la NFRD | Las afectadas aumentan su actividad RSC, más allá de un posible greenwashing | Su identificación usa empresas **fuera** del alcance; el panel de 49 está todo dentro (issue2_finance §5) |
| Christensen et al. (2021), *Review of Accounting Studies* | Revisión | Efectos esperados de los mandatos en mercados, stakeholders y conducta | Marco para leer el confound de mandato de §4.2 del paper |

### 2.6 Validación de medidas textuales (método)

- **Loughran y McDonald (2016)**, *Journal of Accounting Research*: el análisis textual es menos preciso que los métodos
  cuantitativos y hay que demostrar que la medida mide lo que dice.
- **Bochkay et al. (2023)**, *Contemporary Accounting Research*: dedica un apartado a cómo establecer la validez de
  constructo de una medida textual: validez aparente, convergente con medidas alternativas y predictiva.
- **Grimmer y Stewart (2013)**, *Political Analysis*: los métodos automáticos no sustituyen a la validación, que debe hacerse en cada aplicación.
- **Adcock y Collier (2001)**, *American Political Science Review*: distingue la validación de contenido, la convergente o discriminante
  y la nomológica. La validación nomológica (relaciones con resultados que la teoría predice) es solo una de las tres.
- El paper ya cita a Campbell y Fiske (1959) para la validez convergente con QUANT. La comprobación propuesta en §5 es una matriz
  multirrasgo-multimétodo en sentido estricto.

## 3. Qué resultados externos son realistas con 49 empresas × 7 años

Referencia de potencia (issue2_finance §3): con 49 clústeres, el MDE para un resultado continuo es
**0,21-0,35 DT por DT del score**. En la H2 binaria con EF de año hace falta una OR ≈ 1,8-2,0, y con EF de empresa no se alcanza el 80 %.
En un subconjunto de G empresas, el MDE escala aproximadamente con √(49/G). Es un escalado, no una simulación.

| Resultado externo | Fuente y coste | Cobertura en el panel | Potencia | Problemas de interpretación | Veredicto |
|---|---|---|---|---|---|
| Dispersión y error de analistas (H1) | I/B/E/S vía WRDS o LSEG (P) | 49/49 previsibles | MDE 0,21-0,35 DT | El mandato mueve a la vez score y dispersión; en la literatura los efectos se documentan con miles de obs. | Solo con WRDS, como **un único contraste preespecificado** |
| Controversias (H2) | RepRisk vía WRDS (P) | Probable 49/49 | Recuento con EF de año: aceptable; binaria con EF de empresa: no | En Li et al. (2024) la señal aparece en el riesgo de fuente pública y **no** en el de medios | Solo con WRDS, como recuento y fuera del cuerpo principal |
| Emisiones verificadas (talk-walk) | EU ETS / EUTL (**G**), enlazado a empresas con JRC-EU ETS-FIRMS (Letout, 2022) o a mano | Material en ≈ 8 empresas (energía, utilities, química, construcción y materiales) y ≈ 13 con automóvil. Recuento por supersector, no verificado en el EUTL. Banca y seguros, 0 | MDE ≈ 0,4-0,7 DT con G = 8-13 | Las instalaciones de la UE no son el perímetro del grupo; desinversiones (Duchin et al., 2025) | **No** para el paper; como mucho descriptivo |
| Objetivos SBTi (validados o comprometidos) | SBTi Target Dashboard (**G**) | Casi todas las no financieras tendrán objetivo en 2024; poca variación | Entre empresas, casi un corte transversal de 49 | Es *talk* verificado, no *walk*; más sustancia por construcción (el score baja por mecánica) | No como validación; quizá descriptivo |
| Puntuación CDP clima | cdp.net (consulta pública por empresa, sin descarga masiva verificada) | Probable alta | Entre empresas | CDP puntúa sobre todo la **calidad de la divulgación** → circular | No |
| Gestión TPI (MQ) | TPI Centre (acceso y términos por verificar) | Sectores intensivos y bancos | Pequeña | Datos de LSEG/FTSE Russell construidos desde la propia divulgación → circular | No |
| Sanciones y violaciones | Violation Tracker Global (consulta G, descarga ≥ 25 USD/mes) | Pocas por año | Muy baja (eventos raros) | Retraso y lagunas en la UE | No (opcional como robustez cualitativa) |
| Ratings ESG y su divergencia | MSCI, Sustainalytics, LSEG… (P, sin histórico gratis) | — | — | Berg et al. (2022) y Raghunandan y Rajgopal (2022): el rating mide divulgación y diverge entre proveedores | No |
| Volatilidad idiosincrática y asimetría negativa | Yahoo (**G**) | 49/49 | MDE 0,21-0,35 DT | Ya estimado: nada robusto (issue2_finance §6) | No en el cuerpo. Como mucho una frase con el MDE |
| **Instrumento textual independiente (ClimateBERT)** | Modelos abiertos en Hugging Face (Apache-2.0): `climatebert/distilroberta-base-climate-{detector,commitment,specificity,sentiment}` (**G**) | 343/343 informes (solo pasajes climáticos) | Correlaciones ≥ 0,3 detectables con 49 clústeres (mismo orden que el MDE de H1) | Mismo texto (varianza de método compartida); solo la parte climática | **Sí: la única opción con potencia, gratuita y defendible** |

## 4. Qué esperan las revistas de finanzas aplicadas

Las páginas de normas de autor de Elsevier devolvieron 403. Lo que sigue se apoya en lo que publican las revistas, no en
normas escritas.

- **International Review of Financial Analysis** (la revista del antecedente). Publicó Lagasio (2024): una medida textual en
  corte transversal **sin ningún criterio externo**, con ANOVA de correlatos. También publicó Li et al. (2024), que relaciona tono y
  riesgo ESG futuro con datos de pago y N grande. Por tanto, una contribución de medida pura es publicable en IRFA, pero el listón
  de la revista incluye ya un artículo con validación predictiva sobre el mismo tipo de medida. Es la opción más natural para este paper.
- **Finance Research Letters**. Cartas de < 2.500 palabras (según Wikipedia y una ficha universitaria; la guía de Elsevier no
  se pudo abrir). Bingler et al. (2022) publicaron allí la versión con un criterio de grupo conocido (apoyo a la TCFD), y la
  versión con emisiones y noticias negativas pasó al *JBF*. El paper actual (≈ 15 páginas técnicas) no encaja sin reducirlo a un resultado.
- **Journal of Corporate Finance**. Su línea es la identificación causal sobre decisiones corporativas. Un panel de 49 empresas sin
  variación exógena en el regresor y con el mandato colineal con el año no encaja ni con H1/H2. **No se recomienda.**
- **Business Strategy and the Environment**. Publica medidas y determinantes de greenwashing, a menudo en paneles chinos grandes, y
  estudios talk-walk con emisiones (Coen et al., 2022, con 725 informes). Acepta trabajos de medida, pero el estándar implícito es N de cientos.
- **Sobre la validación.** Ninguna revista exige un resultado financiero. Lo que la literatura de método pide (Loughran y McDonald, 2016;
  Bochkay et al., 2023; Adcock y Collier, 2001) es **evidencia de validez de constructo**, y la validez convergente con una medida
  alternativa es una de sus formas aceptadas. La objeción previsible de un evaluador ante un paper de medida es "¿mide esto
  lo que dice?", más que "¿predice la dispersión de analistas?". Un contraste financiero de baja potencia con resultado nulo
  no responde a ninguna de las dos.

## 5. Recomendación: (c), con un añadido acotado de (b)

**(a) Sección completa H1/H2: no.**
1. Necesita datos de pago que hoy no están (issue2_finance §2).
2. Con 49 clústeres el MDE de H1 es 0,21-0,35 DT. La literatura que documenta estas relaciones (García-Sánchez et al., 2021;
   Muslu et al., 2019; Dhaliwal et al., 2012; Bingler et al., 2024; Li et al., 2024) trabaja con miles de observaciones. No
   hay ningún precedente de N ≈ 50 con un resultado financiero.
3. El mandato es colineal con el año y todo el panel está dentro del alcance de la NFRD, así que no hay grupo de control al estilo
   de Fiechter et al. (2022). Las especificaciones C y D de issue2_finance §5 solo usan la variación de régimen entre 4 grupos de 5 a 20 empresas.
4. Excede el presupuesto de páginas de las directrices (§9: ≤ 15 páginas técnicas; resultados 5, robustez 2, limitaciones 0,75).

**(b) Subsección exploratoria con datos gratuitos de conducta o mercado: no.**
- Los proxies de mercado ya se estimaron y no dan nada robusto: 1 de unas 50 pruebas cerca del 5 % (issue2_finance §6).
- Las emisiones del EU ETS cubren de forma material ≈ 8-13 empresas (MDE ≈ 0,4-0,7 DT).
- SBTi, CDP y TPI son circulares: miden compromiso o calidad de divulgación, que es justo lo que el score lee.
- Un nulo de baja potencia en el cuerpo del paper invitaría a leerlo como "el score no predice nada", cuando lo único que dice es que el panel no detecta efectos menores de ≈ 0,3 DT.

**(c) Limitación y trabajo futuro: sí, reforzada.** Redacción sugerida para la viñeta *No external criterion* de §6:
nombrar H1 y H2 como contrastes de validez predictiva pendientes, citar la evidencia de N grande que los motiva
(Bingler et al., 2024; Li et al., 2024; García-Sánchez et al., 2021) y dar el MDE con 49 empresas, sin presentar ninguna estimación.

**Añadido acotado (≈ ½ página, dentro del presupuesto de robustez): validez convergente y discriminante con ClimateBERT.**
- **Qué.** Aplicar a los pasajes climáticos de los 343 informes ya extraídos los clasificadores abiertos de Bingler et al.:
  - detector de texto climático;
  - compromiso (sí/no) y especificidad (específico/no específico), con los que se calcula el índice de cheap talk
    de Bingler et al. (2022, 2024): proporción de párrafos de compromiso no específicos;
  - sentimiento (oportunidad, neutro, riesgo).
- **Contraste preespecificado (matriz multirrasgo-multimétodo, Campbell y Fiske, 1959, ya citado).**
  - Convergente: el componente de sustancia (Z(SUS), y QUANT en el ESGSI extendido) debería ir en sentido contrario a la proporción
    no específica, y el ESGSI calculado sobre los mismos pasajes en el mismo sentido que el CTI.
  - Discriminante: Z(SEN) debería correlacionar más con el sentimiento de ClimateBERT que con la especificidad, y Z(SUS) al revés.
  - Correlaciones de Spearman a nivel de informe y pendiente estandarizada con SE agrupados por empresa y EF de año, igual que el resto de robustez.
  - Todo como scores continuos, sin umbrales.
- **Por qué es la opción más creíble y viable.**
  1. El instrumento es independiente: lo entrenaron otros autores sobre otros datos y es neuronal, no de diccionario.
  2. Es gratis y cubre 343/343 informes.
  3. Su potencia es suficiente para correlaciones del orden de 0,3.
  4. Es el único puente disponible hacia la validez externa: el CTI está validado contra emisiones y noticias negativas
     en ≈ 14.600 informes (Bingler et al., 2024). El paper debe decir que esa validez **no se transfiere** automáticamente.
- **Límites que hay que declarar.**
  - Es validez de constructo, no de conducta: el texto es el mismo, así que hay varianza de método compartida.
  - Solo cubre la parte climática del pilar ambiental.
  - ClimateBERT se entrenó en inglés: hay que comprobar el idioma de los informes extraídos.
  - Si la convergencia sale débil, el resultado es igualmente informativo y debe publicarse.
- **Coste estimado.** Una pasada de inferencia de modelos DistilRoBERTa sobre los pasajes ya extraídos. No exige datos nuevos.

**Si el autor consigue WRDS.** H1 (dispersión de I/B/E/S en el mes posterior al informe, especificación B, con el MDE declarado)
puede ir como **un único contraste preespecificado en un apéndice en línea**, nunca en el cuerpo y sin búsqueda de especificaciones.
H2 se haría con el recuento de incidentes de RepRisk, separando incidentes de fuente regulatoria y de medios, siguiendo
el hallazgo de Li et al. (2024). La conclusión de (c) no cambia: aunque haya datos, la potencia no da para una sección.

## 6. Pendiente y no verificado

1. **Mahmood et al.**: sin identificar. Hace falta el título o el DOI.
2. Tamaños muestrales marcados *n.v.* (Bingler et al., 2022; Gorovaia y Makrominas, 2025; Li et al., 2024; Cho et al., 2010; Muslu et al., 2019;
   Melloni et al., 2017; Hummel y Schlick, 2016) y proveedores marcados *(sec.)* (Urgentem y RepRisk en Bingler et al., 2024). Hay que leer el texto completo antes de citarlos con cifras.
3. Guías de autor de IRFA, FRL, JCF y BSE: no accesibles (403). El límite de 2.500 palabras de FRL procede de fuentes secundarias.
4. Recuento de empresas con instalaciones en el EU ETS: estimado por supersector, sin consultar el EUTL.
5. Acceso y términos de uso de los datos del TPI y descarga histórica del CDP: sin verificar.
6. Idioma de los pasajes extraídos (para ClimateBERT): comprobar en el corpus.

## 7. Fuentes web consultadas (además de los DOI de `refs.bib`)

- Survey de detección de greenwashing en texto: https://arxiv.org/abs/2502.07541
- Modelos ClimateBERT: https://huggingface.co/climatebert y https://huggingface.co/climatebert/distilroberta-base-climate-specificity
- Working paper de Bingler et al. (SFI 22-54): https://www.sfi.ch/en/publications/n-22-54-cheap-talk-in-corporate-climate-commitments-the-effectiveness-of-climate-initiatives
- Resumen de Bingler et al. en RePEc: https://ideas.repec.org/p/chf/rpseri/rp2201.html
- Gorovaia y Makrominas en SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4904389
- Muslu et al.: https://digitalcommons.kennesaw.edu/facpubs/3978
- Li et al. (2024), resumen: https://cem.njfu.edu.cn/2024/1004/c1670a61861/page.htm
- Yu et al. (2020), versión aceptada: https://eprints.bbk.ac.uk/id/eprint/30701/
- JRC-EU ETS-FIRMS: https://publications.jrc.ec.europa.eu/repository/handle/JRC127805
- SBTi Target Dashboard: https://sciencebasedtargets.org/target-dashboard
- TPI Centre: https://www.transitionpathwayinitiative.org/corporates
- Finance Research Letters (límite de extensión): https://en.wikipedia.org/wiki/Finance_Research_Letters
