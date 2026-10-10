# Issue #4: validez convergente con ClimateBERT (medidas y corpus corregidos)

Informe de trabajo (10 de octubre de 2026). Rehace el piloto del issue #2 (rama `issue-2-climatebert`)
con las medidas corregidas de la rama `fix/corpus-and-measures`: SEN = polaridad de Loughran-McDonald
sobre formas exactas del texto crudo, con negación; QUANT = solo cifras; HEDGE = Uncertainty +
WeakModal de L&M sin *risk/risks* ni "May" como mes. El corpus también es el corregido: cuatro
informes de 2024 sustituidos. Todo son scores continuos, sin umbrales ni etiquetas, y las empresas
aparecen solo con su identificador anónimo (`firm_label`).

## Resumen

| Pregunta | Respuesta |
|---|---|
| Cobertura | 343/343 informes; 318.258 pasajes, que cubren el 84,5 % de las palabras extraídas. 118.258 son climáticos (37,2 %) y 59.057 son de compromiso. Un informe de 2018 no tiene ningún pasaje climático, así que el panel con CTI tiene 342 informes |
| Coste | La caché por hash del texto reutilizó 279.444 de los 283.619 textos distintos. Solo hubo que clasificar 4.175 pasajes con el detector y 1.878 con cada uno de los otros tres modelos, que son los de los cuatro informes de 2024 sustituidos (entre un 64 y un 96 % de pasajes nuevos en cada uno; en el resto de informes no cambió ninguno). Inferencia: 3,6 min de CPU; segmentación 1,2 min; SUS por alcance (spaCy) 10 min; análisis 2 min |
| Tono | **Converge, y algo más que en el piloto.** Z(SEN) frente al sentimiento climático neto: ρ = 0,58 [0,42; 0,72]; intra-empresa r = 0,51 [0,30; 0,69]; entre empresas r = 0,62 [0,42; 0,77]. Con SEN medido solo sobre los pasajes climáticos, ρ = 0,66 [0,53; 0,77]. A nivel de pasaje, la ρ mediana dentro de cada informe es 0,42 y es positiva en el 99,7 % de los informes. Cumple el criterio de Campbell-Fiske en las cuatro vistas |
| Sustancia | **La densidad no es especificidad; QUANT sí converge, aunque débilmente.** Z(SUS) frente a la cuota de pasajes específicos: ρ = −0,11 [−0,29; 0,06]; intra-empresa r = −0,35 [−0,49; −0,20]. Con SUS medido solo sobre los pasajes climáticos, ρ = −0,31. Z(SUS) frente a la cuota climática: ρ = 0,65 [0,52; 0,76]. Z(QUANT) frente a la cuota específica: ρ = 0,20 [0,01; 0,39]; a nivel de pasaje, 0,45 (positiva en el 99 % de los informes) |
| Matización | **Sin convergencia.** Z(HEDGE) frente al CTI: ρ = 0,15 [−0,08; 0,35]; intra-empresa r = 0,03. Con HEDGE solo sobre pasajes climáticos, ρ = 0,03. HEDGE sí va con un tono menos optimista (ρ = −0,35 con el sentimiento neto) |
| Compuesto | **No converge.** ESGSI frente al CTI: ρ = −0,06 [−0,27; 0,14]. ESGSI_ext frente al CTI: ρ = 0,02 [−0,20; 0,23] |
| ¿Ve ClimateBERT la ruptura de 2024? | **Sí, pero otra.** Su sentimiento neto cae −0,69 DT en 2024 frente a 2018-2023 (EF de empresa; [−0,88; −0,50]) porque hay menos pasajes de *oportunidad*, no más de riesgo. La caída de SEN (−0,30 DT) **desaparece** al quitar 24 palabras de temas ESRS (−0,02; p = 0,87). La de ClimateBERT sigue al quitar los pasajes con terminología ESRS (−0,60). En 2024 también saltan el CTI (+0,36 DT) y la cuota climática, y caen la especificidad y la cuota de compromiso |
| Tendencias (EF de empresa) | CTI plano (+0,002/año, p = 0,59). Sube la cuota climática (+0,028/año, p < 10⁻¹²). Cae el sentimiento neto (−0,022/año, p < 10⁻⁵), con una tendencia que ya existía en 2018-2023 (−0,014/año, p = 0,004). La especificidad y la cuota de compromiso solo caen por 2024 (en 2018-2023, p = 0,54 y 0,65) |
| **¿Al paper?** | Sí, como un párrafo en §5 (borrador en `draft_robustness_climatebert.tex`, ≈ 230 palabras). Valida el tono, acota la lectura del compuesto y da una prueba independiente de la lectura de la ruptura de 2024 |

## 1. Datos, segmentación y reutilización de la caché

- **Texto y pasajes.** Igual que en el piloto (`scripts/cb_segment.py`). El texto son las zonas ESG
  de `data/chunks_lexical_v3_sec`, re-extraídas el 10-10-2026. Un pasaje es un párrafo de 30 o más
  palabras; los de más de 250 se parten por frases. Salen 318.258 pasajes, con una mediana de
  60 palabras (p5 32, p95 178) y una mediana de 793 por informe (mínimo 19). Idioma: 99,9 % inglés
  en una muestra de 20.523 pasajes con langdetect; ningún informe baja del 95 %
  (`segmentation.json`).
- **Caché por hash.** El piloto guardaba las predicciones por posición (`doc_key`, `pid`), así que un
  cambio de segmentación las invalidaba. Ahora la clave es el SHA-1 del texto exacto del pasaje
  (`scripts/cb_classify.py --seed-from`): se importaron las predicciones del piloto y solo se
  infirieron los textos nuevos. Los pasajes nuevos (4.218) están todos en los cuatro informes de
  2024 sustituidos; ningún otro informe cambió de texto. Comprobación: al re-inferir 300 textos de la
  caché por modelo, la diferencia máxima de probabilidad es < 10⁻⁶ y la clase coincide en el 100 %
  de los casos (`cache_check.json`).
- **Tiempos** (i7-9700K, 8 hilos, ≈ 44-47 pasajes/s): detector 89 s; compromiso, especificidad y
  sentimiento, 43 s cada uno. Sin la caché habrían hecho falta ≈ 3,4 h.
- **Comprobación de las medidas.** SEN recalculado con `ESGSIAnalyzer.sen_counts` sobre el
  `relevant_text` de cada JSON reproduce `SEN_Score` (diferencia máxima 5·10⁻⁵, que es el redondeo;
  r = 1,0000 con Z_SEN). El SUS sobre los pasajes, con el mismo TextProcessor del pipeline, tiene
  ρ = 0,974 con `SUS_density`, que se mide sobre todo el texto extraído.

## 2. Instrumento y definición

- **Modelos.** `climatebert/distilroberta-base-climate-{detector,commitment,specificity,sentiment}`,
  con licencia Apache-2.0 y la revisión fijada en `scripts/cb_common.py`. El detector se aplica a todos
  los pasajes; los otros tres modelos, solo a los climáticos (p > 0,5). La clase predicha es el
  argmax.
- **CTI** = pasajes climáticos de compromiso **no específicos** / pasajes climáticos de compromiso.
  La definición está verificada en el resumen de Bingler et al. (2024, JBF), "the share of
  imprecise versus precise climate commitments", y en la presentación de los autores
  (CTI = COMMIT ∩ NONSPEC / COMMIT). Volví a intentar leer el texto completo (ScienceDirect, SSRN y
  la presentación) y todos devuelven 403. Siguen sin verificar los detalles de segmentación y
  filtrado de párrafos de Bingler et al. y un posible mínimo de compromisos por informe; conviene
  leerlos antes de llamar "fiel" a la réplica. Usar probabilidades en lugar de clases (CTI_prob) da
  ρ = 0,96 con el CTI.
- **Otras medidas por informe**:
  - cuota climática: pasajes climáticos / pasajes;
  - cuota específica: específicos / climáticos;
  - sentimiento neto: (oportunidad − riesgo) / climáticos; su versión con probabilidades tiene
    ρ = 0,995 con él;
  - cuotas de oportunidad y de riesgo;
  - cuota de compromiso.
- **Alcance.** SEN, QUANT y HEDGE se recalculan con las funciones del pipeline solo sobre los pasajes
  climáticos (SEN_clim, QUANT_clim, HEDGE_clim), y SEN también sobre los no climáticos
  (SEN_nonclim). SUS se recalcula con el TextProcessor del pipeline sobre los climáticos y los no
  climáticos (SUS_clim, SUS_nonclim; `scripts/cb_sus_scope.py`).

Descriptivos (342 informes): CTI con media 0,49 (DT 0,12); cuota específica 0,35 (0,10); sentimiento
neto 0,15 (0,18); cuota climática 0,34 (0,14); cuota de compromiso 0,50 (0,11). Hay 11 informes con
menos de 10 pasajes de compromiso (mínimo 2); excluirlos no cambia nada (ρ del CTI con ESGSI = −0,03).

## 3. Correlaciones y matriz multirrasgo-multimétodo

En el agregado se da la ρ de Spearman y en las demás vistas la r de Pearson. Los IC al 95 % salen de
un bootstrap de empresas (2.000 réplicas, percentiles). Las vistas son:

- *intra*: desviaciones respecto a la media de cada empresa;
- *entre*: las 49 medias de empresa;
- *doble EF*: desviaciones de empresa y de año.

La tabla completa está en `correlations.csv`.

| ClimateBERT | Score | Agregado ρ | Intra r | Entre r | Doble EF r |
|---|---|---|---|---|---|
| Sent. neto | Z(SEN) | **+0,58 [+0,42; +0,72]** | +0,51 [+0,30; +0,69] | +0,62 [+0,42; +0,77] | +0,49 [+0,27; +0,67] |
| Sent. neto | Z(SUS) | −0,06 [−0,28; +0,14] | −0,29 [−0,43; −0,16] | +0,03 [−0,35; +0,42] | +0,02 [−0,10; +0,17] |
| Sent. neto | Z(HEDGE) | −0,35 [−0,53; −0,15] | −0,19 [−0,38; +0,02] | −0,30 [−0,62; +0,02] | −0,20 [−0,36; −0,02] |
| Sent. neto | ESGSI | +0,48 [+0,30; +0,64] | +0,58 [+0,42; +0,70] | +0,50 [+0,23; +0,71] | +0,40 [+0,17; +0,58] |
| Cuota específica | Z(SUS) | **−0,11 [−0,29; +0,06]** | −0,35 [−0,49; −0,20] | +0,04 [−0,23; +0,28] | −0,24 [−0,42; −0,03] |
| Cuota específica | Z(QUANT) | **+0,20 [+0,01; +0,39]** | +0,13 [+0,02; +0,29] | +0,26 [−0,04; +0,47] | +0,14 [+0,02; +0,33] |
| Cuota específica | Z(SEN) | +0,15 [−0,06; +0,37] | +0,11 [−0,13; +0,37] | +0,10 [−0,18; +0,39] | +0,04 [−0,22; +0,31] |
| Cuota climática | Z(SUS) | **+0,65 [+0,52; +0,76]** | +0,77 [+0,70; +0,85] | +0,57 [+0,33; +0,73] | +0,59 [+0,46; +0,71] |
| Cuota climática | Z(QUANT) | **+0,39 [+0,21; +0,56]** | +0,19 [+0,05; +0,41] | +0,60 [+0,26; +0,80] | +0,19 [+0,01; +0,31] |
| CTI | Z(HEDGE) | **+0,15 [−0,08; +0,35]** | +0,03 [−0,15; +0,20] | +0,11 [−0,19; +0,38] | +0,01 [−0,18; +0,20] |
| CTI | Z(QUANT) | −0,16 [−0,36; +0,06] | −0,15 [−0,27; −0,04] | −0,25 [−0,50; +0,15] | −0,14 [−0,28; −0,02] |
| CTI | Z(SUS) | +0,06 [−0,12; +0,25] | +0,20 [+0,00; +0,37] | −0,01 [−0,30; +0,32] | +0,17 [−0,06; +0,39] |
| CTI | **ESGSI** | **−0,06 [−0,27; +0,14]** | −0,13 [−0,26; −0,01] | +0,00 [−0,34; +0,32] | −0,06 [−0,21; +0,09] |
| CTI | **ESGSI_ext** | **+0,02 [−0,20; +0,23]** | −0,06 [−0,21; +0,08] | +0,12 [−0,27; +0,45] | +0,01 [−0,16; +0,16] |
| Cuota compromiso | ESGSI | +0,44 [+0,24; +0,60] | +0,62 [+0,48; +0,73] | +0,35 [−0,02; +0,65] | +0,53 [+0,32; +0,67] |
| Cuota climática | ESGSI | −0,46 [−0,62; −0,30] | −0,52 [−0,67; −0,33] | −0,44 [−0,66; −0,12] | −0,13 [−0,39; +0,14] |

**Matriz agregada** (Spearman, 342 informes; `mtmm_*.csv` tiene también las vistas intra, entre y
doble EF):

|  | Z(SEN) | Z(SUS) | Z(QUANT) | Z(HEDGE) | ESGSI | ESGSI_ext | Sent. | Espec. | Clima | CTI |
|---|---|---|---|---|---|---|---|---|---|---|
| Z(SEN) | 1 | 0,17 | 0,21 | −0,49 | 0,61 | 0,27 | **0,58** | 0,15 | 0,03 | −0,03 |
| Z(SUS) | | 1 | 0,46 | −0,29 | −0,62 | −0,79 | −0,06 | **−0,11** | **0,65** | 0,06 |
| Z(QUANT) | | | 1 | −0,30 | −0,20 | −0,53 | 0,18 | **0,20** | **0,39** | −0,16 |
| Z(HEDGE) | | | | 1 | −0,19 | 0,27 | −0,35 | −0,19 | −0,31 | **0,15** |
| ESGSI | | | | | 1 | 0,83 | 0,48 | 0,18 | −0,46 | **−0,06** |
| ESGSI_ext | | | | | | 1 | 0,27 | 0,07 | −0,65 | **0,02** |
| Sent. neto | | | | | | | 1 | 0,51 | −0,02 | −0,37 |
| Específica | | | | | | | | 1 | 0,26 | −0,88 |
| Climática | | | | | | | | | 1 | −0,26 |
| CTI | | | | | | | | | | 1 |

El criterio de Campbell y Fiske que se aplica aquí es este: el valor de validez, con el signo
esperado, debe superar el máximo |r| heterorrasgo-heterométodo de su fila y su columna. Para los
pares de componentes no se cuentan ESGSI ni ESGSI_ext como "otro rasgo", porque contienen a los
componentes. El piloto sí los contaba, y por eso allí el tono fallaba en la vista intra-empresa.

| Par de validez | Agregado ρ | Intra r | Entre r | Doble EF r |
|---|---|---|---|---|
| Z(SEN) ~ sent. neto | 0,58 > 0,35 ✔ | 0,51 > 0,29 ✔ | 0,62 > 0,30 ✔ | 0,49 > 0,40 ✔ |
| Z(SUS) ~ específica | −0,11 ✘ | −0,35 ✘ | 0,04 ✘ | −0,24 ✘ |
| Z(SUS) ~ climática | 0,65 > 0,31 ✔ | 0,77 > 0,29 ✔ | 0,57 > 0,28 ✔ | 0,59 > 0,43 ✔ |
| Z(QUANT) ~ específica | 0,20 > 0,19 (≈) | 0,13 < 0,15 ✘ | 0,26 > 0,25 (≈) | 0,14 ≈ 0,14 (≈) |
| Z(QUANT) ~ climática | 0,39 > 0,31 ✔ | 0,19 < 0,29 ✘ | 0,60 > 0,28 ✔ | 0,19 < 0,43 ✘ |
| Z(HEDGE) ~ CTI | 0,15 < 0,35 ✘ | 0,03 ✘ | 0,11 ✘ | 0,01 ✘ |
| ESGSI ~ CTI | −0,06 ✘ | −0,13 ✘ | 0,00 ✘ | −0,06 ✘ |
| ESGSI_ext ~ CTI | 0,02 ✘ | −0,06 ✘ | 0,12 ✘ | 0,01 ✘ |

**Mismo alcance** (los componentes del diccionario medidos solo sobre los pasajes climáticos;
Spearman agregado, IC por bootstrap):

| | Sent. neto | Cuota específica | CTI |
|---|---|---|---|
| SEN_clim | **+0,66 [+0,53; +0,77]** | +0,12 [−0,08; +0,32] | −0,01 [−0,21; +0,17] |
| SEN_nonclim | +0,42 [+0,21; +0,59] | +0,08 [−0,14; +0,30] | +0,02 [−0,22; +0,24] |
| SUS_clim | −0,21 [−0,41; −0,03] | **−0,31 [−0,47; −0,14]** | +0,27 [+0,10; +0,43] |
| SUS_nonclim | +0,02 [−0,21; +0,24] | −0,08 [−0,27; +0,09] | +0,02 [−0,16; +0,22] |
| QUANT_clim | +0,22 [−0,02; +0,42] | **+0,32 [+0,11; +0,50]** | −0,26 [−0,47; −0,04] |
| HEDGE_clim | −0,42 [−0,56; −0,28] | −0,12 [−0,30; +0,07] | **+0,03 [−0,16; +0,22]** |

Lectura:

- **Tono: converge y discrimina.** SEN va con el sentimiento de ClimateBERT (0,49-0,62 en todas las
  vistas, 0,66 con el mismo alcance) y no con la especificidad ni con el CTI. Que SEN_nonclim también
  converja (0,42) indica que el tono de un informe es bastante homogéneo entre sus partes climática
  y no climática.
- **Sustancia por densidad: mide cuánto texto ESG o climático hay, no cuán concreto es.** Z(SUS) va
  con la cuota climática (0,65; 0,77 intra-empresa), con la especificidad en contra dentro de cada
  empresa (−0,35), y con más cheap talk dentro de los pasajes climáticos (SUS_clim ~ CTI = +0,27).
  Esto es coherente con la lectura del paper de que la subida del SUS es una divulgación más densa,
  no más concreta. El emparejamiento "sustancia = especificidad" falla; el de "sustancia = cantidad
  de divulgación temática" funciona.
- **QUANT converge con la especificidad**, débilmente por informe (0,20; 0,32 con el mismo alcance) y
  con claridad por pasaje (0,45). Va en contra del CTI (−0,16; −0,26 con el mismo alcance).
- **HEDGE no converge con el CTI.** En el piloto, ρ = 0,30 con el HEDGE anterior. Al quitar
  *risk/risks* y "May", el vínculo se queda en 0,15 [−0,08; 0,35] y en 0,03 con el mismo alcance.
  Probablemente era en buena parte vocabulario de riesgo y no matización (dato interno, no para el
  paper: las directrices excluyen comparar con especificaciones anteriores). HEDGE se relaciona con
  un tono menos favorable (−0,35; −0,42 con el mismo alcance) y, por pasaje, con menos compromisos
  (§6), pero no con compromisos más vagos.
- **El compuesto no converge.** El ESGSI se parece a "tono favorable y muchos compromisos" (+0,48
  con el sentimiento y +0,44 con la cuota de compromiso) en informes con poca cuota climática
  (−0,46). No se parece a "compromisos vagos" (CTI −0,06).
- **Deciles extremos** (34 de 342; esperado por azar 3,4; nulo hipergeométrico). ESGSI frente al CTI:
  1 informe compartido arriba (p = 0,98) y 8 abajo (p = 0,011). ESGSI_ext frente al CTI: 5 arriba
  (p = 0,24) y 9 abajo (p = 0,003). Hay algo de acuerdo abajo (informes de tono bajo y mucha
  sustancia con compromisos específicos) y nada arriba.

## 4. ¿Ve ClimateBERT la ruptura de 2024?

Cada medida se expresa en DT agregadas. Se estima (a) el salto de 2024 frente a 2018-2023 con EF de
empresa, y_it = a_i + b·1[2024], y (b) el salto sobre una tendencia lineal común. Los errores se
agrupan por empresa (t₄₈) y p_w sale de un wild cluster bootstrap (Rademacher, 9.999 réplicas; en
(b) se aplica por Frisch-Waugh-Lovell). La última columna es la pendiente 2018-2023 en DT/año.
Medias anuales con IC agrupados en `yearly_ci_z.csv` y en unidades naturales en
`yearly_means_raw.csv`.

| Medida | 2024 − (2018-23) | p | p_w | Sobre tendencia | p | Pendiente 2018-23 (p) |
|---|---|---|---|---|---|---|
| **Z(SEN)** | **−0,30 [−0,50; −0,09]** | 0,005 | 0,006 | −0,28 | 0,006 | −0,005 (0,87) |
| Z(SEN) sin lista A (24 palabras ESRS) | **−0,02 [−0,20; +0,17]** | 0,87 | 0,87 | −0,03 | 0,79 | +0,003 (0,92) |
| Z(SEN) sin lista B (48 palabras) | +0,04 [−0,14; +0,23] | 0,63 | 0,64 | +0,03 | 0,74 | +0,004 (0,90) |
| Z(SEN) solo pasajes climáticos | −0,23 [−0,41; −0,06] | 0,011 | 0,010 | −0,06 | 0,49 | −0,047 (0,08) |
| Z(SEN) pasajes no climáticos | −0,36 [−0,56; −0,15] | 0,001 | 0,002 | −0,23 | 0,017 | −0,035 (0,22) |
| **Sentimiento neto (CB)** | **−0,69 [−0,88; −0,50]** | 2·10⁻⁹ | < 10⁻⁴ | −0,41 | < 10⁻⁴ | −0,080 (0,004) |
| Cuota de oportunidad (CB) | −0,81 [−0,98; −0,64] | 9·10⁻¹³ | < 10⁻⁴ | −0,53 | 2·10⁻⁷ | −0,081 (0,011) |
| Cuota de riesgo (CB) | +0,21 [+0,01; +0,42] | 0,043 | 0,043 | +0,04 | 0,70 | +0,047 (0,07) |
| Sent. neto sin pasajes con marco ESRS | −0,60 [−0,79; −0,42] | 4·10⁻⁸ | < 10⁻⁴ | −0,33 | < 10⁻³ | −0,077 (0,006) |
| Sent. neto sin marco ESRS ni riesgo físico/de transición | −0,54 [−0,73; −0,35] | 5·10⁻⁷ | < 10⁻⁴ | −0,32 | 0,001 | −0,061 (0,03) |
| **CTI** | **+0,36 [+0,16; +0,56]** | < 10⁻³ | < 10⁻³ | +0,49 | < 10⁻⁴ | −0,035 (0,31) |
| Cuota específica | −0,69 [−0,89; −0,49] | 8·10⁻⁹ | < 10⁻⁴ | −0,76 | < 10⁻⁴ | +0,019 (0,54) |
| Cuota de compromiso | −0,94 [−1,18; −0,71] | 2·10⁻¹⁰ | < 10⁻⁴ | −0,89 | < 10⁻⁴ | −0,016 (0,65) |
| Cuota climática | +0,81 [+0,64; +0,98] | 2·10⁻¹² | < 10⁻⁴ | +0,15 | 0,048 | +0,189 (10⁻¹¹) |
| Z(SUS) (referencia) | +1,00 [+0,81; +1,18] | 9·10⁻¹⁵ | < 10⁻⁴ | +0,45 | < 10⁻⁴ | +0,162 (10⁻⁷) |

Medias anuales en DT agregadas (IC agrupados por empresa en `yearly_ci_z.csv`):

| | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 |
|---|---|---|---|---|---|---|---|
| Z(SEN) | +0,10 | +0,04 | −0,01 | +0,07 | −0,01 | +0,08 | −0,26 [−0,51; −0,00] |
| Z(SEN) sin lista A | +0,04 | −0,01 | −0,05 | +0,01 | −0,05 | +0,07 | −0,01 [−0,26; +0,24] |
| Sentimiento neto CB | +0,22 | +0,31 | +0,10 | +0,11 | −0,04 | −0,11 | −0,59 [−0,81; −0,37] |
| Cuota de oportunidad CB | +0,19 | +0,36 | +0,15 | +0,13 | −0,03 | −0,11 | −0,69 [−0,92; −0,45] |
| Cuota de riesgo CB | −0,19 | −0,10 | +0,03 | −0,03 | +0,04 | +0,06 | +0,18 [−0,02; +0,38] |
| CTI | +0,04 | −0,14 | +0,16 | −0,07 | −0,11 | −0,19 | +0,31 [+0,11; +0,50] |
| Cuota específica | +0,02 | +0,17 | −0,04 | +0,15 | +0,13 | +0,16 | −0,59 [−0,79; −0,40] |
| Cuota de compromiso | +0,00 | +0,21 | +0,24 | +0,25 | +0,12 | −0,02 | −0,80 [−1,01; −0,60] |
| Cuota climática | −0,49 | −0,47 | −0,28 | −0,01 | +0,18 | +0,38 | +0,70 [+0,48; +0,91] |

**Las listas de palabras.** La lista A se define por el texto de las normas y no por los datos. Son
las palabras de la lista Negative de L&M que **nombran un tema o un dato que las ESRS obligan a
divulgar**, no una valoración del desempeño:

- *negative(ly), adverse(ly)*: "negative/adverse impacts", el eje de la doble materialidad en
  ESRS 1 §3.4, ESRS 2 SBM-3 e IRO-1;
- *incident(s), complaint(s), grievance(s), concern(s)*: incidentes y quejas en S1-17; cauces para
  plantear inquietudes en S1-3, S2-3 y G1-1;
- *harassment, harassed*: S1-17;
- *forced*: trabajo forzoso, en S1-1 y S2;
- *violation(s), violate(d)*: violaciones de derechos humanos, en S1-17;
- *bribery, bribe(s), corruption, corrupt*: G1-3 y G1-4.

La lista B añade salud y seguridad (S1-14: *accident(s), injury/ies, fatality/ies, fatal, severe,
severity*) y sanciones e infracciones (G1-4: *convicted, conviction(s), fines, penalty/ies,
breach(es), infringement(s), misconduct, fraud, abuse(s), exploitation*). Estas palabras ya se
divulgaban antes, con GRI.

Honestidad sobre la lista A: aunque su criterio es normativo, coincide en gran parte con las palabras
que el paper ya identificó como las que más crecen. Las 10 negativas que más crecen de 2023 a 2024
(`sen_negative_words_2023_2024.csv`) están todas en la lista A. No es, por tanto, independiente de los
datos. La lista A explica el 78 % del aumento de la tasa de palabras negativas entre 2023 y 2024, y la
lista B el 91 %. Su peso entre las negativas pasa del 11-13 % en 2018-2023 al 19,5 % en 2024. Quitar
las 10 o las 25 negativas que más crecen, elegidas con los datos y por eso una cota, también elimina
el salto: −0,07 (p = 0,49) y +0,08 (p = 0,41). Fuera de esas palabras, la lista Negative no se mueve
en 2024 y la Positive apenas (104-106 por 10.000 palabras en 2019-2023 y 103,4 en 2024; las que más
bajan son *progress, best, innovation, strong, leadership*).

**El marco ESRS en el texto climático.** Son pasajes que citan las ESRS, los IRO, la doble
materialidad o "impacts, risks and opportunities". Son el 0 % de los pasajes climáticos en 2018-2020
y el 8,9 % en 2024; con riesgo físico o de transición, el 0,3 % y el 11,7 %. Quitarlos solo reduce
la caída de ClimateBERT de −0,69 a −0,60 (−0,54 en la variante amplia).

**Amplitud por empresa** (`summary.json`, `ruptura_2024_por_empresa`): en 2024 el sentimiento neto
de ClimateBERT cae respecto a la media 2018-2023 de la propia empresa en 42 de 49 empresas; la cuota
de oportunidad, en 44; la de compromiso, en 43. SEN cae en 34 y SEN sin la lista A en 27 (≈ la
mitad). La caída de ClimateBERT es mayor donde más texto climático de 2024 usa el marco ESRS: la ρ
del cambio con la cuota de marco es −0,24 para el sentimiento, −0,28 para la oportunidad y −0,37 para
la cuota de compromiso.

Lectura:

1. **La caída de SEN en 2024 es vocabulario de temas ESRS.** Al quitar esas 24 palabras desaparece
   por completo (de −0,30 a −0,02 DT) y el SEN restante no tiene tendencia ni salto en 2018-2024. El
   paper puede afirmarlo con un contraste explícito, no solo con la lista de palabras que más crecen.
   La caída también existe en los pasajes no climáticos (−0,36), que es donde están S1-S4 y G1.
2. **ClimateBERT también ve un cambio en 2024, pero en el texto climático y de otra naturaleza.**
   No aumenta el riesgo (+0,04 sobre la tendencia, p = 0,70). Bajan los pasajes de oportunidad, que
   pasan a neutros (cuota de oportunidad 0,30 → 0,22), y bajan los compromisos (0,50 → 0,41) y la
   especificidad (0,37 → 0,29), de modo que el CTI sube (0,47 → 0,53). Es lo que cabe esperar de una
   declaración climática prescrita (E1: políticas, acciones, métricas, contabilidad de GEI) frente a
   un relato voluntario de oportunidades y objetivos. La caída está en casi todas las empresas, es
   mayor donde hay más texto con el marco ESRS y persiste sin esos pasajes, así que no es un artefacto
   de unas pocas frases tipo.
3. **Diferencia antes de 2024.** El sentimiento de ClimateBERT ya bajaba en 2018-2023 (−0,08 DT/año,
   p = 0,004); SEN no (−0,005, p = 0,87). Son constructos distintos: ClimateBERT mide el encuadre
   oportunidad/riesgo de la cuestión climática, y L&M, el tono financiero general de todo el texto
   ESG.
4. **Consecuencia para el paper.** Las dos herramientas coinciden en que 2024 es otro tipo de texto,
   más prescrito, más neutro y con menos compromisos. Ninguna indica que las empresas describan peor
   su desempeño. La conclusión del paper ("not because firms came to describe their ESG performance
   less favourably") queda reforzada para SEN. Para ClimateBERT hay que matizarla: menos encuadre de
   oportunidad, que no es más negatividad.

Nota: Bax, Paterlini y Valentini (2026, *Economics Letters*), según un resultado de búsqueda,
aplican el CTI a los informes de sostenibilidad del EURO STOXX 50 de 2020-2024 y encuentran una
subida tras la CSRD. Es coherente con el salto del CTI en 2024. **No pude acceder al texto ni
verificar la referencia** (403 en el repositorio de la Universidad de Trento); habría que
comprobarla antes de citarla.

## 5. Convergencia a nivel de pasaje

Sobre los 118.258 pasajes climáticos se calcula la ρ de Spearman dentro de cada informe (con 10 o más
pasajes) y se da la mediana entre informes. La polaridad L&M de cada pasaje es la de
`ESGSIAnalyzer.sen_counts`, (P − N)/(P + N), y vale 0 si el pasaje no tiene palabras de tono.

| Par | Informes | ρ mediana | RIC | Cuota > 0 | ρ con todos los pasajes juntos |
|---|---|---|---|---|---|
| **Polaridad L&M ~ p(oport.) − p(riesgo)** | 340 | **0,42** | 0,37-0,48 | 99,7 % | 0,43 |
| ídem, solo pasajes con tono L&M (64 %) | 339 | 0,47 | 0,39-0,52 | 99,7 % | |
| QUANT ~ p(específico) | 330 | 0,45 | 0,38-0,50 | 99,4 % | 0,47 |
| *Heterorrasgo:* polaridad L&M ~ p(específico) | 340 | 0,08 | 0,01-0,15 | 77 % | 0,08 |
| *Heterorrasgo:* QUANT ~ sentimiento CB | 330 | 0,16 | 0,09-0,24 | 95 % | 0,19 |
| HEDGE ~ p(específico) | 340 | −0,06 | −0,14-0,00 | 26 % | −0,08 |
| HEDGE ~ sentimiento CB | 340 | −0,22 | −0,29 a −0,14 | 4 % | −0,23 |
| HEDGE ~ p(compromiso) | 340 | −0,16 | −0,23 a −0,10 | 7 % | −0,17 |

Por año, la ρ mediana entre la polaridad L&M y ClimateBERT está entre 0,40 y 0,44 todos los años
(0,41 en 2024): la relación por pasaje no cambia con las ESRS. Los pasajes matizados son menos
optimistas y menos a menudo compromisos, pero apenas menos específicos.

## 6. Tendencias de los índices de ClimateBERT (EF de empresa)

Se usa `trend_block` de `scripts/analysis_v3.py`: EF de empresa, SE agrupados por empresa (t₄₈) y
wild cluster bootstrap (9.999 réplicas). Las pendientes van en unidades naturales por año; la última
columna es la pendiente con EF solo en 2018-2023.

| Medida | Pendiente/año | IC 95 % | p | p_w | R²_w | 2018-2023 (p) |
|---|---|---|---|---|---|---|
| CTI | +0,0020 | [−0,0053; +0,0093] | 0,59 | 0,59 | 0,002 | −0,0044 (0,31) |
| CTI_prob | +0,0010 | [−0,0033; +0,0053] | 0,63 | 0,62 | 0,002 | −0,0035 (0,17) |
| Cuota específica | −0,0065 | [−0,0121; −0,0009] | 0,023 | 0,022 | 0,039 | +0,0020 (0,54) |
| Sentimiento neto | −0,0221 | [−0,0307; −0,0136] | 4·10⁻⁶ | < 10⁻⁴ | 0,182 | −0,0143 (0,004) |
| Cuota de compromiso | −0,0125 | [−0,0196; −0,0053] | 0,001 | 0,001 | 0,097 | −0,0018 (0,65) |
| Cuota climática | +0,0280 | [+0,0225; +0,0335] | 10⁻¹³ | < 10⁻⁴ | 0,511 | +0,0257 (10⁻¹¹) |

El CTI no tiene tendencia: oscila sin orden entre 0,47 y 0,51 hasta 2023 y salta a 0,53 en 2024. La
cuota climática sube de forma sostenida, de 0,28 a 0,44, igual que el SUS. Es la misma divulgación
más densa vista con otro instrumento. El sentimiento baja de forma sostenida. Especificidad y
compromiso solo bajan por 2024. Bingler et al. (2024) documentan una subida del CTI en 2010-2020 en el
MSCI World; aquí la ventana y la muestra son otras.

## 7. Evaluación honesta

**Lo que valida parcialmente la propuesta:**

1. **El componente de tono**, que es la mitad del ESGSI. Converge con un instrumento independiente en
   las cuatro vistas, discrimina y también converge por pasaje. Con SEN por forma exacta la
   convergencia es algo mayor que en el piloto.
2. **QUANT como cuantificación.** Va con la especificidad por pasaje (0,45) y, débilmente, por
   informe. Su signo en el ESGSI_ext (resta) es coherente con que va en contra del CTI.
3. **La lectura de la subida del SUS** como más divulgación temática, no más concreta: converge con
   la cuota climática (0,65) y va en contra de la especificidad dentro de cada empresa.
4. **La lectura de la ruptura de 2024** como vocabulario prescrito y no como peor desempeño. La
   caída de SEN es exactamente el vocabulario de temas ESRS, y ClimateBERT ve en 2024 un texto más
   neutro y con menos compromisos, no más negativo.

**Lo que no valida:**

1. **El compuesto no mide cheap talk.** ESGSI y ESGSI_ext no tienen relación con el CTI (−0,06 y
   0,02) en ninguna vista. El paper no debe sugerir que el ESGSI sea un proxy de compromisos vagos ni
   de greenwashing en el sentido de Bingler et al.
2. **HEDGE**, en su forma corregida, no converge con el CTI. El argumento a favor de HEDGE en el
   ESGSI_ext que daba el piloto ya no se sostiene. HEDGE es más bien "tono reservado" (va con menos
   optimismo y menos compromisos) que "vaguedad de los compromisos".
3. **"Sustancia" como concreción.** El SUS no es especificidad. Si el paper llama *substance* al SUS,
   conviene que diga explícitamente que es densidad de vocabulario, no concreción. Ya lo dice en
   parte; esto lo refuerza.
4. **Validez externa.** Es el mismo texto, así que hay varianza de método compartida: la convergencia
   del tono es una cota superior, no una validación externa. La validez del CTI (emisiones y
   noticias negativas en el MSCI World) no se transfiere a estos informes.

## 8. Recomendación

Incluir un párrafo en §5 (borrador en inglés en `draft_robustness_climatebert.tex`, ≈ 230 palabras,
sin tabla). Dice tres cosas:

- el tono valida;
- el compuesto no es un proxy de cheap talk;
- la caída de SEN en 2024 desaparece al quitar el vocabulario de temas ESRS, mientras ClimateBERT ve
  en 2024 menos encuadre de oportunidad y no más riesgo.

Además:

- En §4 (resultados, ruptura de 2024) conviene añadir el contraste de la lista A: "removing 24 words
  of the negative list that name ESRS disclosure topics removes the 2024 fall (−0.02 SD, p = 0.87)",
  con la lista en una nota. Así la afirmación del paper ("that fall comes from the negative list")
  pasa de descriptiva a contrastada.
- Quitar del paper cualquier frase que apoye HEDGE en su relación con compromisos vagos, si la hay.
- Referencias necesarias: `bingler2022`, `bingler2024`, `webersinke2021`, verificadas en
  `results/issue2_lit/refs.bib` de la rama `issue-2-lit-review`. Hay que añadirlas a
  `paper/references.bib`. `campbell1959` ya está.

## 9. Límites

1. **Alcance.** El CTI y sus componentes cubren solo el texto climático (37 % de los pasajes) y
   nuestros scores, E, S y G. Las versiones con el mismo alcance muestran que el patrón no se debe a
   eso.
2. **Réplica no verificada al detalle.** No pude leer la segmentación ni los filtros de Bingler et al.
   Aquí los pasajes salen de las zonas ESG extraídas, no del informe completo: un párrafo climático
   sin vocabulario ESG nunca se clasifica.
3. **Clasificadores.** Están entrenados con ≈ 1.300 párrafos y tienen error propio. El cambio de
   tipo de texto en 2024 (declaraciones ESRS) puede afectar a su calibración, sobre todo a la de
   especificidad.
4. **La lista A no es independiente de los datos** (§4). La lista B y las cotas Top-10 y Top-25 dan
   el mismo resultado.
5. **Panel.** Un informe sin pasajes climáticos queda fuera; el panel tiene 342 informes y el doble
   EF se calcula por proyecciones alternadas.

## Reproducción

```
# entorno aislado del piloto (torch CPU + transformers): .venv-cb
.venv-cb/Scripts/python.exe scripts/cb_segment.py                       # 1 min
.venv-cb/Scripts/python.exe scripts/cb_classify.py --seed-from <caché del piloto>   # 3,6 min con caché
.venv-cb/Scripts/python.exe scripts/cb_analysis.py --export-scope       # texto por alcance
<python del pipeline, con spaCy en_core_web_md> scripts/cb_sus_scope.py  # 10 min (6 procesos)
.venv-cb/Scripts/python.exe scripts/cb_analysis.py                      # 2 min
```

Los scripts leen `data/chunks_lexical_v3_sec` y `results/analysis_v3/doc_level_internal.csv` del clon
principal, solo en lectura. La caché (`data/climatebert/`) no se versiona. Versiones: torch
2.14.1+cpu, transformers 5.19.0, spaCy 3.8.11; las revisiones de los modelos están en
`summary.json` (`runtime_inferencia`).

## Ficheros

- `doc_level_cb.csv`: medidas por informe (identificador anónimo, año, país, industria, scores del
  paper, medidas de ClimateBERT, componentes por alcance, variantes de SEN).
- `correlations.csv`: todas las correlaciones con IC, en los bloques `paper_scores`,
  `climate_scope_dictionary` y `sen_variants`.
- `mtmm_*.csv`: matrices multirrasgo-multimétodo.
- `yearly_ci_z.csv`, `yearly_means_raw.csv`: medias anuales.
- `sen_negative_words_2023_2024.csv`: tasas por palabra de la lista Negative de L&M y su cambio.
- `summary.json`: cobertura, tiempos, descriptivos, criterios de Campbell-Fiske, ruptura de 2024,
  pendientes, deciles y nivel de pasaje.
- `segmentation.json`: segmentación e idioma.
- `cache_check.json`: re-inferencia de 300 textos por modelo.
- `draft_robustness_climatebert.tex`: borrador para §5, no insertado en el paper.
