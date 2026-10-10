# Validez convergente con otros modelos ESG abiertos (issue #5)

Continuación del #4 (ClimateBERT, solo pasajes climáticos). Aquí se aplican al
mismo texto extraído modelos abiertos que cubren el tono de todo el texto y los
pilares S y G, y se mide además el *recall* de la extracción. Solo scores
continuos; ninguna salida lleva nombres de empresa (solo `firm_label`).

Scripts (rama `esg-models`): `scripts/esgm_common.py`, `esgm_sample.py`,
`esgm_classify.py`, `esgm_recall_extract.py`, `esgm_analysis.py`.
Salidas: `doc_level_esgm.csv` (343 informes), `correlations.csv` (todas las
correlaciones con IC), `summary.json` (cobertura, tiempos, saltos de 2024,
pendientes, nivel de pasaje, recall), `yearly_means.csv`, `yearly_ci_z.csv`,
`recall_by_year.csv`, `sample.json`.

## 1. Modelos

| Clave | Repositorio HF | Revisión fijada | Licencia | Etiquetas | Unidad usada |
|---|---|---|---|---|---|
| tone | `yiyanghkust/finbert-tone` | `4921590d3c0c` | sin licencia en la ficha HF; el repositorio del código (yya518/FinBERT) es Apache-2.0; la ficha pide citar Huang et al. (2023) en trabajos académicos | Neutral / Positive / Negative | **frase** (se entrenó con 10.000 frases de informes de analistas) |
| esg4 | `yiyanghkust/finbert-esg` | `f79fefa034aa` | ídem | None / E / S / G | pasaje |
| esg9 | `yiyanghkust/finbert-esg-9-categories` | `af56509508a6` | ídem | 9 temas | pasaje |
| env | `ESGBERT/EnvironmentalBERT-environmental` | `804e3f23cf99` | Apache-2.0 | none / environmental | pasaje |
| soc | `ESGBERT/SocialBERT-social` | `e0bef3b8ea33` | Apache-2.0 | none / social | pasaje |
| gov | `ESGBERT/GovernanceBERT-governance` | `9e9f825dad0b` | Apache-2.0 | none / governance | pasaje |
| action | `ESGBERT/EnvironmentalBERT-action` | `43c5ef138293` | Apache-2.0 | none / action | pasajes ambientales (env > 0,5), como pide la ficha |
| netzero | `climatebert/netzero-reduction` | `25cf57e30613` | Apache-2.0 | none / reduction / net-zero | **todos** los pasajes climáticos del #4 (detector > 0,5), como pide la ficha |

Revisiones completas en `summary.json` (`modelos`). Ningún modelo se descartó.
Los FinBERT no declaran licencia en Hugging Face: se usan para investigación,
citando el artículo como pide la ficha; conviene mencionarlo si se citan en el
paper. Los tres repositorios FinBERT no traen `model_type` ni
`tokenizer_config`; se cargan con `BertForSequenceClassification` y
`BertTokenizerFast` en minúsculas, como en la ficha. Todas las etiquetas se
comprobaron con los ejemplos de las fichas (`data/esg_models/runtime.json`,
`selftest`: todas las clases esperadas con p ≥ 0,95).

Clase de cada texto: argmax de las probabilidades (pipeline estándar); en los
detectores binarios, p > 0,5. Las clases se usan solo para formar cuotas por
informe, que son scores continuos.

## 2. Cobertura, tiempo y muestreo

- Unidad: los 318.258 pasajes (párrafos de ≥ 30 palabras) del #4. CPU i7-9700K,
  8 hilos. Rendimiento medido: BERT-base (FinBERT) ≈ 27 pasajes/s;
  DistilRoBERTa (ESGBERT, NetZero) ≈ 54/s. Un pase completo costaría ≈ 3,3 h por
  FinBERT y ≈ 1,6 h por ESGBERT, unas 15 h en total.
- **Muestra común estratificada por informe**: 200 pasajes por informe sin
  reemplazo (todos en los 17 informes que tienen menos), semilla 20261011:
  **67.448 pasajes** (21,2 %), 202.301 frases para FinBERT-tone. Dentro de
  cada informe la muestra es autoponderada.
- **Error de muestreo** (linealizado, con corrección por población finita;
  `summary.json`, `error_de_muestreo_por_informe`): tono neto FinBERT, error
  típico mediano 0,019 (fiabilidad entre informes 0,88); cuota G de FinBERT-ESG
  0,032 (0,96); cuota G de ESGBERT 0,022 (0,88); cuota de acción 0,050 (0,84).
  La atenuación de las correlaciones por muestreo es, por tanto, pequeña.
- NetZero no se muestreó: 118.258 pasajes climáticos (los objetivos son raros,
  5–10 % de los pasajes climáticos, y una muestra daría cuotas muy ruidosas).
- Tiempo de inferencia: tone 44 min, esg4 40 min, esg9 39 min, env/soc/gov 20 min
  cada uno, action 7 min, netzero 34 min; recall 17 min. ≈ 4,7 h en total.
  Predicciones en caché por hash SHA-1 del texto (`data/esg_models/`, sin versionar).

## 3. Experimento 1: tono en todo el texto

Tono neto FinBERT por informe = (frases positivas − negativas) / frases, sobre
los pasajes muestreados. IC 95 % por bootstrap de empresas (B = 2000).

| Par | Agregado ρ | Intra r | Entre r |
|---|---|---|---|
| **Z(SEN) ~ tono FinBERT** | **0,77 [0,68; 0,84]** | **0,73 [0,60; 0,81]** | **0,86 [0,77; 0,91]** |
| SEN en los mismos pasajes ~ tono FinBERT | 0,80 [0,71; 0,86] | 0,73 | 0,85 |
| tono FinBERT ~ sentimiento neto ClimateBERT (#4) | 0,56 [0,42; 0,68] | 0,53 | 0,62 |
| Z(HEDGE) ~ tono FinBERT | −0,44 [−0,59; −0,27] | −0,35 | −0,59 |
| Z(SUS) ~ tono FinBERT | 0,14 [−0,06; 0,31] | 0,04 | 0,20 |
| Z(QUANT) ~ tono FinBERT | 0,18 [−0,04; 0,39] | 0,11 | 0,10 |

- **Converge más que con ClimateBERT** (0,77 frente a 0,58) y cubre todo el
  texto. Cumple Campbell–Fiske: el valor de validez (0,77) supera con holgura
  las correlaciones del tono FinBERT con los demás componentes (máx. |−0,44|,
  HEDGE, que se comporta como tono reservado, como ya se vio en el #4).
- **Por pasaje** (ρ de Spearman dentro de cada informe entre la polaridad L&M
  del pasaje y el tono FinBERT de sus frases): mediana **0,46** (RIC 0,41–0,52),
  positiva en los 343 informes; 0,46 en pasajes no climáticos; estable por año
  (0,44–0,48). Controles con otro rasgo: polaridad ~ QUANT 0,04; polaridad ~
  probabilidad de gobernanza 0,00.
- **Por pilar** (pilar del pasaje según FinBERT-ESG; entre paréntesis, según
  nuestro vocabulario sobre texto crudo):

  | Pilar | Informe: agregado ρ | Intra r | Pasaje: ρ mediana |
  |---|---|---|---|
  | E | 0,66 [0,55; 0,74] (0,70) | 0,51 | 0,45 (0,47) |
  | S | 0,69 [0,59; 0,77] (0,47) | 0,57 | 0,46 (0,48) |
  | G | 0,41 [0,28; 0,53] (0,53) | 0,43 | 0,35 (0,38) |
  | sin tema ESG | 0,68 [0,59; 0,75] | 0,58 | 0,44 |

  El tono converge en los tres pilares, no solo en el clima; en G algo menos
  (el texto de gobernanza tiene poco tono: tono FinBERT medio ≈ 0,05 frente a
  0,13–0,22 en E y S, y SEN negativo).

### ¿Ve FinBERT la ruptura de 2024?

Salto 2024 frente a 2018–2023 con EF de empresa, en DT agregadas, error
agrupado por empresa (p del wild cluster bootstrap entre paréntesis):

| Medida | Salto 2024 | p |
|---|---|---|
| Z(SEN) | −0,30 [−0,50; −0,09] | 0,005 |
| Tono neto FinBERT | **−0,27 [−0,49; −0,06]** | 0,014 |
| … sin los pasajes dominados por la lista A (≥ 50 % de sus negativas L&M) | **−0,26 [−0,47; −0,05]** | 0,017 |
| … sin ningún pasaje con una palabra de la lista A | −0,30 [−0,51; −0,10] | 0,005 |
| Cuota de frases positivas FinBERT | −0,33 [−0,54; −0,12] | 0,002 |
| Cuota de frases negativas FinBERT | −0,03 [−0,23; 0,18] | 0,79 |
| Tono FinBERT en pasajes E / S / G (FinBERT-ESG) | −0,46 / −0,54 / +0,17 (n.s.) | |
| SEN, pasajes ≥ 30 palabras (población) | −0,20 [−0,41; 0,01] | 0,06 |
| … sin las palabras de la lista A | +0,01 | 0,89 |
| SEN, fragmentos < 30 palabras (rótulos, filas de tabla, listas) | **−0,47 [−0,73; −0,21]** | 0,0006 |
| … sin las palabras de la lista A | −0,10 | 0,37 |

- **FinBERT también ve la caída de 2024, pero no desaparece al quitar el
  vocabulario de temas ESRS.** Viene de **menos frases positivas**, no de más
  negativas, y se concentra en los pasajes E y S (en G no cae). Es el mismo
  patrón que ClimateBERT (menos pasajes de oportunidad).
- La caída de Z(SEN) sí es vocabulario ESRS, y está sobre todo en los
  **fragmentos cortos**: allí las palabras de la lista A se triplican en 2024
  (1,3 → 3,8 por 1.000 palabras; en los pasajes largos, 1,4 → 2,2), y sin ellas
  la caída desaparece. Son las filas y rótulos de las tablas de datos de las
  ESRS (incidentes, quejas, impactos negativos), que FinBERT no lee porque la
  segmentación descarta los fragmentos de menos de 30 palabras.
- **Lectura**: en 2024 hay dos cambios distintos. (i) Z(SEN) baja por el
  vocabulario prescrito de las tablas ESRS (artefacto de diccionario, como se
  dijo en el #4). (ii) Por debajo, el texto narrativo E y S es algo menos
  positivo, según dos instrumentos independientes (FinBERT, ClimateBERT); SEN
  sin la lista A no lo registra (+0,01), lo que indica que el diccionario L&M
  capta peor el tono positivo contextual. La frase del paper "la caída
  desaparece sin la lista A" sigue siendo cierta para SEN, pero no conviene
  presentarla como prueba de que el tono del texto no cambió.

## 4. Experimento 2: pilares

Cuotas por informe dentro del texto ESG (pasajes E, S o G / pasajes ESG) frente
a nuestras cuotas de menciones por pilar (`pillar_by_doc`, E sin TRANS):

| Par | Agregado ρ | Intra r | Entre r |
|---|---|---|---|
| G FinBERT-ESG ~ G nuestro | 0,83 [0,76; 0,88] | 0,84 | 0,87 |
| G ESGBERT ~ G nuestro | 0,83 [0,74; 0,88] | 0,86 | 0,87 |
| E FinBERT-ESG ~ E nuestro | 0,88 [0,82; 0,93] | 0,83 | 0,93 |
| E ESGBERT ~ E nuestro | 0,85 [0,78; 0,91] | 0,83 | 0,90 |
| S FinBERT-ESG ~ S nuestro | 0,69 [0,59; 0,77] | 0,47 | 0,83 |
| S ESGBERT ~ S nuestro | 0,70 [0,59; 0,79] | 0,54 | 0,79 |
| cuota de pasajes ESG (ESGBERT, algún pilar) ~ SUS (densidad) | 0,72 [0,60; 0,81] | 0,76 | 0,73 |

Las correlaciones cruzadas son negativas (composición: E sube cuando G baja) y
la diagonal domina en todas las vistas. Densidades: cuota de pasajes E de
ESGBERT ~ densidad E+TRANS 0,88; S ~ densidad S 0,72; G ~ densidad G 0,58.

**Tendencias** (medias de los informes 2018 → 2024; pendiente con EF de empresa):

| Medida | 2018 | 2024 | Pendiente/año | p |
|---|---|---|---|---|
| G nuestro (cuota de menciones) | 0,44 | 0,24 | −0,036 | < 10⁻¹⁰ |
| G FinBERT-ESG (cuota dentro de ESG) | 0,45 | 0,29 | −0,027 | < 10⁻⁸ |
| G ESGBERT (cuota dentro de ESG) | 0,33 | 0,21 | −0,021 | < 10⁻⁶ |
| G FinBERT-ESG, pasajes G / todos los pasajes | 0,28 | 0,20 | −0,015 | < 10⁻⁷ |
| G ESGBERT, pasajes G / todos los pasajes | 0,20 | 0,17 | −0,004 | 0,02 |
| E FinBERT-ESG (dentro de ESG) | 0,22 | 0,36 | +0,025 | < 10⁻¹³ |
| S FinBERT-ESG (dentro de ESG) | 0,33 | 0,35 | +0,002 | 0,54 |

**La caída de G se reproduce con ambos modelos**, en la misma dirección y con
magnitud algo menor (−16 y −12 puntos frente a −21). También cae en términos
absolutos (pasajes G sobre todos los pasajes), con claridad en FinBERT-ESG y
débilmente en ESGBERT, lo que apoya, con matiz, la frase del paper de que
gobernanza "pierde terreno en términos absolutos, no solo por dilución".

## 5. Experimento 3: recall de la extracción

98 informes (2 por empresa, 14 por año, diseño equilibrado, semilla fija). Se
releyeron los PDF con `LexicalDocumentFilter._extract_paragraphs` y PyMuPDF
1.27.2 (el entorno del pipeline; con 1.22.5 los bloques cambian), se
reconstruyeron las zonas con la regla del filtro y **coinciden exactamente con
los JSON guardados en los 98 informes**. Texto extraído y no extraído se
segmentan igual que en el #4 (pasajes ≥ 30 palabras) y se muestrean 80 pasajes
no extraídos y 40 extraídos por informe (11.556 pasajes), clasificados con
ESGBERT E/S/G y FinBERT-ESG.

- La extracción conserva el **52 %** de las palabras de los párrafos del PDF.
  El 96 % de los pasajes no extraídos no tiene ninguna coincidencia del
  vocabulario.
- Cuota de pasajes (ponderada por palabras) clasificada como ESG: ESGBERT 55 %
  en lo extraído frente a 7,2 % en lo no extraído; FinBERT-ESG 67 % frente a 14 %.
- **Cuota estimada del texto ESG que queda fuera de la extracción** (ratio
  ponderado por palabras; IC por bootstrap de empresas y de pasajes):

  | Clasificador | Total | E | S | G |
  |---|---|---|---|---|
  | ESGBERT | **8,1 % [6,7; 9,8]** | 4,1 % [2,7; 5,7] | 5,1 % [3,6; 6,8] | 12,9 % [9,9; 16,7] |
  | FinBERT-ESG | **13,0 % [11,0; 15,2]** | 3,1 % [1,4; 5,1] | 20,5 % [17,1; 24,3] | 12,8 % [9,9; 16,1] |

  Por año (ESGBERT, 14 informes por año): 8,5 / 13,8 / 11,4 / 7,9 / 5,0 / 10,9
  / 4,6 % de 2018 a 2024 (`recall_by_year.csv`); sin tendencia clara.
- **Es una cota superior indicativa.** Se desconoce la tasa de falsos positivos
  de los clasificadores en texto fuera de dominio. En una lectura manual de 40
  pasajes no extraídos que ESGBERT marca como ESG, unos 13 son claramente
  divulgación ESG (formación y clima laboral, accidentes laborales, sustancias
  reguladas, normas de emisiones de vehículos, consumo responsable,
  cumplimiento), unos 6 dudosos y unos 21 no lo son (sobre todo informes de
  auditoría y responsabilidades del consejo sobre los estados financieros, que
  GovernanceBERT marca como gobernanza; biografías de consejeros; descripciones
  técnicas de producto). Con esa precisión (33–48 %), la parte ESG que la
  extracción deja fuera rondaría el **3–4 %** del texto ESG; lo perdido se
  concentra en S (capital humano) y en gobernanza de cumplimiento.
- El pilar E apenas pierde texto (3–4 %, cota superior): el vocabulario
  ambiental es el más completo.

## 6. Experimento 4: sustancia concreta

- **Acción** (EnvironmentalBERT-action en pasajes ambientales; cuota de
  pasajes de acción entre los ambientales): ~ Z(SUS) −0,03 [−0,23; 0,17]
  (intra −0,23 [−0,37; −0,09]); ~ Z(QUANT) 0,08 [−0,13; 0,30]. **No converge.**
  La cuota de pasajes de acción sobre *todos* los pasajes sí sigue a Z(SUS)
  (0,59), pero solo porque sigue a la cuota de texto climático (0,82).
- **Objetivos** (NetZero: net zero o reducción, sobre todos los pasajes
  climáticos): ~ Z(SUS) 0,24 [0,06; 0,41]; ~ Z(QUANT) 0,22 [0,05; 0,40]; solo
  reducción ~ Z(QUANT) 0,31 [0,11; 0,49]; ~ especificidad de ClimateBERT 0,22.
  Convergencia débil pero positiva y con el signo esperado. Con el índice:
  ESGSI −0,19 [−0,33; −0,03], ESGSI_ext −0,22 [−0,38; −0,07] (más objetivos,
  menos tono sobre sustancia).
- Tendencias: los pasajes de objetivos crecen (5,2 % → 10,5 % de los
  climáticos; 1,6 % → 4,6 % de todos; p < 10⁻¹⁰) y se estancan en 2024; la
  cuota de acción entre los ambientales cae (0,54 → 0,45), sobre todo en 2024
  (−0,70 DT), coherente con el texto más prescrito y menos comprometido que
  ClimateBERT vio en 2024.
- **Lectura**: SUS mide cuánto texto ESG hay (converge con los detectores de
  tema: 0,72), no cuánto de él es acción concreta; QUANT converge débilmente
  con los objetivos de reducción. Mismo diagnóstico que el #4 con la
  especificidad.

## 7. Valoración

Lo que valida parcialmente la propuesta:

1. **Tono**: validado con un segundo instrumento independiente y de dominio
   financiero, sobre todo el texto (ρ = 0,77 con Z(SEN); 0,46 por pasaje,
   positivo en todos los informes) y en los tres pilares (también S y G). Es
   la evidencia más fuerte del issue.
2. **Pilares**: nuestras cuotas por pilar coinciden con las de dos
   clasificadores supervisados (0,83–0,88 en E y G, 0,70 en S) y la caída de G
   2018–2024 se reproduce con ambos.
3. **Recall**: la extracción por vocabulario deja fuera poco texto ESG: cota
   superior del 8 % (ESGBERT) o 13 % (FinBERT-ESG), probablemente 3–4 %
   corrigiendo por falsos positivos; casi nada del pilar E.

Lo que no:

1. **SUS no es sustancia concreta**: no converge con la cuota de acción y solo
   débilmente con la de objetivos (0,24); es una medida de volumen temático.
2. **La ruptura de 2024 no es solo vocabulario ESRS.** La caída de Z(SEN) sí lo
   es (y está en las tablas), pero FinBERT y ClimateBERT ven además un texto
   narrativo menos positivo en E y S que SEN no capta.
3. Todo se mide sobre el mismo texto extraído y con clasificadores cuya
   precisión fuera de dominio es desconocida; ninguna prueba es un criterio
   externo de conducta.

## 8. Recomendación para el paper

Página muy justa: **añadir 3–4 frases a §5**, no un párrafo nuevo, integradas
en "An independent text instrument" (borrador en
`draft_robustness_esg_models.tex`, ≈ 190 palabras, para sustituir o ampliar el
cierre del párrafo actual):

1. FinBERT sobre todo el texto: ρ = 0,77 con Z(SEN), también en S y G.
2. Pilares: cuotas convergentes (0,83 en G) y la caída de G se reproduce.
3. Matizar 2024: la caída de Z(SEN) está en el vocabulario de las tablas ESRS,
   pero los dos modelos ven además menos tono positivo en el texto narrativo.
4. **§6** (limitación del vocabulario: "its recall is unknown"): sustituir por
   la cota (≤ 8 % del texto que ESGBERT clasifica como ESG queda fuera de la
   extracción en 98 informes; menos tras revisar falsos positivos). Una frase.

Si no hay sitio, la prioridad es (3), porque matiza una afirmación ya
publicada en §4.1/§5, y después (4). Referencias nuevas: `huang2023finbert`,
`schimanski2024esgbert` (y `schimanski2023netzero` solo si se cita el resultado
de objetivos) en `refs.bib`.
