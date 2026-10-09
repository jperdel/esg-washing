# Issue #1 — La tendencia del QUANT

Script: `scripts/issue1_quant_eda.py` (todas las cifras de este documento salen de él).

```
PYTHONIOENCODING=utf-8 python scripts/issue1_quant_eda.py --root <clon con data/ y results/analysis_v3>
```

Corpus: 343 informes, 49 empresas, 2018–2024 (panel equilibrado, 49 informes por año).
Pendientes: año con efectos fijos de empresa y errores agrupados por empresa, igual que en
el paper. Las empresas solo aparecen como identificadores anónimos `<Industria>-NN`.

## 1. Cómo se calcula el QUANT

`ESGSIAnalyzer.calculate_quant_scores` (`src/esgsi_analyzer.py`) con los patrones de
`config.QUANT_PATTERNS` (`src/config.py`), aplicados sobre el **texto crudo de las zonas
ESG extraídas** (no sobre el PDF completo ni sobre el texto lematizado):

```
QUANT_d = (aciertos de los 4 patrones en el texto) / (nº de tokens separados por espacios)
```

| Familia | Patrón (resumido) | Qué cuenta en realidad |
|---|---|---|
| percentages | `\d+([.,]\d+)?\s*%` | porcentajes (incluidas las celdas `0%` de las plantillas de taxonomía) |
| large_numbers | `\d{4,}` que no sea `19xx`/`20xx` | cifras de 4+ dígitos **sin separador de miles** (no cuenta `14,918`), códigos ISO (14001, 45001…), números de actos legales (2021/2139, 2022/2464…), rangos de años pegados (`20212023`, la extracción quita el guion), 2100 |
| units | `tonne, ton, mt, co2e?, ghg, kwh, mwh, …` | sobre todo las **palabras** `co2`, `ghg`, `co2e` (73 % de los aciertos de la familia), no unidades tras una cifra |
| frameworks | `gri, tcfd, sasb, issb, sdg, ungc, sfdr, csrd, un global compact, paris agreement, taxonomy` | nombres de marcos; `taxonomy` es el 54 % de la familia |

El cálculo es correcto respecto a su definición: el script lo reproduce a precisión completa
(diferencia máxima 1e-16 con la columna del análisis). El problema no es de código, sino de
qué mide la definición.

## 2. El 4,7 no es grande comparado con el SUS, pero sí rápido para lo que dice medir

El 4,7 de la tabla de tendencias está en **unidades de 1e-4 por año**. El SUS está en
menciones por 100 tokens y el HEDGE también en 1e-4, así que los números no son comparables
entre sí. Puestos en la misma escala (`pendientes_componentes.csv`):

| Componente | Pendiente (unid. propias) | DT del corpus por año | % de la media 2018 por año | Cambio 2018→2024 | Aporte a la pendiente del ESGSI_ext (−0,260) |
|---|---|---|---|---|---|
| SUS (densidad) | 0,366 | **0,196** | 6,4 % | +40 % | −0,196 |
| QUANT | 4,7e-4 | **0,150** | 9,7 % | +49 % | −0,075 |
| SEN | 0,002 | 0,015 | — | — | +0,015 |
| HEDGE | −0,4e-4 | −0,010 | −0,2 % | −0,4 % | −0,005 |

En desviaciones típicas el QUANT crece menos que el SUS. Aun así es una tendencia clara
(p < 1e-5, R² intra-empresa 0,21, 40 de 49 empresas con pendiente positiva) y explica
−0,075 de los −0,260 del ESGSI extendido por año.

## 3. De dónde sale la pendiente

El QUANT es una suma de densidades y la pendiente con efectos fijos es lineal, así que la
pendiente total se descompone **exactamente** en la de cada familia y cada subtipo
(`pendiente_por_familia.csv`, `pendiente_por_subtipo.csv`, `fig_aporte_a_la_pendiente.png`).

| Familia | Pendiente (1e-4/año) | Cuota de la pendiente | Cuota de aciertos 2018 → 2024 |
|---|---|---|---|
| frameworks | 2,52 | **54 %** | 6,9 % → 20,5 % |
| units | 1,34 | **29 %** | 18,5 % → 22,3 % |
| percentages | 0,55 (p = 0,40) | 12 % | 67,1 % → 48,8 % |
| large_numbers | 0,27 | 6 % | 7,5 % → 8,4 % |
| **Total** | **4,67** | 100 % | |

Por subtipo (`pendiente_por_grupo_de_subtipos.csv`):

- **La palabra `taxonomy`: +2,29 (49 % de la pendiente).** Pasa de 7 apariciones en 2018 a
  6.462 en 2024: la obligación de informar según la Taxonomía de la UE empieza con el
  ejercicio 2021.
- **Las palabras `CO2`/`CO2e`/`GHG`: +1,21 (26 %).** Son palabras de tema, no unidades; además
  están en el vocabulario del SUS, así que se cuentan dos veces en el ESGSI extendido.
- **`CSRD`/`SFDR`/`ISSB`: +0,35 (8 %).** De 0 apariciones en 2018 a 1.655 en 2024.
- **Porcentajes de las tablas de taxonomía: +1,02 (22 %).** Las celdas `0%` (+0,51; de 87 a
  1.777) y los porcentajes en un entorno de ±300 caracteres que menciona taxonomía o
  elegibilidad (+0,51). El resto de porcentajes **baja** (−0,47).
- **Números de actos legales de la UE (2021/2139, 2021/2178, 2022/2464, 2023/2486…): +0,27 (6 %).**
- Unidades físicas reales, cifras grandes "de verdad", códigos ISO y GRI/SASB/TCFD: ≈ 0.

En conjunto, los subtipos ligados a la **Taxonomía de la UE y a la CSRD** (nombres de marcos,
plantillas de KPI de taxonomía y números de los reglamentos) **y las palabras CO2/GHG** suman
+5,1, más que la pendiente total (4,67). La parte que cuenta cifras (porcentajes fuera de esas plantillas,
cantidades con unidad, cifras grandes) es plana o baja.

### Lo que no explica la tendencia

- **No depende de unas pocas empresas ni de unos pocos informes.** Las cinco empresas con más
  pendiente aportan el 33 %. Si se deja fuera una empresa cualquiera, la pendiente va de 4,41 a
  5,22. Sin el 1 % superior de informes es 5,00, winsorizada al 1–99 % es 5,04 y en logaritmos
  sube un 8,8 % al año. La mediana anual sube de 44 a 73 (en 1e-4).
- **No es la longitud del texto.** Controlando el log de los tokens extraídos, la pendiente es
  4,40. Los tokens extraídos crecen un 112 % entre 2018 y 2024, los aciertos un 213 %, las
  páginas del PDF un 34 % y el rendimiento de la extracción pasa del 32 % al 48 %. El QUANT
  actual sí está correlacionado con la longitud dentro de la empresa (r = 0,29): los informes
  que crecen lo hacen con secciones de taxonomía y clima. Con la propuesta de abajo esa
  correlación es de 0,03.
- **No es el tipo de documento ni el país.** La pendiente es positiva en Annual report (5,3) y
  URD (5,0), en 6 de 7 países (Italia −1,3, 5 empresas) y en 8 de 10 industrias. Las que bajan
  son Utilities (−11,4; 2 empresas, ya partían del nivel más alto) y Health Care (−0,1)
  (`por_grupo.csv`).
- **No es un artefacto de la extracción de años.** Los años 19xx/20xx ya se excluyen. Solo se
  cuelan los rangos de años pegados y "2100" (+0,05).

## 4. ¿Es correcta?

El cálculo es correcto y la tendencia es real, pero **no mide lo que el QUANT dice medir**
("el informe aporta datos duros"). Más de la mitad de la pendiente sale de contar nombres de
regímenes de divulgación (`taxonomy`, `CSRD`, `SFDR`) que no existían o no eran obligatorios
en 2018. Otra cuarta parte sale de palabras de tema (`CO2`, `GHG`) que ya están en el SUS. El
resto procede sobre todo de las plantillas de la taxonomía (celdas `0%`, números de
reglamento). Es la misma confusión con el mandato regulatorio que el paper discute para el
SUS (§ mandate confound), pero amplificada: aquí el término de taxonomía por sí solo aporta la
mitad de la tendencia. Además, el QUANT duplica en parte al SUS: la correlación intra-empresa
QUANT–SUS es 0,58 con la definición actual y 0,23 con la propuesta.

## 5. Propuesta y efecto

Prototipos (`variantes_quant.csv`, `fig_variantes_z_por_ano.png`). El ESGSI extendido se
recalcula como Z(SEN) − Z(SUS) − 0,5·Z(QUANT_v) + 0,5·Z(HEDGE).

| Variante | QUANT: pendiente (1e-4/año) | p | DT/año | ρ Spearman con QUANT actual | ESGSI_ext: pendiente [IC 95 %] | ESGSI_ext: ρ con el actual | Solapamiento decil sup./inf. del ESGSI_ext |
|---|---|---|---|---|---|---|---|
| V0 actual | 4,67 | <1e-5 | 0,150 | 1 | −0,260 [−0,340, −0,180] | 1 | 1 / 1 |
| V1 sin marcos | 2,15 | 0,007 | 0,076 | 0,93 | −0,223 [−0,303, −0,144] | 0,993 | 0,97 / 0,94 |
| V2 sin marcos ni CO2/GHG | 0,94 | 0,19 | 0,037 | 0,84 | −0,204 [−0,281, −0,126] | 0,987 | 0,97 / 0,94 |
| **V2b propuesta**: V2 sin códigos ISO, actos legales ni rangos de años | **0,64** | **0,36** | **0,025** | 0,82 | **−0,198 [−0,275, −0,121]** | **0,984** | **0,97 / 0,94** |
| V3 "solo cifras" (cada número una vez: %, cantidad con unidad, cifra ≥ 1.000 con o sin separador) | −1,35 | 0,31 | −0,035 | 0,68 | −0,168 [−0,247, −0,089] | 0,966 | 0,91 / 0,88 |
| Solo porcentajes | 0,55 | 0,40 | 0,024 | 0,77 | −0,197 [−0,275, −0,120] | 0,980 | 0,94 / 0,85 |

**Propuesta (V2b):** el QUANT debe contar solo cifras. Hay que quitar la familia
`frameworks`, quitar `co2`, `co2e`, `ghg` y `ktco2` de `units` (son tema, no unidad, y ya están
en el SUS) y excluir de `large_numbers` los códigos de norma, los números de actos legales y
los rangos de años. Con esta definición:

- La pendiente del QUANT pasa de 4,67 a 0,64 (de 0,150 a 0,025 DT/año) y deja de ser
  significativa.
- El orden de los informes cambia poco: ρ = 0,82 con el QUANT actual.
- El ESGSI extendido apenas cambia en sección cruzada (ρ = 0,984, solapamiento de deciles
  extremos 0,97/0,94). Su pendiente pasa de −0,260 a −0,198, todavía significativa y más cerca
  de la del ESGSI (−0,181).

V3 ("solo cifras" con todas las cifras ≥ 1.000, también las que llevan separador de miles)
sale con pendiente negativa. Las cifras grandes de la parte financiera y de gobierno
(remuneración, capital, plantilla) crecen menos que el texto extraído (−2,1 en 1e-4). El
signo es frágil: sin el 1 % superior de informes la pendiente es −0,07. Por eso no se propone
como sustituto sin una decisión sobre qué cifras grandes cuentan.

No se ha tocado `src/config.py`. Cambiar `QUANT_PATTERNS` cambia todas las cifras del ESGSI
extendido en el paper y exige volver a ejecutar `main.py` y `scripts/analysis_v3.py`.

## 6. Preguntas abiertas para el autor

1. ¿El QUANT pretende medir **cifras** o también **referencias a marcos**? Si se quieren los
   marcos como "anclas de verificabilidad", deberían ser un componente aparte: hoy duplican
   el pilar transversal del SUS y casi toda su tendencia es la Taxonomía y la CSRD.
2. ¿Se aceptan V2b y el cambio de la pendiente del ESGSI extendido (−0,260 → −0,198) en el
   paper?
3. ¿Cuentan las celdas `0%` de las plantillas de taxonomía? Son cifras divulgadas, pero no
   reflejan esfuerzo cuantitativo de la empresa (+0,51 de la pendiente; V3b las excluye).
4. Las cifras grandes: hoy no cuentan `14,918` (separador de miles) pero sí `14001` (ISO). ¿Se
   prefiere quitarlas (más simple) o definirlas bien? El sentido de V3 muestra que esa
   elección mueve el signo de la tendencia.
5. El extractor elige zonas por densidad de palabras ESG que incluyen `taxonomy`, `CO2` y
   `GHG`, y el rendimiento sube del 32 % al 48 %. Más secciones de taxonomía significan más
   texto extraído con más aciertos QUANT. ¿Merece un párrafo en limitaciones?

## Ficheros

| Fichero | Contenido |
|---|---|
| `pendientes_componentes.csv` | pendientes de SUS, SEN, QUANT y HEDGE en unidades propias, DT/año, % de la media y aporte al ESGSI_ext |
| `pendiente_por_familia.csv`, `pendiente_por_subtipo.csv`, `pendiente_por_grupo_de_subtipos.csv` | descomposición exacta de la pendiente |
| `variantes_quant.csv` | variantes del QUANT y del ESGSI extendido (pendiente, Spearman, deciles extremos) |
| `pendiente_por_firma_*.csv` | pendiente por empresa (identificador anónimo) y su cuota |
| `por_grupo.csv` | medias 2018/2024 y pendiente media por industria, país y tipo de documento |
| `numerador_denominador_por_ano.csv` | aciertos, tokens extraídos, páginas y palabras del PDF, rendimiento de la extracción |
| `medias_anuales.csv`, `summary.json` | medias anuales y todos los escalares |
| `fig_*.png` | familias por año, aporte a la pendiente, variantes estandarizadas |
