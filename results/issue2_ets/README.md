# Issue #2: emisiones verificadas del EU ETS como criterio externo gratuito

Informe de trabajo (octubre de 2026). Es un piloto de viabilidad. Las hipótesis de pago del issue #2
(dispersión de analistas y controversias) están bloqueadas, así que aquí se busca un criterio externo
**gratuito** de "sustancia": las emisiones verificadas de las instalaciones que cada grupo tiene en el
Régimen de Comercio de Derechos de Emisión de la UE (EU ETS).

La prueba enfrenta lo que se dice con lo que se hace. La primera pregunta es si el tono relativo a la
sustancia de un informe (ESGSI y ESGSI extendido, como scores continuos) se asocia con la evolución
posterior de las emisiones verificadas. La segunda es si la densidad de sustancia (SUS) sigue el nivel o
la intensidad de las emisiones.

Se siguen las directrices de `paper/_research/directrices.md`. Los scores son continuos, sin umbrales ni
etiquetas, y se usa solo el vocabulario actual. Ninguna empresa aparece por su nombre: se usan los
identificadores anónimos `<Industria>-NN` de `analysis_v3`. Los nombres solo están en ficheros
`*_internal*` de `data/issue2_ets/`, que no se versionan.

## 0. Resumen

| Pregunta | Respuesta corta |
|---|---|
| ¿Hay datos gratuitos? | **Sí**, sin registro ni pago. La Comisión Europea publica las emisiones verificadas por instalación de 2008 a 2025 y la lista de operadores, con el LEI del titular. EUETS.INFO publica el histórico de titulares de cuentas. GLEIF ofrece la estructura de grupo (matriz → filiales), y Wikidata y el BCE dan los ingresos que sirven de denominador. |
| ¿Cobertura? | **23 de 49** empresas tienen instalaciones fijas con emisiones verificadas en 2018-2025. De ellas, **19** emiten de forma material, con una media de al menos 25 kt CO2e/año en 2018-2024. Por industria: Utilities 2/2, Energy 2/2, Basic Materials 2/2, Consumer Discretionary 5/10, Consumer Staples 3/5, Health Care 2/3, Industrials 3/9 materiales (5/9 cubiertas), Financials 0/11 materiales (2 con centros de datos de menos de 2 kt), Technology 0/3 y Telecommunications 0/2. |
| ¿Asociación prospectiva? | **Ninguna.** Por cada desviación típica (DT) de ESGSI, la variación de log-emisiones entre t y t+1 es de +0,020 (EF de año; p agrupado 0,33, p wild 0,45), +0,025 (EF de año e industria; p 0,23 y 0,33) y +0,006 (EF de empresa y año; p 0,90 y 0,86). El ESGSI extendido, SEN y SUS por separado, el horizonte t+2 y la variante like-for-like dan lo mismo: 0 de 54 contrastes con p wild < 0,05. |
| ¿SUS sigue las emisiones? | **No dentro de cada industria.** Entre empresas, QUANT crece y HEDGE baja con el nivel y con la intensidad de emisiones (Spearman de medias de empresa: QUANT ρ = 0,48 con el nivel; HEDGE ρ = −0,53 con la intensidad). Con efectos fijos de industria todo se queda en ruido. La densidad SUS y el ESGSI no se asocian con ninguno de los dos. |
| ¿Potencia? | Es baja. Con G = 19 empresas, el efecto mínimo detectable (MDE) al 80 % es de 0,058 log-puntos por DT con EF de año (el 37 % de la DT del resultado) y de 0,134 con EF de empresa y año (el 86 %). Cualquier efecto plausible, del orden de 1-2 puntos porcentuales anuales por DT, queda muy por debajo. |
| ¿Vale como validación externa para el paper? | **No como prueba central.** El ámbito no casa: el ETS cubre las instalaciones fijas de la UE y el EEE, una parte del alcance 1. El perímetro de grupo es estático. Los cambios de emisiones los mueven la demanda, los precios de la energía y los cierres de carbón, no la calidad de la divulgación. Y N es pequeño. Sirve, como mucho, como una frase o nota en limitaciones o robustez: un piloto nulo con su MDE (ver §7). |

## 1. Fuentes (todas gratuitas, sin registro)

Descargadas el 9 de octubre de 2026 por `scripts/ets_fetch.py`. Las URL, los tamaños y los SHA-256 están en
`fuentes_ets.json`. Los datos van a `data/issue2_ets/` (no versionado) y se leen con `python -I`.

| Fuente | Contenido | Cobertura | Licencia / uso | Uso aquí |
|---|---|---|---|---|
| **Comisión Europea (DG CLIMA), Union Registry**: `verified_emissions_2025_en.xlsx` | emisiones verificadas por instalación (t CO2e), asignación gratuita, actividad y tipo de cuenta (fija, aviación, marítima) | 2008-2025, 18.993 cuentas (UE-27 + EEE + Irlanda del Norte); extracción del 1-4-2026 | reutilización con cita (política de reutilización de la Comisión, Decisión 2011/833/UE; CC BY 4.0) | **variable de resultado** |
| CE, `policy_ets_registry_operators_ets_en.xlsx` | operador (titular de la cuenta), n.º de registro mercantil, **LEI** (4.070 de 16.394 filas), instalación y permiso | foto actual | igual | cruce por LEI y por nombre |
| CE, `compliance_2025_code_en.xlsx` | estado de cumplimiento y primer y último año de emisiones | 2025 | igual | contexto (sin uso en modelos) |
| **EUETS.INFO** (J. Abrell), versión de octubre de 2024 (EUTL 15.1.3): `eutl_2024_202410.zip` y su documentación | tablas procesadas del EUTL: cuentas (también **cerradas**), titulares con LEI, instalaciones, cumplimiento y transacciones; incluye el `bvdId` del emparejamiento JRC-Orbis | 2005-2023 | datos abiertos con cita (Abrell, 2024); la página no declara una licencia explícita | **histórico de titulares** para fijar el perímetro de grupo |
| **GLEIF API** (Level 1 y Level 2) | LEI de las 49 matrices (por ISIN) y sus filiales consolidadas (`ultimate-children`: 5.414 entidades) | golden copy del 9-10-2026 | CC0 | cruce por LEI |
| **Wikidata** SPARQL (P2139 por ISIN P946) | ingresos totales anuales (EUR o USD) | irregular; 101 de los 133 empresa-año de la muestra principal | CC0 | denominador de la intensidad |
| **BCE** (EXR, USD/EUR, media anual) | tipo de cambio | 1999-2025 | reutilización con cita | convertir los ingresos en USD |

**Revisadas y no usadas.** El conjunto de datos de la AEE (*EU ETS data from the Union Registry*, julio de 2026, 2005-2025) está
agregado por país y actividad, no por instalación, así que no permite asignar emisiones a una empresa. El *Ownership Links and
Enhanced EUTL Dataset* (Jaraitė et al.; EUI Cadmus) es de las primeras fases del ETS y no cubre 2018-2025. El enlace
EUTL-Orbis del JRC (*JRC-EU-ETS-FIRMS*) da BvD ID de cuenta, no la matriz última, y su página web rechazó el acceso
automatizado. El propio Orbis es de pago.

## 2. Cruce instalación → empresa (`scripts/ets_match.py`)

1. **Vía LEI.** Hay acierto cuando el LEI del operador actual o el de cualquier titular, actual o pasado, de una cuenta
   ligada a la instalación pertenece a la matriz o a una de sus filiales según GLEIF Level 2. Los LEI del registro
   vienen con guiones y se normalizan a 20 caracteres.
2. **Vía nombre.** Expresiones regulares por empresa (marcas y filiales conocidas, con exclusiones para escisiones y
   homónimos) sobre el nombre de esos titulares o de la instalación.
3. **Revisión manual** de todos los aciertos, en un fichero interno con el motivo de cada descarte. Se descartaron 32. Casi
   todos eran homónimos o abreviaturas (la abreviatura de «fábrica», siglas de procesos químicos que coinciden con una marca, topónimos). Otros eran
   instalaciones que gestiona un tercero (empresas de servicios energéticos o municipales) dentro de la planta de un
   grupo: se aplica el criterio del operador.
4. **Una instalación, una empresa.** No quedó ninguna instalación asignada a dos grupos.

Resultado: **443 instalaciones fijas** (113 por LEI, 324 por nombre del titular y 6 solo por nombre de la instalación,
revisadas una a una) y 9 cuentas de aviación de 7 empresas.

**Perímetro.** Es el de "instalaciones que alguna vez fueron del grupo", fijo en el tiempo. Así se evitan saltos por
compras y ventas en Δlog E, a cambio de seguir contando una planta vendida bajo su nuevo dueño. GLEIF da la estructura
de **hoy**, no la histórica. Las escisiones grandes del periodo quedan fuera por exclusión explícita. Se excluyen
el registro británico (GB, fuera del EU ETS desde 2021) e Irlanda del Norte (XI). El transporte marítimo (en el ETS
desde 2024) no entra.

### 2.1 Cobertura por industria

| Industria | Empresas | Con instalaciones fijas (emisiones > 0) | Materiales (≥ 25 kt/año) | Con aviación | Instalaciones fijas |
|---|---|---|---|---|---|
| Basic Materials | 2 | 2 | 2 | 1 | 99 |
| Consumer Discretionary | 10 | 5 | 5 | 2 | 68 |
| Consumer Staples | 5 | 3 | 3 | 0 | 12 |
| Energy | 2 | 2 | 2 | 0 | 82 |
| Financials | 11 | 2 | 0 | 1 | 7 |
| Health Care | 3 | 2 | 2 | 0 | 10 |
| Industrials | 9 | 5 | 3 | 2 | 79 |
| Technology | 3 | 0 | 0 | 1 | 2 |
| Telecommunications | 2 | 0 | 0 | 0 | 0 |
| Utilities | 2 | 2 | 2 | 0 | 84 |
| **Total** | **49** | **23** | **19** | **7** | **443** |

Detalle por `firm_id` en `cobertura_ets.csv` y `muestra_ets.csv`. En Financials solo hay centros de datos con calderas o
grupos electrógenos de más de 20 MW (menos de 2 kt/año). Los dos casos fuera de la muestra material en Industrials son
plantas móviles de asfalto y una sala de calderas. Su log-variación es ruido: Δlog con DT de 0,7 a 1,8.

### 2.2 Desajuste de ámbito (importante)

- El ETS cubre las **instalaciones fijas de la UE y el EEE**: combustión de más de 20 MW, refino, química, metales,
  minerales y papel. Es una **parte del alcance 1** del grupo. Quedan fuera las plantas fuera de Europa, la
  combustión pequeña, las flotas, el alcance 2 y **el alcance 3**, que domina en automoción, bienes de consumo,
  farmacia y aeroespacial.
- En utilities, energía y materiales el ETS recoge una parte grande de la huella europea. En el resto son las calderas
  de las fábricas: un indicador del consumo energético de la planta, no del perfil climático del grupo.
- La aviación (vuelos dentro del EEE de aerolíneas de carga o corporativas) se cuenta aparte y no entra en los modelos.

## 3. Panel y variables

- `panel_ets.csv`: empresa-año 2013-2025, con emisiones fijas (t CO2e), instalaciones activas, emisiones de aviación y
  Δlog *like-for-like* (solo instalaciones con emisiones en t−1 y en t).
- Muestra principal: **19 empresas materiales × 7 informes (2018-2024) = 133** observaciones para h = 1 y 114 para h = 2
  (los informes de 2024 no tienen t+2 todavía).
- Resultados: `dlog1` = log E(t+1) − log E(t) (DT 0,155), `dlog2` = log E(t+2) − log E(t) (DT 0,218) y `lfl1` =
  Δlog like-for-like de t a t+1 (DT 0,144). El informe del ejercicio t se publica en t+1, así que t+1 es el primer
  año "posterior".
- Regresores: ESGSI, ESGSI extendido, Z(SEN), Z(SUS), Z(QUANT) y Z(HEDGE) de `analysis_v3/doc_level.csv`, cada uno por
  separado y **reestandarizado en la muestra**. Así β es el cambio en log-puntos por cada DT del score.
- Las emisiones agregadas de las 19 bajan de 117 Mt (2018) a 63 Mt (2024). La caída se concentra en la generación
  eléctrica con carbón. Hay choques comunes fuertes: la Δlog media es de −0,11 en 2020, −0,14 en 2023 y −0,13 en 2024. Los
  efectos fijos de año los absorben.
- Varianza del ESGSI que queda tras quitar EF de empresa y año en esta muestra: 28 % (31 % en el extendido).

## 4. Resultados A: score en t → emisiones en t+1 y t+2 (`modelos_ets.csv`)

MCO con SE agrupados por empresa (CR1, t con G−1 gl) y wild cluster bootstrap restringido (Rademacher, 9.999
réplicas), con los estimadores de `fin_econ.py`. G = 19. Un β > 0 indica que un score más alto va con una variación
de emisiones más alta (menos reducción) después.

**Δlog E(t → t+1)**, muestra material (N = 133):

| Score | EF | β | SE | p agrupado | p wild | MDE 80 % | MDE / DT(y) |
|---|---|---|---|---|---|---|---|
| ESGSI | año | +0,020 | 0,020 | 0,33 | 0,45 | 0,058 | 0,37 |
| ESGSI | año + industria | +0,025 | 0,020 | 0,23 | 0,33 | 0,059 | 0,38 |
| ESGSI | empresa + año | +0,006 | 0,045 | 0,90 | 0,86 | 0,134 | 0,86 |
| ESGSI ext. | año | +0,024 | 0,021 | 0,29 | 0,62 | 0,064 | 0,41 |
| ESGSI ext. | año + industria | +0,035 | 0,021 | 0,11 | 0,33 | 0,061 | 0,39 |
| ESGSI ext. | empresa + año | +0,025 | 0,037 | 0,51 | 0,68 | 0,111 | 0,71 |
| Z(SEN) | año | +0,014 | 0,011 | 0,21 | 0,17 | 0,032 | 0,21 |
| Z(SEN) | año + industria | +0,028 | 0,014 | 0,06 | 0,17 | 0,042 | 0,27 |
| Z(SEN) | empresa + año | +0,002 | 0,024 | 0,95 | 0,95 | 0,072 | 0,47 |
| Z(SUS) | año | −0,009 | 0,018 | 0,62 | 0,82 | 0,052 | 0,34 |
| Z(SUS) | año + industria | −0,003 | 0,013 | 0,85 | 0,84 | 0,040 | 0,26 |
| Z(SUS) | empresa + año | −0,004 | 0,033 | 0,90 | 0,93 | 0,097 | 0,63 |

- **t+2** (N = 114): ESGSI +0,019 / +0,018 / −0,036 (p wild 0,55 / 0,54 / 0,47) y extendido +0,021 / +0,021 / −0,016.
  El MDE va de 0,069 a 0,144.
- **Like-for-like** (t+1): ESGSI +0,014 / +0,015 / +0,009 (p wild 0,66 / 0,57 / 0,73).
- **QUANT y HEDGE**: el único p agrupado < 0,05 de la muestra principal es Z(QUANT) con EF de empresa y año (−0,041,
  p 0,006 en t+1; −0,037, p 0,009 en like-for-like). El wild bootstrap no lo confirma (p 0,23 y 0,27): lo mueven
  pocas empresas. Es lo esperable entre 54 contrastes.
- **Todas las cubiertas** (G = 23, N = 160, con las 4 de emisiones mínimas): igual de nulo (ESGSI con EF de año −0,006,
  p wild 0,90). El MDE sube a 0,10-0,19 por el ruido de las emisoras pequeñas.
- **Resumen**: 0 de 54 contrastes con p wild < 0,05 en la muestra principal (2 con p agrupado < 0,05). Los signos de
  ESGSI y ESGSI extendido son positivos con EF de año, lo que iría en la dirección de "más tono relativo, menos
  reducción posterior", pero son pequeños (2-3 % por DT) y caen a cero dentro de cada empresa.

## 5. Resultados B: ¿la sustancia escrita sigue las emisiones? (`correlaciones_ets.csv`)

Se usan la correlación de Spearman entre medias de empresa (19 empresas; 18 con ingresos) y la correlación parcial
tras efectos fijos, con p agrupado y wild. Nivel = log E(t). Intensidad = log(t CO2e / M€ de ingresos).

| Score | vs. | ρ medias empresa (p) | r parcial, EF año (p wild) | r parcial, EF año + industria (p wild) | r parcial, EF empresa + año (p wild) |
|---|---|---|---|---|---|
| densidad SUS | nivel | +0,28 (0,24) | +0,17 (0,31) | −0,26 (0,31) | +0,19 (0,15) |
| densidad SUS | intensidad | +0,31 (0,21) | +0,35 (0,07) | −0,07 (0,73) | +0,27 (0,19) |
| ESGSI | nivel | −0,16 (0,51) | −0,14 (0,51) | +0,25 (0,18) | −0,16 (0,30) |
| ESGSI | intensidad | −0,24 (0,33) | −0,32 (0,23) | +0,02 (0,89) | −0,30 (0,26) |
| ESGSI ext. | intensidad | −0,35 (0,16) | −0,51 (0,05) | −0,07 (0,60) | −0,31 (0,26) |
| Z(QUANT) | nivel | +0,48 (0,04) | +0,46 (0,01) | +0,18 (0,15) | +0,24 (0,05) |
| Z(HEDGE) | intensidad | −0,53 (0,03) | −0,53 (0,01) | −0,22 (0,27) | −0,12 (0,67) |

Lectura:

1. **Entre empresas**, las más emisoras escriben informes más cuantificados y con menos cautelas. Por eso tienen un
   ESGSI extendido más bajo. La densidad SUS apunta en la misma dirección, pero sin significación.
2. **Dentro de cada industria** desaparece todo. La asociación de partida es composición sectorial: utilities, energía
   y química escriben de otra manera que bienes de consumo o servicios. Es coherente con la heterogeneidad por
   industria que ya documenta el paper.
3. Nada de esto valida ni invalida el ESGSI. Un nivel de emisiones alto no es poca sustancia. La densidad SUS mide
   cuánto vocabulario ESG sustantivo usa el texto, no el desempeño. Es esperable que crezca con la materialidad
   climática del sector.

## 6. Potencia

- MDE al 80 % (α = 0,05, bilateral) = (t₀,₉₇₅;G−1 + t₀,₈₀;G−1) × SE, con el SE agrupado observado.
- Con t+1 y EF de año: **0,058 log-puntos por DT** de ESGSI, que es el **37 % de la DT** de Δlog E. Con EF de empresa
  y año: **0,134**, el 86 % de la DT.
- Para dar una idea de escala: la media de Δlog E(t → t+1) de la muestra es de −0,07 anual. Detectar un efecto equivale a que una
  DT más de ESGSI cambie la variación anual de emisiones en 6-13 puntos porcentuales. Cualquier mecanismo realista
  (decisiones de inversión que se reflejan con años de retraso, más el peso de la demanda y los precios) daría como
  mucho 1-2 puntos. **La prueba no tiene potencia para efectos de tamaño plausible**, y el resultado nulo no es
  informativo más allá de descartar asociaciones muy grandes.
- La potencia no mejora mucho ampliando la ventana: el ESGSI solo existe para 2018-2024. Ampliar a t+3 resta
  observaciones. La palanca sería N: muchas más empresas emisoras, por ejemplo todo el STOXX Europe 600 con su
  informe anual.

## 7. Valoración: ¿merece entrar en el paper?

**Como validación externa central, no.** Hay cuatro motivos.

1. **Desajuste de constructo.** El ESGSI mide tono relativo a sustancia en el texto. Las emisiones del ETS miden una
   parte del alcance 1 europeo, cuya evolución dominan la demanda, los precios del gas y del CO2 y el cierre del
   carbón. No es el "walk" que corresponde a todo lo que dice un informe ESG, que cubre los tres pilares.
2. **Ámbito.** Cubre 19 de 49 empresas, y en 12 de esas 19 (automoción, bienes de consumo, farmacia y aeroespacial) el ETS es solo la sala de calderas.
3. **Perímetro.** La estructura de grupo es la actual (GLEIF) o la que se reconstruye por nombre. No hay un
   perímetro de consolidación histórico gratuito.
4. **Potencia.** El MDE va de 0,37 a 0,86 DT del resultado.

**Como nota breve, sí puede aportar.** Por ejemplo, una o dos frases en limitaciones o en la agenda de investigación:
*"A pilot linking 19 emitting firms to their EU ETS verified emissions (2018–2025) finds no association between the
score and subsequent emission changes (minimum detectable effect 0.06–0.13 log points per SD), and cross-firm
correlations between the components and emission intensity vanish within industry."* Todas las cifras salen de
scripts versionados, no se cita ninguna empresa y no hay umbrales.

**Si se quiere un criterio externo creíble y gratuito**, la vía con más potencial sería ampliar N a cientos de
emisores europeos con informe anual, que pide extender el pipeline, o usar como resultado la **discrepancia entre las
emisiones de alcance 1 que el informe declara y las verificadas en el ETS** de las mismas instalaciones. Esto último
mide directamente si lo que se dice casa con lo que se verifica, pero exige extraer cifras del texto.

## 8. Reproducción

```bash
# 1) Descargas (sin credenciales), GLEIF, ingresos y manifiesto
python -I scripts/ets_fetch.py
# 2) Cruce instalación -> empresa, panel y cobertura
python -I scripts/ets_match.py --results-dir <clon principal>/results
# 3) Modelos, correlaciones y potencia (9.999 réplicas wild; ~1-2 min)
python -I scripts/ets_models.py --results-dir <clon principal>/results
```

Entradas internas no versionadas (con nombres): `data/issue2_ets/ets_firms_internal.csv` (ISIN, patrones y exclusiones
por empresa) y `data/issue2_ets/ets_review_internal.csv` (los 32 descartes manuales, con su motivo).

## 9. Ficheros

| Fichero | Contenido |
|---|---|
| `fuentes_ets.json` | manifiesto de descargas: URL, fecha, bytes, SHA-256 y golden copy de GLEIF |
| `cobertura_ets.csv` | por `firm_id`: industria, país, instalaciones fijas y de aviación por vía de cruce, kt CO2e en 2018 y 2024 |
| `cobertura_ets_resumen.json` | totales y cobertura por industria |
| `panel_ets.csv` | `firm_id` × año (2013-2025): emisiones fijas, instalaciones activas, aviación y Δlog like-for-like |
| `muestra_ets.csv` | descriptivos por `firm_id` y marca de muestra material |
| `modelos_ets.csv` | 108 modelos: 2 muestras × 3 resultados × 6 scores × 3 especificaciones (β, SE, p agrupado, p wild, MDE) |
| `correlaciones_ets.csv` | Spearman de medias de empresa y correlaciones parciales con EF (nivel e intensidad) |
| `modelos_ets_resumen.json` | N, G, parte de la varianza dentro de cada empresa, p mínimos y rango del MDE |
