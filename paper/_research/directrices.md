# Directrices del autor (2026-09-26) — prevalecen sobre cualquier otro fichero

Vinculantes para todo el paper. Si `conventions.md` o `v3_brief.md` dicen otra
cosa, manda este fichero.

## 1. Sin etiquetas: el índice es un score

- Nunca se clasifica un informe como "ESG-washing" / "genuine", ni
  "flagged", ni "señalado". No se usa el umbral ESGSI > 0 ni ningún otro.
- El índice es un score continuo y relativo al corpus (z-scores). Su signo
  solo indica posición respecto a la media del corpus; no se interpreta.
- Fuera: recuentos de informes señalados, matrices de transición entre
  etiquetas, "años señalado" por empresa, anotaciones de señalados en figuras.
- La crítica al umbral de Lagasio puede aparecer como motivo para no usarlo,
  no como análisis central.

## 2. Solo el vocabulario actual (v3)

- Ninguna comparación de resultados con vocabularios anteriores (154, 289,
  289+22, ni la ejecución previa del v3).
- Estabilidad: solo perturbaciones del v3 (borrados aleatorios, por pilar,
  sectoriales, término a término, términos con aviso).
- Auditoría de precisión (KWIC): FUERA del paper. El vocabulario se presenta
  con sus fuentes, criterios y composición, sin cifras de precisión.
- Construcción del v3: proceso y criterios (fuentes de candidatos, reglas de
  admisión, colocaciones, patrones, pilares, sectoriales, frases protegidas),
  sin historial: ni lista de partida, ni tablas de retirados, ni antes/después.

## 3. Enfoque respecto a Lagasio (2024)

- Paper propio. Lagasio es el antecedente del que se toma la arquitectura.
- Su crítica se reduce a una subsección que justifica nuestras decisiones
  (densidad en vez de TF-IDF con normalización L2; sin umbral; z-score).

## 4. Índices

- **ESGSI**: el índice original de Lagasio con nuestras mejoras,
  Z(SEN) − Z(SUS) con SUS por densidad. Es el principal.
- **Extended ESGSI** (ESGSI_ext): la métrica nueva que propone el paper,
  con QUANT y HEDGE. Se presenta como contribución, reconociendo que los
  pesos (0,5) son una elección y mostrando su sensibilidad.

## 5. Robustez: medidas de score, no de etiquetas

- Correlación de Spearman con el score de referencia.
- Solapamiento de los deciles extremos (10 % superior e inferior).
- Pendiente temporal (efectos fijos de empresa, SE agrupados).
- Nada de "etiquetas que cambian" ni escalados basados en ellas.

## 6. Sectoriales

- Dentro del vocabulario principal; sin ellos, como sensibilidad.

## 7. Empresas

- Sin nombres de empresas en texto, tablas ni figuras. Resultados por sector
  (industria/supersector) y país; las empresas, como posiciones anónimas.

## 8. Heredado que se mantiene

- LDA fuera. Inferencia con SE agrupados por empresa y wild cluster
  bootstrap. Todas las cifras, de scripts versionados.
- Marca en §5 para identificar a los revisores del consenso de vocabulario
  (si se sigue describiendo el consenso).
