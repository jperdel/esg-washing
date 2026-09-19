# Procedencia real de la muestra de revisión

`muestra_revision.txt` lleva **tres bloques mal etiquetados**. El texto es real y
procede del corpus, pero la cabecera de esos tres bloques nombra a otra empresa.

## Causa

El script de muestreo indexó los ficheros por **nombre** en vez de por ruta:

```python
paths = {p.stem: p for p in Path('data/chunks_lexical').rglob('*.json')}
```

133 de los 344 informes comparten nombre con otro (`2018 ANNUAL REPORT` lo usan
diez empresas), así que el diccionario se quedó con uno arbitrario de cada grupo.
Corregido en el script: hay que indexar por `país/empresa/fichero`.

## Las tres correcciones

| cabecera en el fichero | contenido real |
|---|---|
| `BAYER \| ALEMANIA \| 2019 \| Annual report` | **Ahold Delhaize**, Países Bajos, distribución alimentaria |
| `BAYER \| ALEMANIA \| 2023 \| Annual report` | **Ahold Delhaize**, Países Bajos, distribución alimentaria |
| `SAFRAN \| FRANCIA \| 2019 \| URD` | **TOTAL**, Francia, petróleo y gas |

Los otros 14 bloques son correctos.

## Composición real

**13 empresas, 7 países**: Deutsche Börse (DE), AB InBev (BE), Iberdrola ×2 (ES),
Nokia (FI), AXA ×2, Sanofi, TOTAL, Airbus (FR), Ferrari, Enel (IT), Ahold ×2,
Stellantis ×2, ASML (NL).

Sectores cubiertos: mercados de capitales, consumo, utilities eléctricas,
telecomunicaciones, seguros, farmacia, petróleo y gas, aeronáutica, automoción,
distribución alimentaria, semiconductores.

## Qué queda afectado

- **La extracción de vocabulario NO se ve afectada**: el texto es real y los
  términos que contiene son los que son.
- **La atribución por empresa sí**: cualquier afirmación del tipo "este término
  solo aparece en Bayer" es inválida para esos tres bloques.
- **Alemania está infrarrepresentada**: una sola empresa, no tres como pretendía
  la estratificación. Farmacia queda cubierta solo por Sanofi.
