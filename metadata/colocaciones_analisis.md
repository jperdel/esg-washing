# `compliance` y `social`: análisis de colocaciones — PROPUESTA

Documento de trabajo, **no configuración activa**. Base empírica para decidir
cómo incorporar los dos términos que concentran un tercio de la masa de la
lista de 147 candidatos.

Medido sobre los 343 informes completos (49 empresas, 7 países, 2018-2024),
sobre **texto íntegro de PDF**, no sobre las zonas ya extraídas. Eso importa
para leer las cifras: la extracción por zonas ya descarta buena parte de las
notas contables, así que la tasa de falsos positivos dentro de nuestro
pipeline será **menor** que la que aparece aquí.

---

## 1. `social` — 48.198 ocurrencias

| contexto | ocurrencias | % |
|---|---:|---:|
| ESG claro | 29.041 | 60,3 % |
| mixto | 5.119 | 10,6 % |
| sin marcadores | 9.813 | 20,4 % |
| **no-ESG: nómina/fiscal** | 2.359 | 4,9 % |
| **no-ESG: comunicación** | 1.071 | 2,2 % |
| **no-ESG: contable/auditoría** | 795 | 1,6 % |

**Menos del 9 % es claramente ajeno al ESG**, y se concentra en tres usos
identificables. Por eso aquí conviene una **lista de exclusión**, no una de
inclusión: con cuatro patrones se recupera el ~91 % restante sin tener que
enumerar una cola larguísima de colocaciones válidas.

### Excluir

```
social security          3.032    cotizaciones y nóminas
social media               890    marketing y comunicación
social network(s)          254    marketing y comunicación
social charge(s)           ~200   "salaries and social charges", contabilidad
```

### Aceptar — las válidas más frecuentes

Se listan para documentar la decisión; con el enfoque de exclusión entran
todas automáticamente.

**Marco y desempeño**
`corporate social responsibility` 4.492 · `social responsibility` 5.489 ·
`social performance` 1.216 · `social impact(s)` 1.561 ·
`social information` 1.172 · `social indicators` 264 ·
`social commitment` 215 · `social sustainability` 265

**Relaciones laborales**
`social dialogue` 1.813 · `social partners` 239 · `social protection` 666 ·
`social standards` 364 · `social audits` 382 · `social compliance` 224

**Riesgo e impacto**
`social risk(s)` 850 · `social issues` 631 · `social challenges` 246 ·
`social inclusion` 340 · `social development` 285 · `social enterprises` 223

**Coordinaciones** (el patrón más frecuente de todos)
`environmental and social` / `social and environmental` ~4.500 ·
`economic and social` 1.083 · `social and governance` 371 ·
`social and societal` 956 · `social and civic` 340

---

## 2. `compliance` — 55.780 ocurrencias

| contexto | ocurrencias | % |
|---|---:|---:|
| ESG claro | 23.306 | 41,8 % |
| mixto | 2.129 | 3,8 % |
| sin marcadores | 24.800 | 44,5 % |
| **no-ESG: contable/auditoría** | 4.874 | 8,7 % |
| no-ESG: otros | 671 | 1,2 % |

Caso distinto: el bloque **sin marcadores es el 44,5 %**, mucho mayor que en
`social`. Son menciones genéricas —"in compliance with the law", "compliance
function"— que no traen contexto suficiente para decidir. Por eso aquí sí
conviene una **lista de inclusión**: aceptar solo las colocaciones que
identifican gobernanza, y dejar fuera el resto.

### Aceptar — gobernanza y ética

**Estructura y función**
`compliance officer(s)` 1.830 · `compliance committee` 2.235 ·
`compliance function` 702 · `compliance department` 696 ·
`compliance management system` ~1.400 · `chief compliance` 937

**Programa y control**
`compliance programme/program(s)` 1.764 · `compliance training` 404 ·
`compliance risk(s)` 2.159 · `compliance violations` 231 ·
`non-compliance` 2.823 · `regulatory compliance` 853

**Vínculo explícito con ética o ESG**
`ethics and compliance` / `compliance and ethics` ~1.600 ·
`compliance with human rights` 163 · `compliance with the code of conduct` ~900 ·
`social compliance` 224 · `tax compliance` 357 ·
`corruption` en ventana con `compliance` 249

### Excluir — contable y auditoría

```
compliance with [German] generally accepted accounting principles   207
compliance with the engagement (letter)                             196
compliance with the independence requirements                       123
compliance with IFRS / accounting standards
```

Son el núcleo del 8,7 % contable, y proceden casi siempre del informe del
auditor y de las notas a los estados financieros.

---

## 3. Resumen de la decisión

| | enfoque recomendado | por qué |
|---|---|---|
| `social` | **exclusión** (4 patrones) | solo el 8,7 % es ajeno y está muy localizado; una lista de inclusión pierde la cola larga de usos válidos |
| `compliance` | **inclusión** (~15 colocaciones) | el 44,5 % es genérico sin contexto; aceptarlo entero metería ruido de gobernanza indiferenciada |

### Lo que finalmente se implementó, y en qué se aparta de esto

`compliance` va por inclusión, como se recomienda. **`social` también va por
inclusión** —ocho colocaciones en `esg_terms.txt`— y no por exclusión.

El motivo es de arquitectura, no de criterio: el vocabulario es una lista de
literales que alimenta a la vez un regex sobre texto crudo y un
`CountVectorizer` sobre texto lematizado, y **ninguna entrada de un vocabulario
TF-IDF puede llevar una exclusión**. Implementar la recomendación exigiría
neutralizar las tres colocaciones ajenas en el preprocesado, antes de
lematizar, lo que altera el texto de entrada de todos los componentes.

El coste es real y conviene tenerlo escrito: se pierde la cola larga de usos
válidos de `social`, que según este mismo análisis es la mayor parte de sus
casi 50.000 ocurrencias. La ventaja colateral es que desaparece el riesgo de
doble conteo con las siete entradas `social ...` que ya existían.

Si en algún momento se quiere la exclusión, hay que hacerla en el
preprocesado y **rehacer todas las cifras**: el recuento de 289 entradas y
todos los resultados de robustez cambiarían.

**Aviso sobre doble conteo.** El vocabulario actual ya contiene `csr`,
`social impact`, `social responsibility`, `social compliance`,
`social commitment`, `social standard` y `social norm`. Si `social` entra por
exclusión, esas siete entradas pasan a ser redundantes y hay que retirarlas o
se contarán dos veces.

**Pendiente.** Estas cifras clasifican por vocabulario circundante, que es un
proxy. La validación contextual por muestreo que propone Cowork sigue siendo
la forma de medir la precisión real de cada colocación; este análisis sirve
para decidir *qué* validar y acota el problema, no lo sustituye.
