# Pendientes para el autor (2026-09-26, tras adaptar el paper a las directrices)

El paper sigue `paper/_research/directrices.md`: score continuo sin etiquetas,
solo el vocabulario actual, sin auditoría, sin nombres de empresa, Lagasio como
antecedente, ESGSI + Extended ESGSI. Compila limpio (55 páginas). Nada está
commiteado.

## Decisiones y tareas para el autor

1. **§5, revisores.** Marca visible `[Author: identify the reviewers of the
   candidate lists.]`: quién revisó las listas de candidatos.
2. **Resumen, introducción larga, literatura, discusión y conclusión.** Siguen
   comentados en `main.tex` (placeholders P1–P5).
3. **Título.** Propuesto: "Tone Relative to Substance in European ESG
   Disclosure: A Continuous Corpus-Relative Score, 2018–2024". Revisar.
4. **§12** repite como limitaciones algunas cifras de HEDGE y de la rejilla de
   pesos, con referencia a §5 y §10. Coherente; recortar es opcional.
5. **Calibración del filtro de navegación.** Sus cifras se han quitado del
   paper porque no hay script versionado que las reproduzca. Si se quieren,
   hay que versionar ese cálculo.
6. **Decisiones de vocabulario abiertas** (declaradas en §5): pilar de la
   familia *taxonomy*; umbrales de concentración de los sectoriales no están
   en ningún script.
7. **`scripts/build_kwic_sample.py`** contiene las constantes de dispersión
   (≥ 50 disparos en ≥ 5 empresas) que §5 cita; su nombre remite a la
   auditoría retirada. Se podría mover esas constantes a otro sitio.
8. **Commit.** Hay cambios del pipeline (frases protegidas, HEDGE del maestro
   L&M sobre texto crudo, NUL) y scripts nuevos sin commitear; §13 lo declara.
   Las columnas `Etiqueta`/`Etiqueta_ext` que escribe `main.py` no se usan en
   el paper; se pueden quitar del pipeline.

## Resultados principales (para orientarse)

- ESGSI: tendencia −0,181/año (efectos fijos, p agrupado 3,7·10⁻⁶, wild
  bootstrap < 0,0001). Extended ESGSI: −0,260/año.
- La caída la lleva la sustancia; el tono (SEN) no tiene tendencia.
- Sin nombres de régimen −0,149; sin el pilar transversal −0,117. La
  confusión con los mandatos regulatorios queda sin resolver.
- Las empresas pesan más que los años (≈67 % frente a ≈7 % de la varianza;
  ICC 0,676). Industria y país explican poco entre empresas.
- Rejilla de pesos del extended: ρ ≥ 0,865 con el de 0,5/0,5 en toda la
  rejilla; el orden depende sobre todo de w_h, la pendiente de w_q.
- Estabilidad del vocabulario: borrados aleatorios del 10–50 % mantienen ρ
  0,99–0,92 y la pendiente negativa en todos los sorteos menos uno; quitar el
  pilar ambiental es lo que más mueve el score.
