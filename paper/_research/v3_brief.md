# Brief: moving the paper onto the v3 vocabulary (2026-09-25)

Binding on every agent working on this round. Read it in full, then read
`paper/_research/conventions.md` (style rules §1–4, §8–10 remain binding; its
**numbers** are the old 154-term basis and are superseded by the files listed
below).

## What changed

The paper was drafted on a 154-term vocabulary (`results/metrics/results.csv`).
The project has since built and audited a new vocabulary, **v3**, which is now
the adopted specification:

- Source of truth: `metadata/ESG_terms_v3.csv` (434 entries: 402 terms + 32
  regex patterns; pillars E 168 / S 139 / G 95 / TRANS 32; 36 flagged
  sectoral). Per-decision log: `metadata/ESG_terms_v3_decisiones.csv`.
  Pillar map of the pre-v3 list: `metadata/esg_terms_pilares.csv`.
- Exported pipeline files: `metadata/esg_terms.txt`, `esg_terms_sectorial.txt`,
  `esg_patterns.txt`, `esg_patterns_sectorial.txt`; lemmatised by
  `scripts/build_lexicons.py` into `esg_terms_lemmatized.txt` and
  `esg_terms_sectorial_lemmatized.txt`.
- How v3 was built (read the scripts' docstrings and the reports):
  `scripts/build_consensus*.py`, `build_kwic_sample.py`, `build_vocab_v3.py`,
  `build_vocab_v3_report.py`, `export_vocab_v3.py`; working files under
  `paper/_research/kwic*/`, `term_mass*.csv`, `vocab_v3_resumen.json`,
  `revision_*.csv`, `ronda2_*.csv`, `terminos_ronda2.txt`.
- Spanish working reports (not versioned, but readable): `informes/ESG_vocabulario_v3.md`,
  `informes/estabilidad_vocabulario_v3.md`, `informes/arquitectura_lexicos.md`,
  `informes/colocaciones_analisis.md`, `informes/ESG_deep_consensus*.md`,
  `informes/ESG_terms_final.md`.
- Extractor change: table-of-contents / cross-reference paragraphs are now
  dropped (`_is_navigation` in `src/lexical_document_filter.py`); regex patterns
  moved from code to `metadata/esg_patterns*.txt`.
- Stability tests of the index against the vocabulary: `scripts/stability_v3.py`
  (frozen extraction) and `scripts/stability_v3_e2e.py` (simulated
  re-extraction); outputs in `results/stability_v3/` (`summary.json`,
  `e2e_summary.json`, `borrado_aleatorio.csv`, `e2e_borrado_aleatorio.csv`,
  `quitar_uno.csv`, `variantes.csv`).

## Adopted specification for this round

- Vocabulary: **v3 including the sectoral entries** (`ESG_RUN_TAG=v3`,
  `ESG_INCLUDE_SECTORAL=1`). Without-sectorals is a sensitivity check.
- Substance: `SUS_MODE='density'` (\susdens), as before. The three SUS
  specifications are still all computed and reported for robustness.
- Index, labels, z-score (population sd), extended weights: unchanged.
- `board of directors` is kept in the vocabulary; it is the single entry that
  moves the most labels (27 in leave-one-out) and its lemmatised form
  (`board director`) over-counts relative to the audit (59,912 vs 32,446).
  Report this as a disclosed sensitivity, do not silently drop the term.
- Pipeline outputs of the adopted run (being regenerated on 2026-09-25):
  `data/chunks_lexical_v3_sec/` (extraction JSONs), `data/clean_v3_sec/processed_texts.csv`,
  `results/metrics_v3_sec/results.csv` (343 rows). The previous run of the
  same spec is backed up at `results/_prev_v3_sec/results.csv`.
- Firm → sector: `metadata/empresas_supersector.csv` (`;`-separated, UTF-8;
  columns Empresa, País, Supersector STOXX/ICB). `Empresa` equals the folder
  name, i.e. the `Compañía` column of results.csv, exactly. 16 supersectors,
  many with 1–2 firms; group into ICB industries where useful.

## Environment

- Python: `C:/Users/Jorge/anaconda3/envs/esgwashing/python.exe` (the base
  anaconda python is 3.8 and cannot import `src/config.py`). statsmodels,
  scipy, matplotlib, pandas, sklearn, spacy are installed there.
- Run repo scripts with `PYTHONIOENCODING=utf-8` and, when they import
  `src/config.py`, `ESG_RUN_TAG=v3 ESG_INCLUDE_SECTORAL=1`.
- LaTeX: `pdflatex`/`latexmk` (MiKTeX) available. Paper: `paper/main.tex`,
  sections in `paper/sections/`, bib in `paper/references.bib`
  (style `elsarticle-harv`).
- Do NOT run `main.py`, do NOT touch `data/`, and do not commit anything.

## Scope of the rewrite

Rewrite ONLY: §3 data (`03_data.tex`), §4 extraction (`04_extraction.tex`),
§5 vocabulary selection (`05_lexicon.tex`), §6 index specification
(`06_index_specification.tex`), §10 robustness (`10_robustness.tex`),
§11 results (`11_temporal_results.tex`, broadened to temporal + cross-sectional
results by firm, sector, country). Leave §1, §2, §7, §8, §9, §12, §13 untouched
even though they carry 154-term numbers; keep every `\label` those sections
reference (`sec:*`, `tab:*`, `eq:*`) alive or the build breaks.

## Evidence discipline (carried over, still binding)

- Every number in a section must come from a file in the repo (analysis
  outputs, metadata, results, stability outputs) or be recomputed by a script
  committed under `scripts/`. No throwaway computations for published numbers.
- Counts always with denominator (e.g. 160/343).
- p-values: firm-clustered standard errors are now available and must be
  used for every regression on the panel; document-level OLS may be shown
  only alongside and labelled as such.
- Country/document-type confound (§3) still holds; sector cuts across
  countries, which is what makes it partially informative — but sector cells
  are small (1–7 firms). Say so wherever sector results are read.
- The temporal finding is not the paper's headline (conventions §8).

## Correction and analysis layer (added during the round)

- The 154-term paper run is `results/metrics_baseline_paper/results.csv`
  (171 flagged). `results/metrics/results.csv` is the 289-term run (164
  flagged); `results/metrics_sec/` is 289+22 sectoral.
- Analysis layer: `scripts/analysis_v3.py` → `results/analysis_v3/`
  (`summary.json` holds every quotable scalar; CSVs; booktabs tabular
  fragments in `tex/`, tabular only — caption/label go in the section). `scripts/figures_v3.py` → `paper/figures/*.pdf`.
  `scripts/vocab_v3_stats.py` → `results/analysis_v3/vocab_stats.json`.
- `results/` is git-ignored, so the paper must not \input files from it.
  Transcribe table values into the .tex by hand from the fragments, exactly.
- 11 v3 terms never reach the scoring vocabulary (lemmatisation collapses or
  empties them, e.g. well-being → empty, iso 14001 → iso); listed in
  summary.json `vocabulario.terminos_v3_fuera_del_vocabulario_tfidf`.
  Lemmatised scoring vocabulary V = 390; longest entry 4 tokens → ngram (1,4).

## Round 2 fixes (2026-09-25 evening) — supersede anything above

1. **Protected phrases.** 11 vocabulary terms used to vanish or collapse under
   preprocessing (well-being → empty; iso 14001 → iso; say on pay → pay; speak
   up, first aid, b corp, cop 28, 2030 agenda, iso 37001/45001/50001) and were
   excluded from the scoring vocabulary. `scripts/build_lexicons.py` now writes
   `metadata/esg_protected_phrases.txt` (regex → single token) and
   `TextProcessor` substitutes them before tokenising, for the corpus and the
   vocabulary alike. Scoring vocabulary: 365 base + 34 sectoral lemmatised
   entries (was 358 + 32); recompute V from the files. Preprocessing and the
   index were re-run (extraction unchanged: the extractor regex did not change).
2. **HEDGE on raw text.** The lemmatised text is stopword-filtered, so the
   modals (may, might, could, must, will) and always/never/perhaps could never
   count. `calculate_hedge_scores` now takes the raw extracted text, tokenised
   as lowercase letter runs with internal hyphens, and divides by that word
   count. Only HEDGE and ESGSI_ext change. Known residual: "may" as a month.
3. **`board of directors` is NOT over-counted.** The lemma `board director`
   (59,912) matches the exact phrase on the v3 extracted raw text (58,985
   `board(s) of director(s)`); the audit figure 32,446 was measured on the
   pre-v3 extraction, where the phrase occurs 33,284 times. The gap is the
   extraction feedback: admitting the term pulled ~26k further occurrences
   (governance paragraphs) into the zones. Every sentence in the paper that
   calls it a lemmatisation over-count is wrong and must be corrected; the
   27-label leave-one-out sensitivity stands (recompute on the new run).
4. The reviewer-identity placeholder in §5 stays as is (author will fill it).
5. **HEDGE lexicon rebuilt (supersedes item 2's lexicon details).**
   `metadata/RAW_LM_dictionary.csv` is an expanded, non-master list
   (Uncertainty 767 vs 297; 896 underscore phrases). HEDGE now uses the
   Loughran–McDonald master dictionary shipped with pysentiment2 (same source
   as SEN): Uncertainty 297 ∪ Constraining 184 ∪ WeakModal 27 ∪ StrongModal 19
   = 495 original forms → `metadata/lm_hedge.txt`, matched on raw text.
   Index re-run: ESGSI/SUS/SEN/QUANT bit-identical; HEDGE mean 0.0187;
   ESGSI_ext 167/343, 58 labels differ from ESGSI. Numbers: vocab_stats.json
   key `hedge`, summary.json.
6. Final adopted run: 159/343 flagged; V = 399 (365 + 34). Leave-one-term-out
   "entries moving no label" (275/399) is fragile: it depends on how close the
   nearest documents sit to ESGSI = 0.
