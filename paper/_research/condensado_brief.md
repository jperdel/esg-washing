# Brief: condensed paper (2026-09-29)

Binding: `paper/_research/directrices.md` — especially §9 (no code references,
no version history, each stage explained once, brief vocabulary, page budget).
Style: `paper/_research/conventions.md` §1–3 and §10 (English, technical,
first-person plural, booktabs, "vector normalisation" vs "score
normalisation" always qualified). Macros in `paper/main.tex`.

## Target structure (technical core ≤ 15 pages, 11pt A4, 2.5 cm margins)

| § | File | Title | Budget | Floats |
|---|---|---|---|---|
| 1 | `01_scope.tex` | Scope and contribution | 1 p | none |
| 2 | `03_data.tex` | Data | 1 p | 1 table |
| 3 | `04_method.tex` (new) | Method | 5 p | 2 tables |
| 4 | `11_temporal_results.tex` | Results | 5 p | 3 figures + 2 tables |
| 5 | `10_robustness.tex` | Robustness | 2 p | 1 figure + 1 table |
| 6 | `12_limitations.tex` | Limitations | 0.75 p | none |
| — | end of §6 | Data and code availability (2–3 lines) | | |

Order in `main.tex`: 01, 03, 04_method, 11 (Results), 10 (Robustness), 12.
Removed from main.tex and deleted: `04_extraction.tex`, `05_lexicon.tex`,
`06_index_specification.tex`, `13_reproducibility.tex` (their content is
condensed into `04_method.tex`).

Figures kept: `figures/fig_temporal.pdf`, `fig_industry_year.pdf`,
`fig_pillars.pdf` (Results); `fig_stability.pdf` (Robustness). All others are
dropped from the paper (their message, if needed, becomes one sentence).

## Label contract (use exactly these; do not invent other section labels)

Sections: `sec:scope`, `sec:data`, `sec:method`, `sec:method-extraction`,
`sec:method-vocab`, `sec:method-index`, `sec:method-choices`,
`sec:method-inference`, `sec:results`, `sec:results-time`,
`sec:results-mandate`, `sec:results-groups`, `sec:results-pillars`,
`sec:robustness`, `sec:limitations`.
Equations: `eq:esgsi`, `eq:esgsiext` (defined in §3 only).
Figures: `fig:results-time`, `fig:results-industry`, `fig:results-pillars`,
`fig:robustness-stability`.
Tables: prefix with the section: `tab:data-*`, `tab:method-*`,
`tab:results-*`, `tab:robustness-*`.

## Content ownership (each thing lives once)

- §3 Method defines: extraction (zones, threshold, navigation filter, yield in
  one sentence), vocabulary (sources, admission criteria, pillars, sectoral
  entries, composition in one small table, precision sentence ≈ 82 % estimated,
  stability → §5), preprocessing incl. protected phrases (one sentence),
  components SUS (density) / SEN / QUANT / HEDGE / BREADTH, ESGSI (eq:esgsi)
  and Extended ESGSI (eq:esgsiext, weights 0.5 as a choice → §5), why density,
  z-scores and no threshold (≤ 1 page, one table: L2 invariance + convergent
  validity with QUANT), inference (FE, firm-clustered SE, wild bootstrap).
- §4 Results and §5 Robustness never redefine anything: they cite §3 with \ref.
- Sources of every number: `results/analysis_v3/*.json|csv` (never *_internal*),
  `results/stability_v3/*`, `paper/_research/vocab_v3_resumen.json` (only for
  the 0.821 precision figure). Transcribe; never \input from results/.
- Keep the author marker `\textbf{[Author: identify the reviewers of the
  candidate lists.]}` in §3's vocabulary paragraph.
