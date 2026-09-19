# Shared conventions — binding on every section

Read this before writing. Every number here is verified against
`results/metrics/results.csv` (n = 343) or against the source PDF. Do not
re-derive them, do not round them differently, and do not introduce a number
that is not here or in `implementation.md` / `lagasio_source.md`.

If you believe a number here is wrong, say so in your report-back. Do not
silently substitute your own.

---

## 1. Language and voice

- **English.** Academic register, first person plural ("we").
- Past tense for what we did, present for what the data show.
- No rhetorical build-up, no motivational prose, no ESG-as-a-topic commentary.
  These sections are technical. Someone else writes the framing.

## 2. The terminology split — MANDATORY

Two different operations are both called "normalisation" in this literature.
Conflating them destroys the two arguments that matter most. Always qualify:

- **Vector normalisation** — L2 scaling of a document's term vector to unit
  length. Within-document, at the TF-IDF stage. This is §7's object.
- **Score normalisation** — min–max or z-score rescaling of the finished
  component scores across the sample. Cross-sectional, at index-assembly
  stage. This is §8's object.

Never write "normalisation" unqualified. Never let one section's usage leak
into the other's.

## 3. Notation

Use the macros defined in `main.tex`: `\SUS \SEN \QUANT \HEDGE \BREADTH
\ESGSI \ESGSIext \Z`, and for the three substance specifications
`\susdens \sustfidf \suslag`. Do not define your own.

Index definitions:

$$\ESGSI = \Z(\SEN) - \Z(\SUS) \qquad
\ESGSIext = \Z(\SEN) - \Z(\SUS) - w_q\Z(\QUANT) + w_h\Z(\HEDGE)$$

with $w_q = w_h = 0.5$ (arbitrary; §12 must say so).

## 4. Labels

`\label{sec:scope}`, `\label{sec:original}`, `\label{sec:data}`,
`\label{sec:extraction}`, `\label{sec:lexicon}`, `\label{sec:index}`,
`\label{sec:measurement}`, `\label{sec:artefact}`, `\label{sec:validity}`,
`\label{sec:robustness}`, `\label{sec:temporal}`, `\label{sec:limitations}`,
`\label{sec:repro}`.

Tables `\label{tab:<section>-<slug>}`, equations `\label{eq:<slug>}`.
Cross-reference with `\ref{}`; never write "the section above".

## 5. Corpus — canonical figures (n = 343)

- 343 unique reports, **49 companies × 7 years (2018–2024), exactly 49 per year** — a fully balanced panel.
- Document types: **235** annual reports, **101** URD, **7** standalone sustainability reports.
- Countries: Germany 98, France 119, Netherlands 42, Italy 35, Spain 28, Finland 14, Belgium 7.
- Document type is near-perfectly confounded with country: URD is 99/101 French; Germany is 98/98 annual reports. State this as a design property in §3, not as a buried limitation.
- 344 PDFs were processed; one is a duplicate (Wolters Kluwer 2019, a `_repaired` re-issue of the same report, byte-identical extracted text) and is removed by a content-hash deduplication step. **Never write 344 as the corpus size.**

## 6. The original study — verified facts only

Lagasio, V. (2024). *ESG-washing detection in corporate sustainability
reports.* International Review of Financial Analysis, 96(B), 103742.
DOI 10.1016/j.irfa.2024.103742. Open access CC BY 4.0, so direct quotation
is permitted — quote rather than paraphrase on every contested point.

- Corpus: 749 standalone sustainability reports, 2023 only, English only. Firms reporting ESG inside the annual report were **excluded**.
- $\SEN$: TextBlob polarity. $\SUS$: aggregate of scikit-learn `TfidfVectorizer` values over a GRI/SASB-derived keyword list.
- Disclosed TF-IDF settings: `max_df=0.95`, `min_df=2`, `max_features=10000`, unigrams+bigrams.
- **`norm` is never specified. The aggregation over the vocabulary is never stated.** The keyword list is never published and its size never given. Announced robustness checks (Cohen's Kappa, 5-fold CV, TF-IDF sensitivity) are never reported. Data availability statement: "No".
- §3.3 prose says min–max; the printed formula is a z-score difference. Table 2 (n=749) gives mean $\ESGSI$ $-0.2179$, and $0.3727 - 0.5906 = -0.2179$ exactly, with both components in $[0,1]$ — so the reported numbers are min–max.

**Evidence discipline.** Anything not in `lagasio_source.md` marked VERIFIED
does not go in the paper as fact. Where the original is silent, write that it
is silent — do not infer what "standard practice" would be. The one permitted
inference, and it must be labelled as an inference: the paper lists four
non-default `TfidfVectorizer` settings and no others, which implies the
remaining defaults were in force, including `norm='l2'`.

## 7. Results — canonical figures (n = 343)

Substance specifications, Pearson correlations:

| | dens | tfidf-ℓ | tfidf-L2 | breadth |
|---|---|---|---|---|
| dens | 1 | 0.9926 | 0.4177 | 0.4626 |
| tfidf-ℓ | 0.9926 | 1 | 0.4600 | 0.5118 |
| tfidf-L2 | 0.4177 | 0.4600 | 1 | **0.9733** |

Resulting index vs $\ESGSI(\susdens)$:

| spec | r | flagged | label flips |
|---|---|---|---|
| $\sustfidf$ | 0.9961 | 167/343 | 6 (1.7 %) |
| $\suslag$ | 0.6964 | 159/343 | 96 (28.0 %) |

> **CORRECTION, 2026-08-02.** An earlier revision of this block carried
> pre-deduplication ($n=344$) values under an $n=343$ heading. Two section
> authors caught it independently. Everything below has been **recomputed from
> scratch on the deduplicated corpus**. `implementation.md` was written before
> deduplication and its figures are $n=344$ throughout: where the two disagree,
> **this file wins**. Do not average, reconcile or split the difference — use
> these.

Convergent validity, Pearson (Spearman) with $\QUANT$: $\susdens$ **0.6663**
(0.7076), $\sustfidf$ **0.6817** (0.7220), $\suslag$ **0.3217** (0.4299).
Sublinear TF with length normalisation: **+0.0067**, i.e. no relationship.
Aggregated as a mean over the vocabulary instead, sublinear TF recovers
**+0.5262** — which is what localises the failure to length normalisation
rather than to damping as such.

Sublinear saturation: documents cover **87.75** of the 154 terms on average.
State the saturation claim as *the undamped numerator tracks the distinct-term
count*; once divided by length it correlates **−0.3214** with that count,
because length dominates. Do not quote a single correlation for "sublinear vs
distinct terms" without saying which of the two quantities is meant.

IDF inversion: terms with $\mathrm{idf} < 1.2$ (43 terms, document frequency
> 280) hold **78.35 %** of all ESG occurrences; 46 terms with
$\mathrm{idf} > 2$ hold **2.80 %**. *fair condition* has $\mathrm{idf}=4.8947$
on 8 corpus-wide occurrences, *climate* $\mathrm{idf}=1.0088$: one mention of
the former carries the weight of **4.852** of the latter.

L2 invariance demonstrations, both re-executed: (i) the same document with its
ESG term counts multiplied by 1, 2, 5, 20 yields an identical $\SUS$
(deviation $6.9\times10^{-18}$); (ii) appending **56,000** non-ESG filler
words to a 22,576-word document — 78,576 words total, a 3.48× dilution, ESG
mentions unchanged at 695 — drops density from **3.0785 to 0.8845** per 100
words while $\SUS$ stays at **0.0347763839**, identical to ten decimal places.

Score-normalisation factorial:

| score norm. | substance spec. | mean | flagged |
|---|---|---|---|
| z-score | $\susdens$ | +0.0000 | 171/343 (49.9 %) |
| z-score | $\suslag$ | +0.0000 | 159/343 (46.4 %) |
| min–max | $\susdens$ | −0.0004 | 166/343 (48.4 %) |
| **min–max** | **$\suslag$** | **−0.2388** | **54/343 (15.7 %)** |

Always print counts with the denominator, never a bare percentage: the whole
point of the section is that a denominator was mis-stated elsewhere.

**Z-score convention.** The pipeline standardises with the *population*
standard deviation (`np.std`, $\mathrm{ddof}=0$). Ad-hoc recomputation with a
sample standard deviation ($\mathrm{ddof}=1$, the pandas default) shifts the
third decimal. `results/metrics/results.csv` is authoritative for every index
value; do not recompute the series and print your own.

Temporal, $\ESGSI(\susdens)$ by year, read from `results.csv`: **0.563,
0.471, 0.161, 0.007, −0.121, −0.407, −0.674**. Slope −0.205/yr,
$p = 1.7\times10^{-8}$.
$\ESGSI(\suslag)$ by year: 0.561, 0.453, 0.106, −0.168, −0.274, −0.371,
−0.307 — note the 2024 rebound above 2023, which is why monotonicity is
claimed only for $\susdens$ and $\sustfidf$.
Drivers by year — extracted ESG text 27,906 → **30,038** (2019) → 35,340 →
40,846 → 45,872 → 50,190 → 63,654 tokens; density 4.87 → 7.45 per 100 words;
effective distinct terms 28.99 → 41.73.

Two distinct densities exist and must never be conflated. **Extraction
density** is 3.658 keywords per 100 words: the full regex, including the 24
morphological and numeric patterns, over *raw* text — an §4 quantity.
**Substance density** is the $\susdens$ measure over the 154-term vocabulary on
*lemmatised* text, whose corpus aggregate is 6.224 per 100 words and whose mean
of per-document values runs 4.87 → 7.45 by year. Different numerators,
different denominators, different sections.
$\SEN$ has **no** trend: slope +0.0016/yr, $p = 0.666$.
Germany only (98 docs, one country, one document type): slope −0.301/yr,
$p = 1.2\times10^{-6}$.

**Monotonicity:** the yearly series is monotone for $\susdens$ and
$\sustfidf$ but **not** for $\suslag$, where 2024 (−0.307) rebounds above
2023 (−0.370). Claim direction and significance for all three; claim
monotonicity only for our specification.

## 7b. Robustness checks run after the first review pass — SANCTIONED, use them

Two reviewers independently identified the same missing computation and called
it the paper's largest unforced concession. It has now been run. Both results
are canonical and both convert a concession into a reported robustness check.

**(A) Convergent validity on fully disjoint bases.** The 14 vocabulary entries
that also match a $\QUANT$ pattern — `co2, co2e, csrd, ghg, gri, issb, sasb,
sdg, sfdr, taxonomy, taxonomy align, taxonomy alignment, taxonomy eligible,
tcfd` — were removed from the vocabulary (154 → 140 terms) and $\QUANT$ was
restricted to its two purely numeric families, `percentages` and
`large_numbers`, dropping `frameworks` and `units` entirely. The restricted
$\QUANT$ correlates 0.8738 with the full one.

| specification | full bases | disjoint bases |
|---|---|---|
| $\susdens$ | 0.6663 | **0.4112** |
| $\sustfidf$ | 0.6817 | **0.4125** |
| $\suslag$ | 0.3217 | **0.1799** |
| ratio, length-normalised to $\suslag$ | 2.07× | **2.29×** |

Read it precisely: the overlap inflates all three correlations, so every
absolute figure on the contaminated bases is an upper bound — but the overlap
does **not** manufacture the ranking, and the relative distance is slightly
*larger* once it is removed. §9 must report this. It replaces the rescue
argument its reviewer refuted; do not reinstate that argument.

**(B) Temporal trend excluding all reporting-framework vocabulary.** §11's
reviewer showed the vocabulary contains terms naming regimes that did not
exist in 2018 (`csrd`, `esrs`, `issb`, `sfdr`, `tcfd`, the `taxonomy` family),
so part of the measured rise is chronology rather than conduct. Dropping the
whole 34-term *reporting frameworks and ratings* block (154 → 120 terms):

| | full vocabulary | frameworks excluded |
|---|---|---|
| density by year | 4.87 → 7.45 (+53 %) | 4.43 → **6.35 (+43 %)** |
| $\ESGSI$ slope | −0.205/yr, $p=1.7\times10^{-8}$ | **−0.155/yr, $p=2.5\times10^{-5}$** |

The trend survives at roughly three-quarters of its magnitude and remains
significant at any conventional level. This does **not** dispose of the
mandate confound — regulation elicits substantive disclosure as well as
regime names, and that part is not separable — but it bounds the purely
definitional component. §11 must report it inside the concession, not as a
refutation of it.

## 7c. Lexicon-robustness experiments — SANCTIONED, added after assembly

The paper's specification remains the **154-term vocabulary**. Everything below
is a robustness check run on the same 343 documents, and must be reported as
such. Do not renumber the paper onto the 289-term vocabulary.

**Provenance.** An external audit of 880 extracted zones from four 2023 reports
identified 147 ESG terms present in text but absent from the vocabulary
(§5 currently calls this unresolved — it is now addressed). Each candidate was
measured across all 343 reports for frequency and dispersion (occurrences,
documents, companies, countries, share held by the single largest company) and
classified: **102** broadly dispersed terms accepted; 18 frequent but
polysemous terms handled by collocation, of which only `compliance`, `social`
and `proxy` actually required it; 22 terms flagged sectoral. Vocabulary
154 → 289 TF-IDF entries (296 natural-form entries before lemmatisation).

Three bookkeeping facts a section author will otherwise trip on:

- **102, not 103.** `power purchase agreement` was meant to replace `ppa`
  (the audit found four of five `ppa` mentions were US subsidiary names) and
  was lost to a generation-script bug. It is absent from the vocabulary. One
  term of 289; not worth a re-run, but do not write 103.
- **143 candidate strings measured, not 147.** The audit list contains
  near-duplicate pairs (`conflict of interest`/`conflicts of interest`,
  `fair trade`/`fairtrade`, `cop 28`/`cop28`, `lca`/`life cycle assessment`).
  The correct decomposition is **103 classified as accept + 18 collocation +
  22 sectoral = 143**, of which **102 shipped** — `power purchase agreement`
  was lost to the script bug. Write "103 classified, 102 shipped"; do not
  write 102 + 18 + 22 = 143, which is arithmetically false (it is 142).
  Never write "147 candidates were measured".
- **Two counting bases, both correct.** The dispersion table counts prefix
  matches (`\bsocial`), the collocation analysis counts whole words
  (`\bsocial\b`): `social` 50,026 vs 48,198, `compliance` 55,974 vs 55,780.
  The gap is `socially`, `socialism` and similar. State which base a figure
  comes from, or use the whole-word one.

**`social` ships as an inclusion list, not an exclusion list.**
`informes/colocaciones_analisis.md` recommends exclusion on the evidence — only
8.7 % of `social` is non-ESG and it sits in three enumerable strings — but the
vocabulary architecture has no exclusion mechanism, and a bare `social` in a
TF-IDF vocabulary cannot carry a negative lookahead. Eight collocations were
implemented instead. The cost is recall on the long tail of valid uses. Say
what shipped; do not describe an exclusion mechanism that does not exist.

> **CRITICAL — the vocabulary feeds TWO stages.** `esg_terms.txt` drives both
> the paragraph-selection regex in `lexical_document_filter.py` AND the TF-IDF
> vocabulary. Changing it therefore changes *which text is extracted* as well
> as *how that text is scored*. The end-to-end pipeline runs are consequently
> **confounded** as a test of the scoring vocabulary: between the 154 and 289
> runs, $\SEN$ — which the Loughran-McDonald dictionary computes with no
> reference whatever to the ESG vocabulary — differs in **343 of 343**
> documents, which is only possible if the extracted text changed. Corpus size
> went 14.4M → 16.42M lemmatised tokens.
>
> **Report the SCORING-ONLY figures below.** They hold the corpus fixed (the
> 289-term extraction) and vary only the vocabulary used to score it, which is
> the isolation the claim needs. Never write that the extraction was held
> constant across the pipeline runs; it was not.

**(A) Near-doubling the scoring vocabulary, corpus fixed.**

| | 154 terms | 289 terms |
|---|---|---|
| $\susdens$ mean | 5.1682 | 6.5162 (**+26.1 %**) |
| Flagged | 172/343 | 164/343 |
| Label flips | — | **18 (5.2 %)** |
| Index correlation | — | **0.9882** |
| Trend slope | −0.219/yr, $p=4.7\times10^{-10}$ | −0.194/yr, $p=3.4\times10^{-8}$ |

The 154 entries are a strict subset of the 289, so the increment is additive.

**(B) Adding the 22 sectoral terms, corpus fixed.**

| | 289 | 289 + 22 |
|---|---|---|
| $\susdens$ mean | 6.5162 | 6.5406 (+0.37 %) |
| Flagged | 164/343 | 167/343 |
| Label flips | — | **3 (0.9 %)** |
| Index correlation | — | **0.99980** |

Company-level effect of (B), mean change in $\susdens$: **Danone +0.354**
(`nutrition`, `food safety`), Enel +0.111 (`energy storage`), Schneider
Electric +0.099. The sectoral terms do exactly what the objection to them
predicts — they reward a firm for its industry rather than its disclosure.
They carry 2.0 % of the candidate mass, so the index barely moves; that is a
consequence of their mass share, **not** evidence that they are harmless, and
must not be reported as the latter.

**Two measurement defects in `term_dispersion.csv`, both now corrected here.**
The unigram pass tokenised on `[a-z]+`, so terms containing digits were
recorded as zero. Re-measured with word boundaries: `sf6` 563 occurrences in
17 companies, largest share 0.867 — correctly sectoral; `cop28` 83
occurrences in 19 companies, largest share 0.422 — **misclassified, it meets
the acceptance rule** and sits in the sectoral file in error. Impact is
negligible (83 occurrences inside a 22-term set that moves 3 labels), but the
misclassification is real and should be stated rather than quietly fixed.
`board oversight` was checked and stands as sectoral: 8 companies, not 30.

**The hierarchy — and the objection to presenting it as one.**

| choice varied | labels moved |
|---|---|
| Vector normalisation (§10) | 96/343 (28.0 %) |
| Scoring vocabulary, 154 → 289 terms | 18/343 (5.2 %) |
| Sectoral terms, ±22 | 3/343 (0.9 %) |

The table survives the adversarial exchange, but restructured. The reviewer's
argument for cutting it — that normalising flips by the index disturbance
$(1-r)$ reverses the ranking — was tested and **does not hold**: fitted over
2,354 random perturbations, flips scale as $\sqrt{1-r}$
($\log \text{flips} = 5.349 + 0.572\log(1-r)$), so dividing an $x^{0.57}$
quantity by $x^{1.0}$ makes *any* perturbation set "reverse". Under the correct
scaling the three rows are 174 / 166 / 212 — a factor of 1.3 across
displacements differing by a factor of 1,500. No ranking survives
normalisation, but there is no artefact either.

Requirements on any use of the table: a column naming each row's baseline (row
3's baseline is the non-adopted 289-term list, so the caption must not claim
all rows are measured against the adopted specification), a caption stating
the rows are not commensurable perturbations, and the note that measured
against the adopted specification the 289+22 vocabulary moves **17** labels,
not 3. It licenses: *of the three choices we varied, the undisclosed one
changes the most classifications*. It does not license: *vector normalisation
is intrinsically a larger methodological choice than vocabulary*.

**Do not overstate — four limits, all to be stated in the text:**

1. The 289-term list was built by auditing **this** corpus and admitted on
   dispersion measured over **these** documents. The perturbation is not
   independent of the sample.
2. The 154 entries are wholly retained inside the 289 and the additions are
   mostly low-frequency, so a high correlation is partly structural. With
   $t = \sigma(\text{increment})/\sigma(\text{base}) = 0.2359$, the minimum
   attainable correlation is $\sqrt{1-t^2} = 0.9718$ at the component level.
   **Compare like with like:** the component-level observation is 0.9791
   against that 0.9718 floor; the index-level observation is 0.9882 against an
   index-level floor of 0.9841, a window 0.0159 wide. At both levels the
   observation sits at **26 %** of its achievable window — not "well above the
   floor", which an earlier revision of this file wrongly claimed by pairing an
   index-level observation with a component-level floor.
3. **The random-deletion null reproduces; this is conceded.** 5,000 draws
   deleting 30 of the 154 entries give a median of 17 flips and mean
   $r = 0.98775$. Our 18 flips sit at the 56th percentile and our $r$ at the
   58th; conditioning on the same index displacement, at the 48th. Our result
   is an average draw from the null on every margin tested.

   **Binding consequence.** The claim is: *the index is insensitive to
   vocabulary perturbation of this magnitude.* It may **not** imply the
   curation was validated. One clause may note that matched on mass rather
   than term count our result sits at the 18th percentile — a tilt, not
   evidence of curation quality.
4. The classification thresholds were set with results visible. §12 already
   concedes there was no blind specification choice; this addition falls under
   that concession and must not be written as if it escaped it.

**Known omission.** `power purchase agreement` was meant to replace `ppa` and
was lost to a script bug; it is absent from the vocabulary. One term of 289.
It is documented rather than fixed because adding it would change every figure
above and require a further hour of recomputation for no analytical gain —
but the reported vocabulary is 289 entries **including** that gap, and the
paper should not imply otherwise.

## 8. Claims discipline

- The central claim is §8's: a published conclusion is an artefact of two undisclosed choices. Every other section supports it or is explicitly subordinate.
- **Do not overclaim the corpus reproduction.** Lagasio's own Table 2 establishes *which* score normalisation was used — that is the strong evidence. Our corpus reproduction establishes the *consequence*. A matching mean on a different corpus with a different vocabulary is consistency, not proof, and must be worded as such.
- The factorial percentages are **our-corpus quantities**. They cannot be presented as estimates of the original's flagged share.
- **The original's flagged share is never reported, but its own Table 2 bounds it.** The published 75th percentile of $\ESGSI$ is $-0.1054 < 0$, so at least 75 % of the 749 reports fall below the threshold and **at most 25 % are flagged**. This is Move-1 class evidence — it uses only the original's published quantiles — and it corroborates the min–max reading independently, since the printed z-score formula would put the figure near 50 %. Use it; do not describe the flagged share as unknowable.
- The temporal finding must not be headlined. Having argued the instrument is unsound under one specification, we report the series under ours and show it is robust to the choice — we do not present it as the paper's contribution.
- Where we differ from the original by design (corpus, lexicon, $\SEN$ dictionary), say so plainly. This is an adaptation, not a strict replication.

## 9. Hard gate

The possible fabricated references in the original are **out of scope for
every draft**. Do not mention, allude to, or hint at them. This is a serious
allegation, unverified, and unnecessary to any argument we make.

## 10. Mechanics

- Write **only** your own section, as a fragment: start at `\section{...}`, no preamble, no `\begin{document}`.
- File: `paper/sections/<NN>_<slug>.tex`, exactly the name given in your brief.
- Tables: `booktabs` (`\toprule \midrule \bottomrule`), no vertical rules.
- Cite the original as `\citep{lagasio2024}`; add any other key you use to your report-back so the shared `references.bib` can be assembled.
- Target length is in your brief. Prefer a table to a paragraph restating it.
