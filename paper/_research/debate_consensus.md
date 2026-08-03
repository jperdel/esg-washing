# Consensus: how the lexicon-robustness material enters the paper

Binding merge specification for `_draft_10_lexicon.tex`, `_draft_11_lexicon.tex`,
`_draft_05_audit.tex`, `_draft_12_sectoral.tex`. It supersedes both
`debate_against.md` and the drafts as written. A merge pass should be able to
execute this file without reading either.

Everything numbered here was recomputed on the fixed-corpus setup
(`data/clean/processed_texts.csv` deduplicated to 343 documents = the 289-term
extraction, 16.42 M lemmatised tokens; the 154-term vocabulary from
`git show ce6804a:metadata/esg_terms_lemmatized.txt`; `SEN` held at its value on
that corpus). The setup reproduces every figure in `conventions.md` §7c to the
last printed digit: 5.1682 → 6.5162 (+26.1 %), 172/343 → 164/343, 18 flips,
$r = 0.988247$, slope $-0.2192 \to -0.1942$; sectoral 6.5162 → 6.5406 (+0.37 %),
164 → 167, 3 flips, $r = 0.99980$, Danone $+0.3544$, Enel $+0.1106$, Schneider
$+0.0994$.

---

## Part 0 — the two verifications, and what they cost us

### (1a) The random-deletion null REPRODUCES. Concede it.

5,000 random deletions of 30 of the 154 entries, rescored on the fixed corpus
against the 154-term index:

| terms deleted at random | draws | median flips | mean flips | p05–p95 | mean index $r$ | mean mass removed |
|---|---|---|---|---|---|---|
| 10 | 2,000 | 6 | 7.35 | 1–19 | 0.99656 | 6.5 % |
| 20 | 2,000 | 12 | 13.10 | 4–24 | 0.99251 | 13.0 % |
| **30** | **5,000** | **17** | **17.88** | **8–29** | **0.98775** | **19.5 %** |
| 40 | 2,000 | 22 | 22.04 | 12–34 | 0.98259 | 26.2 % |
| 50 | 2,000 | 26 | 25.99 | 15–38 | 0.97583 | 32.7 % |
| 75 | 2,000 | 34 | 34.90 | 23–48 | 0.95542 | 48.7 % |

Our 18 flips sit at **P(flips ≤ 18) = 0.560**; our $r = 0.9882$ at
**P(r ≥ 0.9882) = 0.578**. Conditional on the same index displacement (random
draws with $r$ within $\pm0.0010$ of ours, $n = 882$), the median is **19 flips
against our 18 — the 48th percentile**. Our result is an average draw from the
null on every margin tested. The sceptic's median of 17 flips reproduces exactly;
his mean $r$ of 0.9882 against our 0.98775 is within the Monte Carlo error of his
300 draws (our median $r$ at $k=30$ is 0.98966). **The claim survives in full.**

One partial defence, and it is genuinely partial. The sceptic matched on *term
count*; a 30-term deletion removes only 19.5 % of the base density mass while our
expansion adds 26.1 %. Matched on **mass** instead (3,000 draws, median 41 terms
deleted), the random null gives a median of **23 flips** at mean $r = 0.9808$;
our 18 flips sit at the **18th** percentile and our $r$ at the **12th**. So per
unit of mass moved, the curated expansion is somewhat more stable than an
arbitrary one. That is a tilt, not a result — 12th percentile is not evidence of
curation quality, and the text must not present it as one. It is worth one
clause, no more.

**Consequence, binding.** The claim weakens to: *the index is insensitive to
vocabulary perturbation of this magnitude.* It may not imply the curation was
validated. Every sentence in the drafts that says or implies otherwise comes out
(list in Part 2).

### (1b) The floor derivation is CORRECT — and `conventions.md` §7c states it in a way that flatters us.

With $A$ = 154-term density, $D$ = the 135-entry increment:
$\sigma(A) = 1.8269$, $\sigma(D) = 0.4309$, $t = 0.2359$ — the sceptic's figures
exactly. $r(A, A{+}D) = (1+\rho t)/\sqrt{1+t^2+2\rho t}$; the derivative
$t^2(t+\rho)/\mathrm{den}^3$ vanishes at $\rho = -t$, giving
$r_{\min} = \sqrt{1-t^2} = \mathbf{0.971787}$. Brute force over
$\rho \in [-1,1]$ confirms the minimum to six places. **The derivation holds.**

Two corrections, one of which goes against us:

- **Against us.** §7c limit 2 compares the *observed index* correlation (0.9882)
  with the *component-level* floor (0.972) and concludes it sits "well above the
  floor". Like for like: the component observation is **0.9791** against floor
  0.9718 — **26 %** of the way up a window 0.0282 wide. The **index-level** floor
  at the same $\sigma$ ratio is **0.9841** (solved analytically on the
  $\{A, \mathrm{SEN}\}$ subspace, with $\mathrm{corr}(\mathrm{SEN},A) = +0.108$),
  so the index observation 0.9882 sits at **26 %** of a window **0.0159** wide.
  Both levels agree: roughly a quarter of the achievable range, not "well above".
  **Amend §7c limit 2 accordingly and use the index numbers in the paper**, since
  the index is what the table reports.
- **For us, weakly.** The floor is not attainable by any real increment. $D$ is a
  sum of counts, so $D \ge 0$ elementwise ($\min D = 0.4101$: every document
  gains at least 0.41). Re-allocating the *observed* increment anti-monotonically
  in $A$ — the most hostile rearrangement available — overshoots to
  $\rho = -0.955$ and yields $r = 0.9959$, *above* what we observed. The 0.972
  floor requires $\rho$ to land on $-0.236$ almost exactly. Worth knowing; not
  worth printing. **Do not use this to soften the limit.**

---

## Part 1 — what did not survive the sceptic's case

Adjudicated on recomputation. These are the corrections a merge pass should make
to the *sceptic's* claims where the drafts or the conventions were closer to
right.

**(A) "The ordering reverses under any normalisation of perturbation size" —
FALSE, and it is the load-bearing argument for cutting the hierarchy table.**
The sceptic normalises label flips by $(1-r)$. That exponent is wrong. Fitted
over 2,354 random vocabulary perturbations spanning $k = 5$ to $100$ deleted
terms:

$$\log(\text{flips}) = 5.349 + 0.572\,\log(1-r), \qquad \text{i.e. flips} \propto (1-r)^{0.57} \approx \sqrt{1-r}$$

Dividing a quantity that scales as $x^{0.57}$ by $x^{1.0}$ yields $x^{-0.43}$,
which is monotonically decreasing in $x$ — so *any* set of perturbations obeying
the empirical law will "reverse" under his normalisation. It is an artefact of
the exponent. Under the correct one:

| choice varied | flips | $1-r$ | flips$/(1-r)$ | flips$/\sqrt{1-r}$ | predicted flips |
|---|---|---|---|---|---|
| Vector normalisation | 96 | 0.3036 | 316 | **174** | 106 |
| Vocabulary 154 → 289 | 18 | 0.0118 | 1,532 | **166** | 17 |
| Sectoral $+22$ | 3 | 0.0002 | 15,000 | **212** | 2 |

Across three perturbations that differ by a factor of **1,500** in $(1-r)$, the
scale-free statistic ranges only over **166–212**, a factor of 1.28, and each
row's observed flips are within a factor of 1.9 of what the law predicts from its
displacement alone. (The sectoral row is the least reliable of the three: 3 flips
is a small count and $1-r = 0.0002$ carries one significant figure.) There is
**no ranking** once size is controlled — which kills the drafts' inference as
thoroughly as the sceptic's, but it also means the table is not the misleading
artefact he says it is. `conventions.md` §7c's
sentence — *"normalising label flips by the index disturbance each causes
$(1-r)$ reverses the ranking"* — is arithmetically true and inferentially wrong;
**amend it** to name the mis-specification.

**(B) "The §7b(B) framework deletion moves the slope five times as much as the
addition, and would sit above the vocabulary row" — FALSE like-for-like.**
The sceptic compared $-0.205 \to -0.155$ (adopted extraction) against
$-0.205 \to -0.194$ (cross-extraction). On one corpus, both measured against the
154-term index:

| direction | terms | flips | index $r$ | slope |
|---|---|---|---|---|
| **inward** — delete the 34-term frameworks block (154 → 120) | $-34$ | **14/343** | 0.9867 | $-0.219 \to -0.172$ (−21 %) |
| **outward** — the audited expansion (154 → 289) | $+135$ | **18/343** | 0.9882 | $-0.219 \to -0.194$ (−11 %) |

The inward deletion moves the slope about **1.9×** as much, not 5×, and moves
**fewer** labels (14 vs 18), so it would sit *below* the vocabulary row, not
above it. The sceptic's own reconstruction ($-0.193$, 10 flips) does not
reproduce: the correct fixed-corpus figures are $-0.1725$ and 14 flips. He
appears to have missed the block's lemmatisation artefacts (`esr`, `ecovadi`,
`sustainalytic`, `science base target`, `taxonomy align`, `global compact`,
`non financial`), all 34 of which I matched. **The published $-0.155$ in
`tab:temporal-frameworks` is correct on the adopted extraction and stands
unchanged.**

**(C) "The index-level correlation is the lower of the two, breaking §10's
regularity" — FALSE on the fixed corpus.** His 0.9906 was cross-extraction.
Corrected: component $r(\mathrm{dens}_{154},\mathrm{dens}_{289}) = 0.9791$ against
index $0.9882$; component $r(\mathrm{dens}_{289},\mathrm{dens}_{289+22}) = 0.99965$
against index $0.99980$. Index runs above component in both pairs. §10's stated
regularity holds and **needs no defensive sentence.**

**(D) "Sixteen `social …` collocations" — wrong; there are eight.** Counted in
`metadata/esg_terms.txt`: 15 `compliance` collocations, **8** `social`, 2
`proxy`. This does not touch his §6 point, which stands and is already binding in
§7c — only his arithmetic.

**(E) Draft 05's `compliance` 55,988 / `social` 50,090 match the shipped
`term_dispersion.csv` exactly.** `conventions.md` §7c gives 55,974 / 50,026 for
the same "prefix base" and does **not** match the file it names. The draft is not
the erring party. *Flagged for the author per the conventions' own instruction:
either §7c's prefix-base pair is stale or it was re-measured on the current
extraction without saying so.* The paper should quote the whole-word base
(55,780 / 48,198) and name it, which sidesteps the conflict.

**(F) `board oversight` — both numbers are right, on different bases.** The
dispersion table and direct natural-form measurement give **8 firms**, max share
0.333/0.364: sectoral by the stated rule, and the classification stands. But the
lemmatised bigram the TF-IDF vocabulary actually matches occurs in **30 firms /
77 documents / 105 occurrences** on the extracted corpus, because stopword
removal manufactures adjacency (`the board's oversight of` → `board oversight`).
The classification was made on one base; the instrument operates on another. This
is a *third* instance of the two-counting-bases problem §7c already flags, not a
misclassification. **One clause in §12, no re-litigation.**

**(G) The classification arithmetic — everyone is wrong, including
`conventions.md`.** `term_dispersion.csv` has **143** rows. The shipped files
hold 102 accepted + 15 admitted bare + 22 sectoral = 139; adding `compliance`,
`social`, `proxy` (collocation-handled) and `ppa` gives 143. §7c's
"102 + 18 + 22 = 143" is arithmetically false (= 142). **The correct
reconciliation, and the one the paper must use:**

> 143 candidate strings measured; **103 classified as broadly dispersed, of which
> 102 reached the vocabulary**; 18 frequent but polysemous (15 admitted bare,
> three — `compliance`, `social`, `proxy` — only in collocation); 22 flagged
> sectoral. $103 + 18 + 22 = 143$. `ppa` was rejected in favour of
> `power purchase agreement` (four of its five mentions were a US subsidiary
> name) and the replacement was lost to a generation-script bug, so 102 entries
> shipped.

Vocabulary arithmetic, verified: 154 original + 142 new = **296 natural-form
entries** (102 + 15 + 15 `compliance` + 8 `social` + 2 `proxy`) → **289**
lemmatised TF-IDF entries.

**(H) The headline verdict — "roughly a third of its length" — is rhetorical
symmetry, and it is refuted by the sceptic's own remedy list.** Prose word
counts: draft 10 = 618, draft 11 = 335, draft 05 = 533, draft 12 = 184; total
**1,670**. His 16 required edits are **3 deletions and 13 corrections or
additions** — he asks to keep `tab:lexicon-context`, restore a verdict sentence,
disclose what shipped for `social` and what it cost, add the achievable-range
sentence, add a §11 reconciliation clause, add a sectoral-heterogeneity
concession. Executed together with the four limits `conventions.md` §7c makes
binding, and after taking the largest available cut (the duplicated sectoral
paragraph, below), the material lands at **≈2,350 words: it grows by about
40 per cent rather than shrinking to a third.** That is the honest arithmetic and
the merge pass should not fight it.

The reason is structural and worth stating once. A robustness check that claims
a hierarchy and states no limits is short. A robustness check that claims only
insensitivity-of-this-magnitude and states four limits — corpus dependence,
one-sidedness, the random-deletion equivalence, the acceptance-rule floor — is
longer, because each limit costs two or three sentences and none of them can be
gestured at. **What contracts by two thirds is the claim, not the text.** The
section loses its headline conclusion, its hierarchy inference, its `BREADTH`
sentence, its "settled" claim and its "answerable and answered" flourish.

Two levers exist if the author still wants the count down, in this order. The
first is already taken below: the sectoral firm-level material is stated twice
across the four fragments and is cut from §10, saving ~110 words. The second, if
pressed, is `tab:robustness-hierarchy` and its paragraph, ~150 words and a table
— it is the marginal item in the whole package and §10's argument survives
without it. **Neither lever is the limits.** A version that keeps the drafts'
length by omitting the limits is the one outcome both sides agree is
unpublishable.

---

## Part 2 — the four drafts, decided

### Preconditions that block the merge (repository state, not prose)

Fix these first; three of the four drafts are unpublishable until they are done.

1. `metadata/esg_terms.txt`, `metadata/esg_terms_lemmatized.txt`,
   `src/lexical_document_filter.py`, `src/config.py`,
   `scripts/build_lexicons.py` are modified and uncommitted. `git checkout
   ce6804a --` restores the 154-term state; re-extract; recompute. Whatever the
   author chooses, **`results/metrics/results.csv` must be the paper's adopted
   154-term run on the 154-term extraction** (currently that run survives only in
   `results/metrics_baseline_paper/`, which no section mentions). `conventions.md`
   §7 declares `results/metrics/results.csv` authoritative for every index value
   and it currently is not.
2. `13_reproducibility.tex` names `metadata/esg_terms_lemmatized.txt`
   **(154 entries)** as an output artefact. That file holds 289. Either restore
   it or amend the table and add the robustness vocabulary as a separate,
   named artefact.
3. Keep the three robustness artefacts under names §13 can publish:
   `esg_terms_lemmatized_289.txt`, `esg_terms_sectorial_lemmatized.txt`,
   `results/metrics_lexicon289/results.csv`. A referee following §13 must not
   silently get the robustness vocabulary.

### `_draft_10_lexicon.tex` — KEEP, rewritten. Target ≈820 words + 2 tables.

Up from 618 despite ~275 words of cuts, and the growth is all disclosure: three
of the four limits `conventions.md` §7c makes binding are not in the draft at
all. If the subsection must come in under 700, drop `tab:robustness-hierarchy`
and its paragraph; drop nothing else.

**Stays:** the framing paragraph (the standing objection to dictionary methods);
`tab:robustness-lexicon` with corrected figures; the closing specification
paragraph; the "two limits bound the check narrowly" material, promoted to carry
the argument rather than trail it.

**Cut outright** (≈300 words):
- The paragraph reading `tab:robustness-hierarchy` (*"Vector normalisation moves
  nearly six times as many labels … answerable and answered"*). Every clause of
  it is now unsupported.
- The `\BREADTH` sentence. Recomputed: 35.75 at 154 entries, 56.30 at 289, 56.79
  at 289+22 — it rises because the ceiling is the entry count. Mechanical; carries
  no information; the draft's 56.81 is also wrong.
- The full restatement of the classification counts. Draft 05 carries it; §10
  cites `\S\ref{sec:lexicon}` and gives the totals only.
- The trend $p$-values to two significant figures in the caption. §12 concedes
  these are anti-conservative. Report direction and significance.
- **The whole sectoral firm-level paragraph.** The 22 terms are §12's material,
  the draft-12 fragment states them better, and between the four fragments they
  are currently stated twice. §10 keeps the table column and one sentence:
  *"The 22 terms held back as sectoral move 3 labels of 343 when added on top of
  the expansion; \S\ref{sec:limitations} reports what they do at firm level and
  why they are excluded."* This is the single largest available cut and it costs
  the section nothing.

**Corrections to the numbers in `tab:robustness-lexicon`** — replace the whole
table body with the fixed-corpus figures:

| | 154 terms | 289 terms | 289 $+$ 22 sectoral |
|---|---|---|---|
| Mean $\susdens$ | 5.168 | 6.516 | 6.541 |
| Change vs preceding column | — | $+26.1$ % | $+0.4$ % |
| Flagged | 172/343 | 164/343 | 167/343 |
| Label flips vs preceding column | — | 18 (5.2 %) | 3 (0.9 %) |
| Pearson $r$ vs preceding column | — | 0.9882 | 0.9998 |
| $\ESGSI(\susdens)$ slope | $-0.219$/yr | $-0.194$/yr | $-0.193$/yr |

**Must be added.** Three items.

*(i) The design statement.* Delete *"Both experiments run on the same 343
documents and the same extraction architecture … not how those zones are found"*.
It is false and cannot be reworded. Replace with, verbatim:

> Both experiments run on the same 343 documents and score them with different
> vocabularies; the extracted text is held fixed at the expansion's own
> extraction. That isolation is deliberate and it is not free: our vocabulary
> file drives the paragraph-selection regex of \S\ref{sec:extraction} as well as
> the TF-IDF vocabulary, so an end-to-end comparison of the two vocabularies
> would confound what is scored with what is extracted. Holding the corpus fixed
> removes the confound at the cost that the first column of
> Table~\ref{tab:robustness-lexicon} is not the index reported elsewhere in this
> paper: the adopted specification of \S\ref{sec:lexicon} scores its own
> extraction, where mean $\susdens$ is 5.897 rather than 5.168. The difference
> between those two figures is an extraction effect, and \S\ref{sec:limitations}
> reports it.

*(ii) The random-deletion null.* Verbatim:

> The check is not specific to the terms we chose. Deleting 30 of the 154 entries
> at random and rescoring --- 5,000 draws --- moves a median of 17 labels at a
> mean index correlation of 0.988; our 18 flips and 0.9882 sit at the 56th and
> 58th percentiles of that distribution, and among the random draws that displace
> the index as far as ours does the median is 19 labels against our 18. What the
> experiment measures is therefore the index's insensitivity to a vocabulary
> perturbation of this magnitude, not the adequacy of the particular terms
> admitted. Matched on occurrence mass rather than on term count the curated
> expansion is modestly the more stable of the two, at the eighteenth percentile
> of the random distribution; we report that without leaning on it.

*(iii) The achievable range.* Verbatim:

> Part of the agreement is structural rather than empirical. The 154 entries are
> retained whole inside the 289, so the increment is additive, and its standard
> deviation is $0.236$ of the base measure's. That ratio alone bounds the
> correlation between the two substance scores below at
> $\sqrt{1-0.236^{2}} = 0.972$ and the correlation between the two indices at
> $0.984$. The observed $0.9882$ lies about a quarter of the way up a window
> $0.016$ wide. The exercise is not vacuous --- a fifth of the expanded numerator
> is new material --- but the range in which the answer could have landed was
> fixed by the acceptance rule before any document was scored, because terms
> admitted on dispersion contribute a near-constant per-document increment.

**Denominators.** §7 requires the denominator with every percentage. Wherever the
sectoral mass appears — here or in §12 — state both: the 22 carry **2.0 %** of
the 143 candidates' occurrence mass and **0.37 %** of the substance numerator
they are added to. Do not print the first alone; the paper's central claim is
that a denominator was mis-stated elsewhere.

### `tab:robustness-hierarchy` — SURVIVES, restructured. This is the settled answer.

It is not cut. The argument for cutting it — that its ordering reverses under
normalisation — is false (Part 1A). But the drafts' reading of it is equally
false, so it survives only with a new caption, a third column, and a replacement
paragraph.

**Table** — add a column naming each row's baseline, since the current caption's
*"measured against the specification adopted for that choice"* is untrue of row 3
and exists to paper over exactly that:

| Choice varied | Measured against | Labels moved |
|---|---|---|
| Vector normalisation (Table~\ref{tab:robustness-index}) | $\ESGSI(\susdens)$, 154 terms | 96/343 (28.0 %) |
| Scoring vocabulary, 154 $\to$ 289 terms | $\ESGSI(\susdens)$, 154 terms | 18/343 (5.2 %) |
| Sectoral terms, $+22$ | the 289-term vocabulary | 3/343 (0.9 %) |

**Caption** — verbatim:

> Labels moved by each of the three choices this paper varies ($n = 343$;
> $z$-score normalisation and the $\ESGSI > 0$ threshold throughout). Each row
> names its own baseline: the third is an increment on the robustness vocabulary,
> not on the adopted one. The rows are not commensurable perturbations and the
> column does not rank the choices by importance; see the text.

**Replacement paragraph** — verbatim, and it must not be softened:

> The three rows are not perturbations of comparable size, and the column should
> not be read as ranking the choices by how much each matters. Across 2,354
> random perturbations of this vocabulary, label flips scale as $(1-r)^{0.57}$,
> close to the square root of the index displacement. On that scaling the three
> rows differ by a factor of $1.3$ although the displacements behind them differ
> by a factor of $1{,}500$: each moves about as many labels as any perturbation
> of its size would move. What separates the rows is how far each moved the
> index, not what kind of choice each was. Nor is the third row's margin over the
> second what it appears: measured against the adopted specification rather than
> incrementally, the full $289+22$ vocabulary moves 17 labels, not 3. What the
> column licenses is narrower than a hierarchy and is still this section's point:
> of the three choices we varied, the one \citet{lagasio2024} never states in
> terms is the one that changes the most classifications.

### `_draft_11_lexicon.tex` — KEEP, lightest touch. Target ≈400 words.

**Stays:** the whole argument, including the harvested-from-FY2023 aggravation,
which is the most careful writing in the four fragments and must not be trimmed.

**Cut, both sentences, without replacement in kind:**
- *"Nothing in this section turns on where the boundary of the vocabulary was
  drawn: the paper's specification remains the 154-term list, and the series
  would read the same under either."*
- *"The first is settled here"* (keep *"the second is not settled anywhere in
  this paper"*).

**Replacement for the first** — verbatim:

> The series does turn on where the boundary is drawn, and this section contains
> both directions. Moving it outward, by the 135 audited entries, changes the
> slope by about a ninth; moving it inward, by striking the 34-term
> reporting-frameworks block (Table~\ref{tab:temporal-frameworks}), changes it by
> about a fifth, and the two are of the same order rather than of different ones.
> What the outward check bounds is the sensitivity of the series to a defensible
> extension of this vocabulary, built from these documents and admitted on
> dispersion measured over them. It bounds nothing about a vocabulary built on
> other principles.

**Replacement for the second** — verbatim:

> Robustness to the lexicon and identification of what the lexicon counts are
> different questions. The first is bounded here, in one direction, for a class
> of terms selected on this corpus for the property that makes them least able to
> move a density ranking; the second is not settled anywhere in this paper.

**Correct the baselines.** The subsection reports the fixed-corpus experiment, so
its slopes are $-0.219 \to -0.194$ (and $-0.193$ with the sectoral terms), not
$-0.205 \to -0.194 \to -0.195$. **Add one clause** stating that these are
computed on the expansion's extraction and are therefore not the $-0.205$ of
Table~\ref{tab:temporal-series}, with a pointer to §10's design statement. The
$-0.155$ of `tab:temporal-frameworks` is on the adopted extraction and is
**unchanged**; the inward/outward comparison above is stated in proportions
precisely so the two corpora are never differenced.

### `_draft_05_audit.tex` — KEEP, amended substantively. Target ≈660 words + table.

**Stays:** the occurrence-weighting reversal (first paragraph); the re-measurement
paragraph; `tab:lexicon-context`; the two-bases paragraph and the sentence
*"which bounds the problem from above and does not estimate it"* — the best
writing in the four fragments, keep it exactly.

**Corrections:**
- "103 are broadly dispersed" → use the reconciliation in Part 1(G). Suggested:
  *"on that base 103 are broadly dispersed --- 102 of which reached the
  vocabulary, one surface form having been lost to a generation-script bug ---
  18 are frequent but polysemous, and 22 are concentrated enough in one industry
  to be called sectoral."*
- `compliance` 55,988 / `social` 50,090 → quote the whole-word base **55,780 /
  48,198** and name it: *"counted as whole words over the full text of all 343
  reports"*. This removes the three-figures problem without adjudicating it.
- "2.0 % of the candidates' occurrence mass" → keep, it is correctly denominated
  here.

**Must be added — the `social` disclosure.** The paragraph currently states the
exclusion case for `social`, the inclusion case for `compliance`, and then
describes only what was implemented for `compliance`. Replace from *"so the valid
uses are a long tail…"* to the end of that paragraph with, verbatim:

> so on the evidence the right treatment of \emph{social} is an exception list.
> The vocabulary architecture supplies none: a bare entry in a TF-IDF vocabulary
> cannot carry a negative lookahead, and no exclusion mechanism exists at any
> stage. What shipped for \emph{social} is therefore the opposite of what its own
> analysis recommends --- an inclusion list of eight collocations, admitting only
> the uses they name and discarding the long tail of valid ones the analysis
> identified as the majority of its mass. \emph{Compliance} inverts the evidence:
> 41.8\,\% are unambiguously ESG, 8.7\,\% are accounting or audit usage sitting
> almost entirely in the auditor's report and the notes to the financial
> statements, and the largest block, 44.5\,\%, carries no marker either way, so
> for that term an inclusion list of fifteen collocations is what the evidence
> supports as well as what was built. \emph{Proxy} was adjudicated on the same
> evidence, against \emph{proxy hedging}.

**Must be added — the recall verdict.** The merge instruction deletes *"The gap
stands as an open, unresolved bound on recall"* and ends the subsection on a
specification statement. That is a rhetorical retirement of a bound that is
numerically unchanged: **zero** of the 147 audited omissions are in the adopted
154-term vocabulary. The paragraph may not end where the draft ends it. Append,
verbatim:

> What the dispersion analysis retires is the sector-bias objection, which is now
> measured, and not the recall bound, which is unchanged: the adopted vocabulary
> incorporates none of the 147, and every figure in this paper is computed on the
> 154 entries. The bound stands exactly where \S\ref{sec:limitations} leaves it.

### `_draft_12_sectoral.tex` — KEEP, and it absorbs §10's sectoral prose. Target ≈300 words + 2 table rows, plus ≈175 words elsewhere in §12.

Nearest to publishable of the four; its framing — disclosed, immaterial, not
necessary, earlier framing overstated — is the register the other three should
have used. Keep it.

**Corrections:**
- "one label of 343" → **3 of 343**, on the fixed corpus.
- "6.5162 to 6.5327" → **6.5162 to 6.5406**; "164/343 to 165/343" → **164/343 to
  167/343**; "$-0.194$ to $-0.195$" → **$-0.194$ to $-0.193$**; "0.99978" →
  **0.99980**.
- "0.206 for Danone" → **0.354**; "0.101 for Schneider Electric" → **0.099**.
- "$\pm 22$ sectoral terms" → **the 22 natural-form entries yield 19 live TF-IDF
  entries**: `b corp`, `iso 37001` and `cop 28` are lost to preprocessing, and
  two survive as artefacts of the class §12 already lists — `better cotton` →
  `well cotton`, `living income` → `live income` (the latter scoring 2
  occurrences in one firm). Say 19, not 22.

**Must be added — the firm-level effect, qualified.** This replaces the draft's
unqualified *"They do meanwhile exactly what the objection to them predicts"*
and absorbs what §10 gives up. All firm counts below are the dispersion table's
own, so no new counting base enters the paper. Verbatim:

> Some of them do what the objection predicts, and that is why they are out: mean
> $\susdens$ rises $0.354$ for Danone (\emph{nutrition}, \emph{food safety}),
> $0.111$ for Enel (\emph{energy storage}) and $0.099$ for Schneider Electric,
> against a corpus mean of $0.024$, and 34 of the 49 firms gain less than $0.02$
> --- firms credited for the industry they operate in rather than for what they
> disclosed. The exclusion avoids a bias we can name and measure, and we prefer it
> on that ground; what it does not do is carry any result.

**Must be added — the bin is not homogeneous.** Immediately after. Verbatim:

> The bin is not one kind of term. The rule sorts on concentration without
> distinguishing its causes, so alongside industry vocabulary it holds
> house-style governance language --- \emph{executive compensation} appears in 29
> of the 49 companies and is excluded on the second limb of the rule, that half
> its occurrences sit in one firm --- and one conference name. Two entries were
> measured as zero by a tokeniser that split on letters only: re-measured with
> word boundaries, \emph{sf6} is correctly sectoral and \emph{cop28} meets the
> acceptance rule and sits in the sectoral file in error. And the dispersion of
> \emph{board oversight} depends on the base: 8 companies as a natural-form
> phrase, 30 as the lemmatised bigram the vocabulary actually matches, because
> stopword removal manufactures the adjacency. The rule was applied consistently;
> its inputs were not all measured on the representation the instrument scores.

**Must be added — the extraction was varied, and this is the paper's only
measurement of it.** §12 currently says the extraction stage *"was set by
judgement and never varied … we do not test sensitivity to it"*. That is no
longer true, and the honest repair is to report the measurement rather than to
delete the concession. Insert into "One extraction pipeline, one tone
dictionary", verbatim:

> The extraction stage shares its vocabulary with the measurement stage, so
> building the robustness vocabulary of \S\ref{sec:robustness-lexicon} varied it
> incidentally, and the effect can be read off. Holding the adopted 154-term
> scoring vocabulary fixed and swapping only the extraction --- 154 natural-form
> terms against 296 --- takes the corpus from 14.4 to 16.4 million lemmatised
> tokens, lowers mean $\susdens$ from 5.897 to 5.168, moves 7 of 343 labels and
> leaves the two indices correlated at 0.988. The added zones are therefore
> poorer in adopted-vocabulary terms than the zones already admitted, which is
> what the density threshold predicts of text entering at the extensive margin.
> This is one point, not a sensitivity analysis: the three free parameters of
> \S\ref{sec:extraction} --- the density threshold, the context window and the
> minimum zone length --- are still never varied.

*Verified directly except for the 14.4 M figure, which is §12's existing number
for the adopted extraction and cannot be re-measured until that extraction is
restored on disk (precondition 1). The 16.42 M, the 5.897/5.168 pair, the 7 label
flips and the 0.9878 correlation are measured. **Confirm 14.4 M on the re-run
before this ships.***

**Must be added — the non-blind clause.** §12's no-blind-choice paragraph lists
knobs set once. This addition set a knob *after* the results were known and
presented it as a check on them. One clause, verbatim, appended to that
paragraph:

> The classification thresholds of \S\ref{sec:robustness-lexicon} --- twelve
> companies, half the occurrences in one firm --- were set later than the rest
> and with the results they adjudicate already computable, so they fall inside
> this concession rather than outside it.

**Rows for `tab:limitations-origin`:**

```latex
Sectoral vocabulary excluded by choice (moves 3 labels of 343) & Ours (disclosed) \\
Extraction vocabulary varied once, incidentally (7 labels of 343) & Ours \\
```

**Delete** from the recall-floor paragraph, as the draft's merge instruction
says: *"They are not incorporated, and the omission is unresolved rather than
defended, because naive inclusion is its own hazard: 64 of the 147 occur in only
one of the four audited firms…"* and *"Choosing between the two errors requires a
sector-bias analysis we have not performed."* **Keep** the sentence that the 147
are not incorporated and that recall is bounded — that is the part §05 now
depends on.

---

## Part 3 — the one-line summary of what the paper may now claim

Both sides converge on this and it is what should survive in the reader's memory.
Any sentence in any of the four drafts that claims more than this is out.

> An expansion of the scoring vocabulary to 289 entries, built by auditing this
> corpus and admitted on dispersion measured over it, moves 18 of 343 labels and
> correlates 0.988 with the adopted index. The check is one-sided --- the terms
> were selected for the property that makes them least able to move a density
> ranking --- it is not independent of the corpus it runs on, a random deletion
> of comparable size reproduces it, and a quarter of the achievable correlation
> range was fixed by the acceptance rule before any document was scored. It
> bounds the index's sensitivity to a defensible extension of this vocabulary. It
> does not bound the sensitivity to a vocabulary built on other principles, and
> nothing here does.

---

## Appendix — amendments owed to `conventions.md` §7c

Per §7's standing instruction to report rather than silently substitute:

1. **§7c limit 2** compares an index-level observation (0.9882) with a
   component-level floor (0.972). Like for like the observation sits at 26 % of
   its window at *both* levels; the index-level floor is 0.984 and its window is
   0.016 wide. Replace "well above the floor" with "about a quarter of the way up
   it".
2. **§7c limit 3** is confirmed: 5,000 draws give median 17 flips and mean
   $r = 0.9877$ at $k = 30$; our result sits at the 56th/58th percentile. The
   conditional weakening §7c anticipates is now unconditional.
3. **§7c hierarchy note.** "Normalising label flips by $(1-r)$ reverses the
   ranking" is true arithmetic and a mis-specified normalisation: flips scale as
   $(1-r)^{0.57}$, and on the correct scaling the three rows are
   indistinguishable. Amend so no section author cuts the table on the strength
   of the reversal.
4. **§7c bookkeeping.** "102 + 18 + 22 = 143" is false (= 142). The correct line
   is 103 + 18 + 22 = 143 with 102 shipped; `ppa` is one of the 143 and
   `power purchase agreement` is not.
5. **§7c counting bases.** The stated prefix-base figures (55,974 / 50,026) do
   not match `paper/_research/term_dispersion.csv`, which gives **55,988 /
   50,090**. Reconcile or mark which extraction each was measured on.
6. **§7c(B) sectoral totals** should record 6.5406 and 167/343 (the drafts'
   6.5327 and 165/343 are cross-extraction); the mean rise is 0.0244 and 34 of 49
   firms gain under 0.02.
