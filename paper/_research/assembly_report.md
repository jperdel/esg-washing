# Assembly report

Final assembly pass over the thirteen technical sections, `main.tex` and
`references.bib`. Work order: `handoff.md`; binding reference for every number
and every notation decision: `conventions.md`, including the CORRECTION block in
§7 and the post-drafting results in §7b.

Environment used for verification:
`C:/Users/Jorge/anaconda3/envs/esgwashing/python.exe`.

---

## A. The coordinated argument fix (handoff 1b)

The charge was that §2 treated the unstated `norm` as a reproducibility defect
while §7 built its L2-invariance demonstration on inferring `norm='l2'` — the
convention wanted both ways. Applied the handoff's resolution in both sections,
in deliberately matching language so that a referee comparing them finds no
daylight.

**Shared wording now in both**, near-verbatim: *"a reader following the ordinary
convention that only departures from library defaults are reported would infer
that the remaining defaults — `norm='l2'` among them — were in force"*, followed
by the statement that we make that inference, hold it reasonable, and therefore
do not also count the silence as a reproducibility failure.

**§2 `02_original_specification.tex`**
- Section preamble: added the one stated exception to "we record the silence, we
  do not substitute standard practice", naming `norm` explicitly and up front
  rather than in a footnote.
- "First, *vector* normalisation": rewritten. Table 2 records `norm` as *not
  stated* (a fact about the text) but not as a reproducibility gap.
- `tab:original-disclosure`: three rows re-worded — `norm` and
  `sublinear_tf`/`smooth_idf` now read "Not stated; library default reasonably
  inferable"; the aggregation row reads "Not stated; not a library setting, so no
  default supplies it".
- "Consequence for what follows" (now labelled `sec:original-consequence`):
  `norm` removed from the list of things a reader must supply from outside the
  article. New paragraph rests the charge on three gaps no convention can close,
  in strength order: (i) the aggregation from per-term TF-IDF values to a single
  SUS_i — a modelling step, not a library argument, so no default exists;
  (ii) the unpublished, unsized keyword list; (iii) the prose/formula
  contradiction, which is not a silence at all but two inconsistent statements
  the article itself makes.

**§7 `07_measurement_critique.tex`**
- The attribution paragraph now concedes at the outset and points at
  `sec:original-consequence` for where the charge actually rests, naming the same
  three gaps in the same order.
- The "two silences" paragraph rewritten around the real asymmetry: the two
  silences are unequal in what a convention can close (`norm` yes, aggregation
  no) **and** unequal in consequence, in the opposite direction (aggregation is
  annihilated by Z(·) and moves nothing; vector normalisation moves 96 of 343
  labels). This turns the concession into an argument rather than a retreat.
- "What carries into the next section": "the original discloses neither" →
  "states neither", with the asymmetry between an inferable default and a
  self-contradiction spelled out.

**§10 `10_robustness.tex` — consequential, not in the original brief.** §10 said
the `norm` silence "is not a matter of incomplete bookkeeping, because two
implementations that the published text does not distinguish between disagree
about the label of more than one report in four". Left unedited this contradicts
the concession. Reframed as a matter of *stakes*: the setting is recoverable from
the convention, we rely on it, so it is not presented as a reproducibility
failure — it is presented as a first-order choice left to an inference, which is
why we report both implementations rather than settle it by convention. §10's
description of `suslag` now reads "conceded as reasonable in §2 and §7".

**§8 `08_normalisation_artefact.tex` — same reason.** §8's opening said the two
choices behind the headline conclusion are ones "neither of which the paper
discloses". That is too blunt once the concession is on the table, and it is also
imprecise: the score normalisation is specified *twice, inconsistently*, and the
substance aggregation is not specified *at all*. Both the opening and the closing
subsection now say exactly that. "never disclosed" → "never named" where §8
refers to vector normalisation, matching §2's "The `norm` parameter is never
named".

---

## B. The new robustness results (conventions §7b)

### B1. Disjoint-bases convergent validity — §9

New subsection `sec:validity-disjoint`, "The table recomputed on disjoint bases",
with `tab:validity-disjoint`:

| specification | full bases | disjoint bases |
|---|---|---|
| SUS_tfidf-ℓ | 0.6817 | 0.4125 |
| SUS_dens | 0.6663 | 0.4112 |
| SUS_tfidf-L2 | 0.3217 | 0.1799 |
| ratio, dens : L2 | 2.07× | 2.29× |

Also records that the restricted QUANT correlates 0.8738 with the full one (so it
is a materially different criterion, not a cosmetic variant), and that the
vocabulary went 154 → 140 while QUANT was cut to `percentages` and
`large_numbers`.

Read exactly as conventions §7b directs: every absolute correlation falls, so the
contaminated figures are upper bounds and the level of convergent validity is
moderate rather than strong — but the overlap does not manufacture the ranking,
and the relative distance is slightly *larger* once it is removed.

- Deleted "One computation would settle it… We have not run it. That is a gap in
  the evidence and the first thing a reader is entitled to ask us for."
- The refuted rescue argument ("the overlap inflates every row alike") is **not**
  reinstated. The subsection that refutes it is kept intact, and the new
  subsection is explicitly the thing that settles what the argument cannot.
- Added an honest note that the preceding subsection predicted the *wrong sign*:
  it argued the overlap should have widened the gap, and in fact it was
  compressing it. Stated as the reason for running the computation.
- Note the 0.4125 − 0.4112 = 0.0013 gap reinforces §9's existing reasoning that
  the criterion separates the length-normalised pair from L2 but does not
  separate the pair; cross-linked to `sec:validity-adopted`.

### B2. Frameworks-excluded temporal check — §11

New table `tab:temporal-frameworks` **inside** `sec:temporal-mandate`, as the
first of the two counterweights:

| | full vocabulary (154) | frameworks excluded (120) |
|---|---|---|
| mean density 2018 → 2024 | 4.87 → 7.45 | 4.43 → 6.35 |
| ESGSI slope | −0.205/yr | −0.155/yr |
| p | 1.7e-8 | 2.5e-5 |

Framed as conventions §7b requires — a **bound on the definitional component**,
not a refutation of the confound. The paragraph after the table states in terms
that it does not dispose of the mandate confound, because a regime elicits
substantive disclosure as well as its own acronyms and that part is not separable
by any vocabulary edit, "because the words involved are exactly the words a firm
that had genuinely changed its conduct would also use".

Replaced the old first counterweight (the 7.50 % scale argument), which was
weaker and which the same paragraph then had to walk back. The 7.50 % figure and
the frameworks-share drift 10.15 % → 21.87 % remain in the preceding paragraph,
where they belong.

Other §11 consequences: the `tab:temporal-threats` row for post-2018 referents
moves from "None available within the design / Not excluded" to the recomputation
and "Bounded, not excluded"; "No design in this paper can do better" → "no design
can do better *than that bound*"; the closing summary now says sign and
significance "survive at roughly three quarters of magnitude when every
reporting-framework term is struck from the vocabulary".

### B3. §12 had to move with §9 (handoff item 7)

§12 was the section that had *originally* argued the rescue §9's reviewer
refuted — "contamination inflates every specification rather than one" — and also
still said "we have not re-estimated them against a criterion stripped of the two
affected groups". Both are now wrong and they contradicted §9 directly. §12's
QUANT subsection is rewritten to report the same disjoint-bases numbers at the
same strength, to say in terms that the old argument does not hold and that only
the recomputation supports the ranking, and to relocate the "most serious
limitation in this paper" claim onto the failure recomputation *cannot* touch:
QUANT and SUS read the same extracted zones with collinear denominators. One
limitations-table row amended, one added.

---

## C. Verified factual errors (handoff 1c, 1d, 1e)

**1c — §5, the collapse guard.** Confirmed against
`scripts/build_lexicons.py:79-80`: `lemmatized.setdefault(lemma, term)` runs
unconditionally after the term is appended to `collapsed` for a warning.
Confirmed `diligence` is entry 34 of the 154 in
`metadata/esg_terms_lemmatized.txt`, and that bare `agreement` and `transition`
are absent. §5 now describes the mechanism as advisory-plus-manual: the script
warns and adds the term anyway; removal is a separate human decision; it was
taken twice (*paris agreement*, *just transition*, both re-entered as raw-text
patterns in `_EXTRA_PATTERNS`) and declined once (*due diligence* → *diligence*,
4,330 corpus occurrences). Closes with what the step does and does not guarantee.

§4 had inherited the same wrong wording ("the two collocations that §5 explains
are *withheld* from the 154-term list") and is corrected in matching terms; it
also now distinguishes those two phrases from the four precision collocations for
polysemous heads, which were a separate ambiguity.

**1d — §13, the label count.** Verified against `results/metrics/results.csv`:
`Etiqueta` is *Potential ESG-washing* 171 and *Likely Genuine* 172, so 172 is the
count of the other label; `min |ESGSI| = 0.0006`, so no document is inside the
rounding interval. The passage was false in every particular. Deleted and
replaced with the verified statement that counting strictly positive rounded
ESGSI values returns 171, exactly the label count.

**1e — deduplication wording.** §3 was already correct. Checked every other
section: §13 says "MD5 content-hash dedup on `clean_text`" in the pipeline table
and "an MD5 hash of `clean_text`" in the determinism subsection — both correct.
No section inherited the loose "hash over extracted text" wording.

**Handoff item 5 — the 878-of-880 zone reproduction.** Confirmed absent from
every section, including §5. §5 and §12 refer to the audit's 880-zone scope and
to the 147-term gap, both of which are permitted; neither claims the zone-count
reproduction.

---

## D. Consistency sweep

**Cross-reference style.** Adopted `\S\ref{sec:...}` for section references
everywhere; `Table~\ref{}`, `\eqref{}` unchanged. 81 bare `\ref{sec:...}`
converted across §1 (1), §3 (5), §4 (12), §5 (3), §6 (16), §7 (9), §12 (22),
§13 (13). §2, §8, §9, §10, §11 were already conformant. Verified programmatically
that no bare section reference remains.

**Terminology.** Every remaining occurrence of "normalisation" in all thirteen
files is now qualified as *vector* or *score* normalisation, or is a quotation
from the original, or is the meta-mention in §2 of the original's own use of the
word. Fixed: §6 "the normalisation applied" → "the scaling applied to the term
vector", naming vector normalisation explicitly; §10 four bare uses inside the
attribution-bound argument → "vector normalisation"; §12 "it is normalisation
that makes them comparable" → "score normalisation"; §13 "a normalisation
mismatch" → "a lemmatisation mismatch" (which is what the HEDGE defect actually
was). The §7/§8 boundary was checked line by line and is clean.

**House style.** "Weak Modal / Strong Modal" in §5 → "WeakModal / StrongModal",
matching §6, §13 and the literal category strings in
`metadata/RAW_LM_dictionary.csv` (verified: Uncertainty 767, WeakModal 27,
StrongModal 19, Constraining 432 = 1,245 rows of 9,752).

The label string appears in two spellings for a real reason — the original writes
"Potential ESGwashing", our pipeline writes "Potential ESG-washing" — so rather
than flatten them, §6 now states the distinction once at the point of definition
and points at §13 for the output string. §2, §6, §8 use the original's spelling
when quoting its rule; §10, §12, §13 use ours.

**Two parameter tables.** `tab:index-parameters` (§6) and `tab:repro-params`
(§13) were checked value by value and agree on all fifteen shared entries. Both
survive, and a directional rule is now stated in both so they cannot silently
diverge: §6 is definitional for the index stage, §13 is a consolidation that adds
extraction and preprocessing, and "where the two would disagree, §6 governs and
this one is in error". Added the ε = 1e-6 polarity smoothing to §13's SEN row,
which the consolidation had omitted.

**Reference integrity.** `campbell1959` was cited by §9 and missing from
`references.bib`. Added — Campbell & Fiske (1959), *Psychological Bulletin*
56(2), 81–105, DOI 10.1037/h0046016 — an entry I can verify. No other key was
missing and no entry is unused. Cited keys and bib keys now match exactly.

**Structural checks (all pass).** 0 unresolved `\ref`; 0 duplicate labels;
`main.tex`'s thirteen `\input` lines match the thirteen files on disk exactly
(01–12 in the body, 13 after `\appendix`); no `\begin`/`\end` mismatch; brace
counts balanced in all thirteen; table column counts internally consistent in
every table. No LaTeX toolchain is installed in this environment, so this is
static checking, not a compile.

---

## E. Flow

- §11's subsection "Robustness to the substance specification" was word-for-word
  §10's *section* title. Retitled "Sensitivity of the series to the substance
  specification".
- §2's closing paragraph and its "Consequence" subsection both explained why the
  paper carries three substance specifications. Merged into one.
- §4's yield table is on 344 PDFs while everything downstream is on 343, which a
  referee would read as a discrepancy. §4 now states why in one sentence and
  points at §3.
- §9's promise in its opening ("the two respects in which it is not independent
  are stated below with their magnitudes") now also says which one gets removed
  by recomputation and which does not.
- §11's mandate subsection said the objection "is not disposed of" and then
  produced a bound; it now says the definitional part can be bounded and the
  objection survives the bound, so the reader is not surprised by the table.
- §1: added the standalone-report panel evidence that handoff item 13 sanctioned,
  under its non-negotiable wording constraint — "exactly one firm is represented
  by standalone sustainability reports across the window" (Sanofi-Aventis, all
  seven years), plus "43 of the 49 firms are represented by a single document
  type across the whole window". Verified from `results.csv`. Nothing anywhere
  says what any firm *published*; the paragraph states explicitly that our
  records hold the document selected, not what was available.

No sound prose was rewritten for style, and no technical section was allowed to
drift into the abstract / introduction / literature / discussion / conclusion
territory, which remain commented out in `main.tex`.

---

## Verification performed

Reproduced from the repository, exactly, before letting the figure stand:

- Corpus: 343 rows × 17 columns; `Etiqueta` 171 / 172; min |ESGSI| = 0.0006;
  yearly ESGSI means 0.563, 0.471, 0.161, 0.007, −0.121, −0.407, −0.674;
  document types 235 / 101 / 7; country counts.
- Lexicons: 154 canonical terms, 154 lemmatised entries, 85 unigrams / 62 bigrams
  / 7 trigrams, thematic blocks 14/18/17/12/34/25/34 (so the frameworks block is
  34 and 154 − 34 = 120, as conventions §7b states); 1,263 hedge forms; 189
  stopwords; 1,245 of 9,752 L&M rows.
- §7's idf table: 43 terms below idf 1.2 holding 78.35 %, minimum document
  frequency 282 (so "> 280" is right); 46 terms above idf 2 holding 2.80 %;
  *fair condition* idf 4.8947 on 8 occurrences, *climate* 1.0088, ratio 4.852.
- §9/§11/§12's overlap: the 14 shared entries, 7.50 % of occurrence mass.
- §11's driver table in full: 27,906 → 63,654 lemmatised tokens (+128.1 %);
  1,382.7 → 4,832.0 mentions (+249.5 %); density 4.8739 → 7.4483 (+52.8 %);
  BREADTH 28.99 → 41.73 (+43.9 %); zones 178.5 → 275.2 (+54.2 %); distinct
  entries 68.65 → 105.45.
- §4's yield table in full: 74,004 zones; 215.1 mean, 152.9 sd; 26,399,487 words;
  76,743 mean per document; corpus-wide extraction density 3.658.
- §12's token bases: 14,398,434 lemmatised and 26,184,590 raw on the deduplicated
  corpus — 14.4M, 26.2M, ratio 1.82, so "roughly 1.8 times larger" holds.
- §5's collapse figures: *agreement* 15,459, *diligence* 4,330, *esr* 6,282.
- The deduplication: MD5 over `clean_text` collapses 344 → 343.

---

## Could not reconcile — needs a human decision

**1. §11's extraction-stage density pair, 2.61 → 3.93 per 100 words. Not
reproducible; I removed the two numbers.** This mattered enough to act on: the
same artefacts (`data/chunks_lexical/**/*.json`) reproduce §4's entire yield
table to the last digit, so the source and the method are sound, and yet no
reading of the extraction-stage density gives 2.61 and 3.93. The three candidate
readings are:

| reading | 2018 | 2024 |
|---|---|---|
| pooled keyword count ÷ pooled word count | 3.04 | 4.42 |
| mean of per-report densities | 2.99 | 4.36 |
| median of per-report densities | 2.82 | 4.38 |

The claim the sentence is making — that extraction-stage density rises across the
window, so the intensive margin dominates the extensive one — is **true under all
three**, so I kept the claim and stated it qualitatively ("rises monotonically
across the window rather than falling, and it does so under both readings of that
quantity"). A number I have shown to be unreproducible cannot stand in a paper
about documentation failure, and per `conventions.md` I will not silently
substitute my own. **The author should reinstate a specific pair.** The mean of
per-report values (2.99 → 4.36) is the reading consistent with the rest of
`tab:temporal-drivers`, which is per-report means throughout.

**2. §9's shared-entry counts were on the pre-deduplication basis. Corrected.**
The published pair 67,197 of 896,499 reproduces exactly on n = 344; on the
deduplicated n = 343 it is 67,194 of 895,996. `conventions.md` §5 is categorical
that 344 is never the corpus size, so I corrected it and added "over the
deduplicated corpus". The derived 7.50 % is unchanged at either basis, so nothing
in §9, §11 or §12 moves. Flagging because these two raw counts are not in
`conventions.md` and the correction is mine.

**3. `esr` idf: `handoff.md` item 12 says 2.53; the data give 2.5232.** §12 says
2.52 and is right. Kept 2.52; the handoff figure appears to be a mis-rounding.
(Everything else about the entry checks out: 6,282 occurrences across 74 of 343
documents, and it is the largest single entry in the high-idf group.)

**4. `conventions.md` §7 disagrees with itself on one value.** The
ESGSI(SUS_tfidf-L2) series lists 2023 as −0.371; the monotonicity note four lines
later writes "2023 (−0.370)". §11 uses −0.371, following the series. Worth
correcting in `conventions.md` so the next pass does not inherit the ambiguity.

**5. `conventions.md` §7b and §11 round the same density rise differently** —
"+53 %" there, "+52.8 %" in `tab:temporal-drivers`. The precise figure is 52.8 %
(4.8739 → 7.4483). I gave the new table levels only, no percentage column, and
the prose reads "falls from +52.8 % to roughly +43 %", so the two roundings never
appear side by side as if they were different quantities.

**6. The sampling frame still has no stated provenance (handoff 10b).** Nothing
in the repository explains why these 49 firms and these seven countries. §1
correctly names it as a separate sampling-frame decision rather than a
consequence of the panel design, and says nothing further. It cannot be inferred
and must not be invented. **A human must supply it**, and a referee will ask.

**7. §13's reproducibility gap is disclosed but still open.** The three-
specification SUS system, the `Breadth` column and the deduplication step exist
only as uncommitted changes on top of `44114d0`, so the named commit is necessary
but not sufficient to regenerate `results.csv`. §13 says so plainly, which is the
right handling for now, but the fix is a commit, not a sentence.

**8. Placeholders.** Abstract, introduction, literature review, discussion and
conclusion remain unwritten and commented out in `main.tex`, per the brief. The
bibliography style is `elsarticle-harv`, which is not present in this repository
and must be available at build time.

**9. Not compiled.** No LaTeX distribution is installed here. All structural
checking was static.

---

## Assessment: is §8's central claim adequately supported?

Yes, and the support is unusually well insulated, because §8 does not depend on
our pipeline for the part that carries it.

§8's claim is that the original's headline conclusion — ESG-washing predominantly
absent — is an artefact of a score-normalisation choice and a substance
specification rather than a property of the disclosure. Its first two subsections
establish this from the original's own published tables: the components' reported
minima are non-negative and their standard deviations are 0.1321 and 0.0853, so
neither was z-scored; the reported index mean is exactly the difference of the two
reported component means, which a z-score difference cannot be, since that is zero
by construction; and the reconstruction holds in all twenty-eight published rows
across Tables 2–6. Then the analytical step: under z-scores the index is centred
on its own labelling threshold before any document is read, so "predominantly
negative" is not a result the estimator can return. That is Move-1 evidence
throughout. It would survive the deletion of our corpus, our vocabulary and our
code, and §8 says so.

What the surrounding sections add is the mechanism and the bound on the reading.
§7 supplies why the vector-normalisation choice is first-order and §10 measures it
at 96 of 343 labels against 6 for term weighting, with a clean combinatorial
attribution of at least 90 of the 96 to normalisation alone. §9 now supplies the
one thing that was missing: the specification preference is no longer resting on a
correlation table the section itself had conceded was contaminated and undefended.
§8's own §3 — the 2×2 factorial on our corpus — is correctly marked as
corroboration and is explicitly disposable; the section states that a reader who
rejects it entirely still has the claim.

Two residual weaknesses, neither fatal and both now disclosed rather than papered
over. First, the *level* of convergent validity is moderate on clean bases (0.41
against 0.18), not strong; §9 and §12 now say so in the same words. The ranking,
which is what §8 needs, survives and widens. Second, the specification choice was
made non-blind by us, on a criterion we built, over zones our own extraction stage
selected — §12 identifies this as the most serious limitation in the paper and
does not soften it. Neither weakness touches §8's first two subsections, which are
where the claim actually lives.

The main risk to §8 was never evidential. It was the §2/§7 inconsistency on
`norm`, which would have let a referee argue that a paper willing to run an entire
experiment on a reporting convention cannot also charge that the same convention
left the work irreproducible. That charge is now answered by conceding the
inference and relocating the reproducibility argument onto the aggregation rule —
which is genuinely unanswerable, because it is a modelling step with no library
default to fall back on, and which §7 then shows is harmless on our index while
`norm` is not. The concession costs the paper nothing and the asymmetry it exposes
is a better argument than the one it replaces.
