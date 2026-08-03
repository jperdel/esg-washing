# The case against the lexicon-robustness addition

Referee's brief, ordered by damage. Every number below was recomputed from the
repository with `/c/Users/Jorge/anaconda3/envs/esgwashing/python.exe`; scripts in
the session scratchpad. Where an attack failed on inspection I say so — those
paragraphs are marked **ATTACK FAILS** and the defender should not waste time on
them.

Two preliminaries in the drafts' favour, stated up front so the rest is not read
as scattergun:

- **The headline figures are real.** Recomputed from the shipped result files:
  mean `SUS_density` 5.8972 / 6.5162 / 6.5327; index correlations 0.987972 and
  0.999775; flips 17 and 1; flagged 171 / 164 / 165. Every figure in
  `tab:robustness-lexicon` and in the §12 fragment reproduces exactly.
- **The classification rule is clean.** The stated rule in
  `metadata/esg_terms_sectorial.txt` — *"aparecen en 12 empresas o menos, o más
  del 50 % de sus ocurrencias están en una sola empresa"* — partitions the 143
  measured candidates with **zero** misclassifications against
  `term_dispersion.csv`, and there are **zero** Pareto-dominance violations (no
  excluded term beats an included term on all five statistics). See §9 below for
  where the claim about the rule nevertheless breaks.

---

## 1. The experiment described is not the experiment that was run. The extraction stage moved. [DECISIVE]

`_draft_10_lexicon.tex`, closing section, states the design in terms:

> "Both experiments run on the same 343 documents and the same extraction
> architecture (\S\ref{sec:extraction}): what varies is which terms are scored
> inside zones already selected as ESG-relevant, **not how those zones are
> found**"

This is false as implemented, and the shipped result files prove it without any
reference to the code.

`SEN`, `HEDGE` and `QUANT` are computed by the Loughran–McDonald dictionary and
the `QUANT` regex families over the extracted text. **None of them touches the
ESG vocabulary.** If the extracted zones were held fixed, they would be
bit-identical across the three runs. They are not:

| compared runs | docs with a different `SEN` | different `QUANT` | different `HEDGE` |
|---|---|---|---|
| 154 → 289 | **343 / 343** | 328 / 343 | 337 / 343 |
| 289 → 289+22 | 105 / 343 | 27 / 343 | 68 / 343 |

Mean `SEN` moves from **+0.004682** to **−0.017591**; the largest single-document
shift is 0.1453. `ESGSI = Z(SEN) − Z(SUS)`, so the sentiment term — a term the
two "vocabularies" do not differ on at all — moved on **every document in the
corpus** between the two columns of `tab:robustness-lexicon`.

The mechanism is in the uncommitted diff. `src/lexical_document_filter.py`:

```python
def _load_esg_terms() -> list[str]:
    terms = _read_terms(_METADATA_DIR / "esg_terms.txt")
```

`metadata/esg_terms.txt` is the file that went from 154 to 296 natural-form
entries. It feeds `_ESG_KW_RE`, which scores every paragraph at the 1.0
keyword/100-word threshold and decides which zones exist. The same diff also adds
seven new `_EXTRA_PATTERNS` (`disabilit\w+`, `fatalit\w+`, `philanthrop\w+`,
`ergonomic\w*`, `absenteeism`, `local\s+communit\w+`, `equal\s+opportunit\w+`).
`data/chunks_lexical/` was re-extracted 2026-08-03 18:05, after the vocabulary
was rewritten at 17:30–17:33 and after `results/metrics_baseline_paper/` was
computed at 17:33.

Consequences that reach the page:

- **The corpus grew.** Current extraction: **16.42 M** lemmatised tokens.
  §12 states the corpus is **14.4 M** lemmatised tokens. That sentence is now
  wrong, and it is wrong because of this addition.
- **"+10.5 %" is a net of two confounded effects working in opposite
  directions** (more terms in the numerator, more text in the denominator). Held
  on one extraction, the 154-term vocabulary scores mean density **5.1682**, not
  5.8972, and the vocabulary effect is **+26.1 %**, not +10.5 %. The draft's
  gloss — *"so the added terms match text rather than nothing"* — is inferred
  from a number that understates its own effect by a factor of 2.5.
- **The Danone figure is wrong for the same reason.** Vocabulary-only, the
  sectoral terms raise Danone's mean `SUS_density` by **0.354**, not 0.206.
  (Enel 0.111 and Schneider 0.099 survive.) The draft's own illustration of the
  sectoral hazard is understated by 70 %.
- **§12 says this stage is never varied.** *"Every result is conditional on the
  zone extraction of §4, whose free parameters … were set by judgement and never
  varied … we do not test sensitivity to it."* It has now been varied, silently,
  as a side effect of a robustness check about something else.

**The one thing that survives.** Recomputing 154 vs 289 on a single fixed corpus
(current extraction, `SEN` held at its value on that corpus) gives **18 flips
(5.2 %)** and **r = 0.9882** against the reported 17 (5.0 %) and 0.9880. So the
headline robustness numbers are approximately right — *by luck, not by method*.
The defender may keep the numbers. He may not keep the sentence that describes
how they were obtained.

**Remedy:** recompute the 154-term column on the current extraction, or revert
`esg_terms.txt` and `_EXTRA_PATTERNS` and re-extract for the 289 column. Then
delete the "same extraction architecture" sentence, which will be false either
way unless the extraction regex is decoupled from the scoring vocabulary.

---

## 2. The addition breaks §13 and the conventions file. [SEVERE, and unforced]

`paper/sections/13_reproducibility.tex` line 32 names as an output artefact:

> `metadata/esg_terms_lemmatized.txt` (154 entries)

That file now contains **289** entries. Line 33 names
`results/metrics/results.csv` (343 rows) as the measurement output; that file is
now the **289-term run** (mean density 6.5162, flagged 164/343). `conventions.md`
§7 declares it "authoritative for every index value" and §5 declares its own
figures "verified against `results/metrics/results.csv`". Not one canonical
figure in §7 can now be reproduced from the file §7 names.

The paper's adopted specification survives on disk only in
`results/metrics_baseline_paper/`, a directory no section mentions. A referee who
clones the repository and follows §13 gets the robustness vocabulary and the
robustness results, and no way to know it.

This is not a matter of the drafts' wording — it is a state the addition left the
repository in — but it is the drafts that make it publishable, because they are
what turns a scratch experiment into a claim in the paper. It must be fixed
before any of this ships.

---

## 3. The hierarchy table is rhetoric with a table's authority. [SEVERE — recommend CUT]

`tab:robustness-hierarchy` puts 28.0 %, 5.0 % and 0.3 % in one column and the
prose draws the conclusion:

> "Vector normalisation moves nearly six times as many labels as adding 135 terms
> to the vocabulary, and two orders of magnitude more than the sectoral question.
> That sharpens this section's claim rather than qualifying it"

Four objections, each sufficient on its own.

**(a) The three rows are not scaled the same way, and the ranking reverses when
they are.** Label flips are a function of how far the index moved. Normalising by
the index disturbance the drafts themselves report:

| choice varied | flips | r | 1 − r | flips per unit (1 − r) |
|---|---|---|---|---|
| Vector normalisation | 96/343 | 0.6964 | 0.3036 | **316** |
| Vocabulary 154 → 289 | 18/343 | 0.9882 | 0.0118 | **1,532** |
| Sectoral +22 | 3/343 | 0.9998 | 0.0002 | **15,239** |

Per unit of index disturbance, the vocabulary change is ~5× *more* label-efficient
than vector normalisation and the sectoral terms ~48× more. The table establishes
"large perturbations move more labels than small perturbations", which is a
tautology, and presents it as a finding about which methodological choice matters.
§10 already knows this — *"imperfectly correlated indices disagree first about
documents lying near the cut-off"* — and its own defence of the 6-vs-96 contrast
(*"whatever the threshold does to 6, it does to 96"*) works precisely because
those two came from one factorial design on comparable perturbations. It does not
transfer to three perturbations of different kind and three different magnitudes.

**(b) The caption is not true of row 3.** It reads *"measured against the
specification adopted for that choice"*. Rows 1 and 2 are measured against the
paper's adopted 154-term index. Row 3 is measured against the **289-term
vocabulary, which the paper does not adopt** — a robustness instrument the same
subsection says is "not a candidate for re-selection". "The specification adopted
for that choice" is a formulation that exists to paper over exactly this. Against
the adopted specification, 154 → 289+22 moves 18 labels, not 1.

**(c) The check cannot fail, and a random perturbation reproduces it exactly.**
I ran 300 random deletions from the 154-term list at each size:

| terms dropped at random | median flips | mean r vs adopted index |
|---|---|---|
| 10 | 6 | 0.9965 |
| 20 | 12 | 0.9925 |
| **30** | **17** | **0.9882** |
| 50 | 25 | 0.9760 |
| 75 | 35 | 0.9564 |

Deleting **30 of 154 terms at random** produces r = 0.9882 and a median of **17**
label flips — the reported result of the 289-term expansion, to the digit. The
expansion's behaviour is not distinguishable from that of an arbitrary
perturbation of the same magnitude. A check that returns the same answer for a
carefully audited expansion and for a coin flip is not measuring the audit.

**(d) The paper's own vocabulary perturbation is missing from the table.**
§7b(B) — sanctioned, already reported in §11 — strikes the 34-term reporting-
frameworks block and moves the trend slope from **−0.205 to −0.155/yr**, a 24 %
change from a 22 % deletion, against the 5 % change the drafts get from an 88 %
addition. If the hierarchy table is meant to rank vocabulary choices, the paper's
own largest vocabulary effect belongs in it, and it would sit above the row the
drafts use to dismiss the objection.

*(Caveat for the defender: my reconstruction of that deletion on the current
extraction gives slope −0.193 and 10/343 flips, not −0.155. I could not reproduce
§7b(B)'s −0.155 on the present corpus. That discrepancy is itself a symptom of
point 1 and should be checked before either number is used.)*

---

## 4. Insensitivity was bought by the acceptance rule, and the achievable range can be computed. [SERIOUS]

The mandate asks whether the high correlation is near-inevitable. The honest
answer is: **not literally inevitable, but confined to a window the selection
rule fixed in advance.**

Write the expanded density as `A + D`, `A` = the 154-term density, `D` = the
135 added entries. Measured:

- σ(A) = 1.8269, σ(D) = 0.4309, so σ(D)/σ(A) = **0.2359**
- Minimising `r(A, A+D)` over every possible value of `r(A, D)` at that size:
  **the correlation could not have come out below 0.9718.**
- Observed: 0.9791. **The entire window the data could move the answer in is
  0.028 wide.**

So a reader told "r = 0.9880, the index is robust to the vocabulary" is being
shown a statistic whose achievable range was [0.972, 1.000] before a single
document was read. And the width of that window is σ(D) — which is precisely what
the acceptance rule controls. Terms were admitted on *dispersion*: appearing in
many documents, many companies, many countries, with no single firm holding much
of the mass. A term with those properties contributes a near-constant per-document
increment. Compare the two increments actually built:

| increment | mean | sd | CV |
|---|---|---|---|
| 135 accepted entries (selected **for** dispersion) | 1.3479 | 0.4309 | **0.320** |
| 19 sectoral entries (selected **for** concentration) | 0.0244 | 0.0538 | **2.204** |

Per unit of mass the rejected terms are **6.9× more variable** than the admitted
ones. The rule admitted the terms that arithmetically cannot move a
cross-sectional density ranking and set aside the terms that can. The check then
reports that the ranking did not move.

**Where this attack is weaker than it looks, and the defender should be told so.**
The added terms are **19.5 %** of the 289-term numerator — a fifth of the measure
is new material, which is not nothing, and 0.9791 sits meaningfully above the
0.9718 floor. The exercise is not vacuous. What it is not is *evidence about the
vocabulary*: it is a measurement of how much variance the selection rule let
through.

**Related, and cheaper to fix: the denominator is chosen to flatter.** The
sectoral terms are described as carrying *"2.0 % of the mass of the 147
candidates"*. Against the quantity that actually matters — the substance
numerator the index is built from — they are **0.37 %**, and the mean density
they move is **+0.25 %**. `conventions.md` §7: *"Always print counts with the
denominator, never a bare percentage: the whole point of the section is that a
denominator was mis-stated elsewhere."* A paper whose central claim is a
mis-stated denominator should not pick the denominator that makes a null result
look like a measurement.

Same class: `\BREADTH` "rises from 56.30 to 56.81" is offered as if it carried
information. `BREADTH` is `exp(entropy)` over the vocabulary — an *effective term
count*, bounded above by the vocabulary size. Recomputed: 35.75 at 154 entries,
56.30 at 289. It rises because the vocabulary grew. It would rise if the added
terms were random noise.

---

## 5. §5's rewrite retires a bound it has not earned. [SERIOUS]

The merge instruction is explicit about intent:

> "everything after supersedes the 'open, unresolved' verdict"

and it deletes: *"The gap stands as an open, unresolved bound on recall, not a
problem this paper claims to have closed."*

The recall bound is a property of **the adopted 154-term vocabulary**. Of the 147
audited omissions, the number incorporated into the adopted vocabulary is
**zero**. The paper still reports every figure on the 154 terms. The recall of the
instrument the paper uses is numerically, exactly unchanged by everything in these
four drafts.

What *was* settled is the sector-bias question — a different objection, which
entered §12 only as the stated reason the omissions had not been fixed. Draft 12
gets this exactly right (*"converts one item on this list from an unresolved
omission into a design choice we disclose"*), and draft 05's own last paragraph
concedes it (*"What the audit leaves standing is the recall bound of the paragraph
above … not the sector-bias question, which is now measured"*). **ATTACK PARTLY
FAILS**: the content is honest. The defect is structural — the merge instruction
deletes the verdict sentence and replaces a paragraph that ended in "unresolved"
with one that ends in a specification statement, so a reader who does not parse
the fourth paragraph closely finishes the subsection believing the audit gap was
addressed. The bound survives in the prose and dies in the rhetoric.

**Remedy:** keep a verdict sentence. Something with the force of: *the adopted
vocabulary incorporates none of the 147; its recall bound is unchanged, and what
the dispersion analysis retires is the sector-bias objection, not the bound.*

---

## 6. §5 fails to say what shipped for `social`, against an explicit instruction. [SERIOUS]

`conventions.md` §7c, in bold:

> **`social` ships as an inclusion list, not an exclusion list.** … the
> vocabulary architecture has no exclusion mechanism … **Say what shipped; do not
> describe an exclusion mechanism that does not exist.**

Draft 05 says:

> "Of *social*'s occurrences 60.3 % sit in unambiguously ESG context and only
> 8.7 % unambiguously outside it, concentrated in three enumerable strings … **so
> the valid uses are a long tail and the invalid ones are the short list.**
> *Compliance* inverts that … **No short exception list exists, so only
> collocations that positively identify governance usage are admitted.**"

The paragraph states the exclusion case for `social`, states the inclusion case
for `compliance`, and then says what was implemented **for `compliance` only**. A
reader finishes it believing `social` was handled by exclusion. It was not: the
289-term vocabulary contains no bare `social` and no exclusion mechanism, only
sixteen `social …` collocations.

This is not pedantry, because `social` is the largest candidate by mass and the
cost is quantified in the author's own working file
(`metadata/colocaciones_analisis.md`):

> "se pierde la cola larga de usos válidos de `social`, que según este mismo
> análisis es **la mayor parte** de sus casi 50.000 ocurrencias"

Order-of-magnitude check: the shipped `social …` entries match **11,526** times in
the extracted lemmatised corpus, against **29,041** occurrences classified
unambiguously ESG in the full text — roughly **40 %** captured. (Different text
bases; indicative only.) `compliance` fares better, ~79 %. The single largest
term in the expansion enters at about two fifths of its valid mass, and the draft
does not say so.

---

## 7. "103 accepted" is wrong, and it is wrong for a reason the paper elsewhere makes a showpiece of. [MODERATE, trivially fixable, embarrassing if it ships]

`conventions.md` §7c, in bold: **"102, not 103."** `power purchase agreement` was
meant to replace `ppa` and "was lost to a generation-script bug … do not write
103." The block header inside `metadata/esg_terms.txt` — the file §13 publishes —
says `102`. `_new_accept.txt` contains 102 lines.

Both drafts write 103:

- `_draft_10_lexicon.tex`: "**103** broadly dispersed terms were accepted"
- `_draft_05_audit.tex`: "on that base **103** are broadly dispersed"

The paper would contradict its own published vocabulary file. Worse, §5 spends
three paragraphs on precisely this failure mode:

> "A term whose surface form does not survive preprocessing becomes a dictionary
> entry that can never match anything, and **it fails without warning**, because a
> zero count is indistinguishable from the term's genuine absence."

A term was lost without warning from the robustness vocabulary, and the drafts
report the pre-loss count. If this addition ships, it ships with a live instance
of the defect the section it joins is about.

*(Note also that `conventions.md`'s own arithmetic is broken: it writes
"102 + 18 + 22 = 143". 102 + 18 + 22 = 142. The reconciliation is 103 classified,
102 shipped. Whoever amends the drafts should fix the conventions file rather
than copy either number.)*

---

## 8. The "sectoral" bin is heterogeneous, three of its measurements are wrong, and the draft's claim about what those terms do is false for several of them. [MODERATE]

The draft asserts, of all 22:

> "At firm level they do exactly what the objection to them predicts, rewarding
> an industry rather than a disclosure"

and illustrates with Danone/*nutrition*, Enel/*energy storage*, Schneider. Those
three fit. The bin also contains, with the dispersion table's own figures:

| term | occ | docs | firms | countries | max firm share |
|---|---|---|---|---|---|
| `executive compensation` | 647 | 118 | **29** | 7 | 0.522 |
| `board oversight` | 22 | 17 | 8 | 4 | 0.364 |
| `iso 37001` | 69 | 35 | 8 | 3 | 0.275 |
| `cop 28` | 15 | 12 | **10** | 4 | 0.200 |
| `sf6` | **0** | **0** | **0** | 0 | 0.000 |
| `cop28` | **0** | **0** | **0** | 0 | 0.000 |

Concentration in this data has at least four causes, and the bin does not
distinguish them: industry (`nutrition`, `food safety`), **house style**
(`executive compensation` is used by 29 of 49 firms; `board oversight` is
universal governance language), **calendar** (`cop 28` is a 2023 conference, not
a sector), and **measurement failure**.

The measurement failures are checkable and they changed the classification:

| term | dispersion table | actual, in the extracted lemmatised corpus |
|---|---|---|
| `sf6` | 0 occ / 0 docs / 0 firms | **472 occ / 48 docs / 17 firms / 5 countries** |
| `cop28` | 0 occ / 0 docs / 0 firms | **76 occ / 25 docs / 18 firms / 6 countries** |
| `board oversight` | 22 / 17 / 8, max 0.364 | **105 / 77 / 30 firms / 6 countries, max 0.143** |

`board oversight` on its true statistics — 30 firms, max firm share 0.143 —
**passes** the stated rule and should have been accepted. And `sf6` is sulphur
hexafluoride, a greenhouse gas; §13's own defect table lists it by name as a term
the old `is_alpha` filter had been silently deleting. It has been excluded from
an ESG vocabulary as "sector-specific" on a measured frequency of zero that is
actually 472.

Lemmatisation losses compound it. 22 natural-form sectoral entries yield **19**
TF-IDF entries: `b corp`, `iso 37001` and `cop 28` are lost outright, and two
become artefacts of exactly the class §12 lists as a limitation —
`better cotton` → **`well cotton`**, `living income` → **`live income`**
(§12 already names `live wage` and `employee wellbee`). The drafts say
"±22 sectoral terms move one". The perturbation actually applied is 19 entries,
three of them lost and two of them mangled.

Finally, `fair trade` was **accepted** as broadly dispersed and `fairtrade` was
**excluded** as sectoral. Same concept, opposite verdicts, decided by a space.

---

## 9. "Classified on them, not on a judgement" is false for the split it is attached to. [MODERATE]

`_draft_10_lexicon.tex`:

> "Each candidate was measured … on five statistics … and classified on them,
> **not on a judgement about what an ESG term ought to be**."

**ATTACK PARTLY FAILS**: for the sectoral/non-sectoral split this is true, and
verifiably so (0 misclassifications, 0 dominance violations). Two qualifications
survive:

- **Only two of the five statistics enter the rule** (`empresas` and
  `max_empresa`). Occurrences, documents and countries are reported and unused.
- **The accept/polysemous split is pure semantic judgement**, and it is the split
  the sentence is nearest to. `ocean` (2,361 occ, 236 docs, 47 firms) →
  polysemous; `species` (1,613 / 175 / 43) → accepted. `wellbeing` (1,072) →
  polysemous; `mental health` (914) → accepted. No statistic separates these; a
  view about what the word means does.

Related mis-description in the same sentence: *"18 frequent but polysemous terms
were admitted **only in collocation**, of which only `compliance`, `social` and
`proxy` in fact required it."* Fifteen of the 18 shipped **bare** — the block
header in `esg_terms.txt` reads *"términos frecuentes y dispersos, admitidos
sueltos: 15"*. A term admitted bare was not "admitted only in collocation". The
sentence contradicts itself and misdescribes the shipped vocabulary.

---

## 10. §11's claim is contradicted by §11. [MODERATE]

`_draft_11_lexicon.tex`:

> "**Nothing in this section turns on where the boundary of the vocabulary was
> drawn**: the paper's specification remains the 154-term list, and the series
> would read the same under either."

and closes:

> "Robustness to the lexicon and identification of what the lexicon counts are
> different questions. **The first is settled here**"

Both overreach, and §11 already contains the counter-example. §7b(B), which §11
must report inside the mandate concession, moves the boundary the other way — out
34 reporting-framework terms — and the slope goes **−0.205 → −0.155/yr**. The
series demonstrably *does* turn on where the boundary is drawn; it turns on it
about five times as much as the addition the new subsection reports, and the
paper says so eight paragraphs away. Two subsections of §11 will assert opposite
things about the same question.

The direction asymmetry is the substance: the drafts move the boundary **outward
only**, with terms **selected on this corpus** for the property (dispersion) that
guarantees the smallest movement. That is one tail of a two-tailed question. On
one corpus, a random **deletion** of 30 terms — 19 % of the list — reproduces the
expansion's entire effect (§3c above). "Settled" is not available on this
evidence; "bounded in one direction, on this corpus, for a class of terms chosen
for their stability" is.

Draft 11 is otherwise the most careful of the four — it volunteers the
harvested-from-FY2023 aggravation and refuses to run the two arguments together.
It needs its two strongest sentences weakened, not its argument replaced.

---

## 11. Selection effects, and whether this makes §12's concession worse. [MINOR — the mandate's premise is largely answered]

**ATTACK LARGELY FAILS.** The mandate asks whether adding a self-chosen knob
worsens §12's concession that *"the vocabulary, the extraction parameters, the
`QUANT` regex families and the specification set were all fixed by us with the
results visible, and no part of the corpus was held out."* The thresholds (12
firms, 50 % mass) were indeed picked by the author with the answers computable,
and the draft-12 fragment adds a disclosed-choice row rather than a defence,
which is the right handling. `_draft_12_sectoral.tex` is the strongest-written
fragment of the four: *"The decision is disclosed and immaterial, not necessary,
and the earlier framing … overstated what the analysis could settle."* That is a
concession, not a rescue.

What the addition *does* worsen is smaller and specific: §12's non-blind
paragraph lists knobs that were set once. This adds a knob that was set **after**
the results were known and **presented as a check on those results**. The
sentence in §12 should say so — one clause, not a new subsection.

---

## 12. Smaller items a copy-editor will not catch

- **`tab:robustness-lexicon` caption reports trend p-values to two significant
  figures** ($1.7\times10^{-8}$, $3.4\times10^{-8}$) for a design §12 concedes
  yields anti-conservative p-values (49 firms observed 7 times, no clustered
  inference). Report direction and significance; drop the precision.
- **The draft reports only index-level correlations**, where §10 reports both
  levels and explicitly warns the reader not to substitute one for the other. For
  the record the component-level figures are r(dens₁₅₄, dens₂₈₉) = **0.9906** and
  r(dens₂₈₉, dens₂₈₉₊₂₂) = **0.9998**. Here the index-level number is the
  *lower* of the two, which breaks §10's stated empirical regularity that
  *"every index-level correlation … runs above its component-level counterpart"*.
  Worth one sentence in §10, since a reader will check.
- **Three different figures circulate for the same quantity.** Draft 05 writes
  *"compliance occurs 55,988 times and social 50,090"*; `conventions.md` §7c
  gives 55,974 / 50,026 (prefix base) and 55,780 / 48,198 (whole-word base);
  `colocaciones_analisis.md` uses the whole-word base. Pick one, name it.
- **`_draft_10` and `_draft_05` each restate the classification counts in full.**
  Between the four fragments the 102/103–18–22 split is stated three times and
  the sectoral firm-level effects twice. This is ~1,800 words for one null
  result.

---

# Verdicts

### `_draft_10_lexicon.tex` — **KEEP IF AMENDED**, at roughly half length, with `tab:robustness-hierarchy` **CUT**

Required:
1. Fix the extraction confound (point 1): recompute the 154-term column on the
   same extraction, or re-extract for both. Then delete *"the same extraction
   architecture … not how those zones are found"* — it cannot be repaired by
   rewording, only by a re-run.
2. **Cut `tab:robustness-hierarchy` and the paragraph that reads it.** It is not
   scale-invariant, its caption is untrue of its third row, its ordering reverses
   under any normalisation of perturbation size, and it omits the paper's own
   larger vocabulary effect. If a hierarchy claim is wanted, it must be made
   against the adopted specification throughout and must include the §7b(B)
   deletion.
3. 103 → 102. Fix the "admitted only in collocation" sentence.
4. Report the sectoral mass against the substance numerator (0.37 %), not against
   the candidate pool (2.0 %); drop the `BREADTH` sentence or say it is mechanical.
5. Qualify the firm-level sentence: the 22 are not one kind of term, and at least
   `sf6`, `cop28`, `board oversight` and `executive compensation` do not reward an
   industry (point 8). Fix Danone 0.206 → 0.354 or drop the figure.
6. Add one sentence stating the achievable range: at σ(D)/σ(A) = 0.24 the
   correlation could not have fallen below 0.972. It costs a line and it is the
   difference between a robustness check and robustness theatre.

### `_draft_11_lexicon.tex` — **KEEP IF AMENDED** (lightest touch)

1. Delete *"Nothing in this section turns on where the boundary of the vocabulary
   was drawn"* and *"The first is settled here"*. Replace with the one-directional
   claim the evidence supports.
2. Add one clause reconciling with §7b(B): moving the boundary inward moves the
   slope five times as much as moving it outward, and both are in this paper.
3. Correct the baseline slope if point 1 is fixed (like-for-like on one corpus the
   154-term slope is −0.219, not −0.205).

### `_draft_05_audit.tex` — **KEEP IF AMENDED** (substantively)

1. Restore a verdict sentence on the recall bound (point 5). The paragraph may not
   end on the specification statement.
2. **Say what shipped for `social`** — an inclusion list of collocations, not the
   exclusion the paragraph's logic sets up, and say what it costs (point 6). This
   is a standing instruction in `conventions.md` and the draft violates it.
3. 103 → 102.
4. `tab:lexicon-context` is good and should stay; the "bounds from above and does
   not estimate it" sentence is the best writing in the four fragments.

### `_draft_12_sectoral.tex` — **KEEP IF AMENDED** (minimal; nearest to publishable of the four)

The most honest fragment. Its framing — disclosed, immaterial, not necessary,
earlier framing overstated — is what the other three should have imitated. Fixes:
the "1 label of 343" is a cross-extraction figure (vocabulary-only it is 3), and
"±22 sectoral terms" is 19 live TF-IDF entries. Add half a sentence conceding
that the exclusion bin is not homogeneous.

---

# Recommendation

**The material belongs in the paper, but not in this form and not at this length.**

The objection it answers is real and standing — a dictionary method should be
asked whether it merely reports its own dictionary, and no other section asks it.
The finding, correctly stated, is worth about **two paragraphs and one table** in
§10, plus three sentences in §11, plus the §5 and §12 edits, which are
independently correct and should go in regardless.

What must not ship is the framing. As drafted, the addition claims a hierarchy it
has not established (§3), on an experiment whose stated design is not the one that
ran (§1), against a baseline that no longer exists in the repository the paper
publishes (§2), with a statistic whose achievable range was fixed by the author's
own acceptance rule before any data were read (§4). Each is separately fixable.
Together, as written, they turn a modest and defensible result — *within the class
of dispersed terms this corpus supplies, the index is stable* — into a claim the
evidence does not carry: *the objection that a dictionary method merely reports
its own dictionary is answerable and answered.*

The honest version of the sentence is shorter and still worth printing:

> An expansion of the vocabulary to 289 entries, built from an audit of this
> corpus and admitted on dispersion measured over it, moves 18 of 343 labels.
> The check is one-sided — the terms were selected for the property that makes
> them least able to move a density ranking — and it is not independent of the
> corpus it is run on. It bounds the index's sensitivity to a defensible
> extension of this vocabulary. It does not bound the sensitivity to a
> vocabulary built on other principles, and nothing here does.

The drafts already contain most of that paragraph, in §10's "Two limits bound the
check narrowly". They then spend four hundred words unwinding it.
