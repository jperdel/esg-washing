# Agreed paper structure

Negotiated between the implementation side (10-section proposal) and the source-paper side (findings (a)–(e)).
Companion document: `paper/_research/lagasio_source.md` (full-text verification of Lagasio 2024).

**Author constraint, strict:** technical sections only. No introduction, no literature review, no ESG/greenwashing background, no policy or managerial implications. Sections needed for the paper to hold together but excluded by this constraint are listed as **PLACEHOLDER — not to be written**.

---

## Conventions to be applied throughout

**Provenance marking.** Two evidence regimes are in play and must not be blurred:
- Claims about **Lagasio (2024)** are VERIFIED against the full text (open access, CC BY 4.0 — quoting is unproblematic), with page numbers. Where the original is silent, the paper must say *"the original does not specify"*, never *"the original does X"*.
- Claims about **our pipeline** are ours to evidence.

Figures quoted from the implementation side in this document (343 reports, 74,004 zones, 35.5%, r = 0.9925, 28.2%/2.0%, −0.237, 47%/16%, 154 terms, 1,263 forms, 880 zones) have **not** been independently checked by the source-paper side. They are recorded here as their claims and need their own verification pass before drafting.

**Terminology discipline — mandatory.** Two distinct operations are both called "normalisation" and they sit at different pipeline stages. Conflating them would wreck §7 and §8 simultaneously. Fix the vocabulary now and use it without exception:
- **Vector normalisation** — L2, *within-document*, at the TF-IDF stage. The subject of §7.
- **Score normalisation** — min-max or z-score, *cross-sectional across the corpus*, at the index-assembly stage. The subject of §8.

The original discloses neither. That parallel is the structural spine of §7→§8 and should be stated explicitly in the bridge between them.

**Open reconciliation item.** The corpus is described as 343 unique reports, but the document-type split given (236 annual reports + 101 URDs + 7 sustainability reports) sums to 344. Must be resolved before §3 is drafted. In a paper whose central claim concerns reproducibility, an unreconciled count is not survivable.

---

## Section list

### 1. Scope and relationship to the original study — **TECHNICAL**

**Why it exists.** Neither list had a home for (d) or (e), and without it the reader has no way to learn that this is an adaptation rather than a strict replication — the introduction, which would normally carry that, is excluded. This is a technical delta statement (what is replicated, what is adapted, what cannot be recovered), not background prose. Kept deliberately short and tabular.

**Boundary flag for the author:** this is the section closest to the excluded-content line. It must stay a specification delta. If it acquires a single sentence about why greenwashing matters, it has become an introduction and should be cut back.

**Must carry:**
- **(d)** Corpus divergence, stated as a table, not buried: Lagasio uses 749 standalone sustainability reports, 2023 only, English only, drawn from a Refinitiv top-2000 starting sample, and **explicitly excluded firms reporting ESG inside the annual report** (VERIFIED, p. 3). Our corpus is 236 annual reports + 101 URDs + 7 sustainability reports, 2018–2024 — substantially the population the original excluded. Declared as a design consequence: a longitudinal design needs a consistent annually-recurring series, which standalone sustainability reports do not reliably provide. Conclusion to state plainly: **this is an adaptation, not a strict replication.**
- **(e)** The original flags its cross-sectional design as a limitation and calls for longitudinal work (VERIFIED, p. 10: *"The cross-sectional nature of our data limits causal inferences, highlighting the need for longitudinal studies to examine how changes in corporate characteristics and regulatory environments affect ESGSI over time"*), and separately invites *"refining sentiment analysis techniques"*. This is the entire justification for the contribution, obtained without writing an introduction — one paragraph, quoted, no elaboration.
- The original's own statement that a panel is unnecessary (VERIFIED, p. 3) — set against (e), since the two sit in tension in the original and our design resolves it in favour of (e).

**Dependencies:** none. Feeds §3, §11, §12.

---

### 2. What the original specifies, and what it does not — **TECHNICAL**

**Why it exists, and why it is early.** This is (c). It is placed second, before any of our machinery, for a load-bearing reason: it is what makes the rest of the paper necessary. In particular it supplies the missing justification for a design choice the implementation-side list never explains — **why §6 carries three SUS specifications and §10 tests robustness across them.** Three specs exist because the original does not determine one. Without §2, §6 and §10 look like arbitrary over-engineering. It also requires none of our pipeline, so it can stand this early.

**Must carry — all VERIFIED against the full text:**
- The complete TF-IDF disclosure, quoted: scikit-learn `TfidfVectorizer`, v0.24.2, with `max_df=0.95`, `min_df=2`, `max_features=10000`, `ngram_range=(1,2)` described in prose (p. 4). IDF weighting confirmed used.
- **`norm` is never mentioned.** No L2, no Euclidean, no vector normalisation of any kind, anywhere in the paper. Exhaustive term search documented in the companion note.
- **Aggregation from per-term TF-IDF to document-level `SUS_i` is never stated.** No sum, no mean, no formula. `SUS_i` is defined only as *"sustainability score for company i"* — a name, not a construction. The paper's fullest statement is *"It combines sentiment scores with TF-IDF scores of sustainability terms"* (p. 4).
- Keyword vocabulary: GRI (2021) and SASB (2018) provenance VERIFIED; **size never stated, list never published**; three example terms only ('Diversity', 'Carbon Emissions', 'Equal Employment Opportunities').
- Custom stopword list augmenting NLTK defaults — never published, and may overlap ESG vocabulary.
- Announced robustness checks **never reported**: Cohen's Kappa (two raters), 5-fold cross-validation, and sensitivity analysis explicitly covering *"TF-IDF settings"* — announced p. 5, no value or result anywhere in the paper. Note pointedly that the announced TF-IDF sensitivity analysis is precisely the one that would have settled §7 and §8.
- The flagged share of firms is **never reported**, despite "Potential ESGwashing" being the paper's central construct.
- Data availability statement, verbatim and in full: **"No"**.
- Which text reached TextBlob (raw vs. the lemmatised, stopword-stripped, punctuation-stripped pipeline output) is unrecoverable — carried forward as an instrument-validity problem in §7.

**Explicitly NOT here:** any claim that the original *did* sum, *did* L2-normalise, or *did* anything undisclosed. §2 is an audit of silence. Resolution of one of these silences is §8's job and rests on different evidence.

**Dependencies:** none. Feeds §6 (motivates three specs), §7, §8, §10, §13.

---

### 3. Data and corpus construction — **TECHNICAL**

Implementation-side §1, retained. 343 unique reports (pending reconciliation), 49 companies, 7 countries, 2018–2024 balanced panel; document-type assignment; the country/type confound stated up front as a design property rather than discovered later.

**Addition from this side:** the divergence table lives in §1, but §3 must carry the operational consequence — mixed document types within a single index, and what that does to comparability. Cross-reference, do not restate.

**Dependencies:** §1.

---

### 4. Document extraction: the lexical ESG-zone filter — **TECHNICAL**

Implementation-side §2, retained unchanged. Deterministic paragraph-level keyword-density extraction; parameters; yield (35.5% of PDF text, 74,004 zones); why it replaced the embedding-based filter.

**Note:** this step has no counterpart in the original, which vectorises whole reports. That divergence must be stated here and is material to §7 and §8, since it changes the denominator against which any length effect operates.

**Dependencies:** §3.

---

### 5. Preprocessing and lexicon architecture — **TECHNICAL**

Implementation-side §3, retained. The "same representation by construction" build step; the 154-term vocabulary; the 1,263-form L&M hedge lexicon; the external 880-zone audit.

**Addition from this side:** state explicitly that our 154-term vocabulary **cannot** be matched to the original's, because the original never publishes its list or its size (§2). This is the point at which strict replication becomes impossible, and it should be named as such here rather than left to the limitations section. Consequence to state: divergence between our ESGSI and the original's cannot be attributed to any single design choice.

**Dependencies:** §2, §4.

---

### 6. Index specification — **TECHNICAL**

Implementation-side §4, retained. Exact formulas for SUS (three specifications), SEN, QUANT, HEDGE, Breadth, and the two composite indices.

**Addition from this side, required:** reproduce the original's index definition verbatim and flag the contradiction *at the point of definition*, without yet resolving it — resolution is §8. Specifically, the original prints (p. 4):

> ESGSI = (SEN_i − mean SEN)/σ_SEN − (SUS_i − mean SUS)/σ_SUS

while the prose in the same subsection states the scores are *"normalized to a uniform scale using min-max normalization"*. Both quoted; the contradiction named; the reader told it is adjudicated in §8. Also record the labelling rule: ESGSI > 0 → "Potential ESGwashing", ≤ 0 → "Likely Genuine" (VERIFIED, pp. 4–5), definitional, with no calibration.

**Dependencies:** §2 (why three specs), §5.

---

### 7. Measurement critique: TF-IDF is a retrieval representation, not a measurement instrument — **TECHNICAL**

Implementation-side §5, retained as the general/analytic critique. The L2 invariance identity plus two empirical demonstrations; the IDF inversion; weighting is second-order (r = 0.9925), **vector** normalisation is first-order.

**Scope discipline:** §7 is a claim about a *class* of methods, holds analytically, and would hold even if the original had documented everything. It is not a claim about Lagasio's conclusion. Keep that line clean — it is what stops §8 from looking like a pile-on.

**Addition from this side:** a short subsection on the SEN component, which neither list critiques. The original applies TextBlob — a general-purpose, pattern-based polarity tool — to text that its own preprocessing has stripped of stopwords, punctuation and numerals and then lemmatised (VERIFIED, p. 3). TextBlob polarity depends on negation and intensifier handling, which that pipeline destroys. Whether raw or processed text reached TextBlob is unrecoverable (§2). This is an instrument-validity problem on the other half of the index and belongs in the measurement critique.

**Dependencies:** §5, §6.

---

### 8. The normalisation artefact: how an undisclosed choice produced the original's headline result — **TECHNICAL. Carries the central claim.**

**This is (a) + (b), and the decision to make it a separate section rather than fold it into §7 is argued below, not asserted.**

**Why separate from §7 — four reasons, the last decisive:**

1. **Different object.** §7 critiques a method class. §8 asserts that one specific published conclusion is an artefact. Different scope, different burden of proof.
2. **Different epistemic status.** §7 rests on a mathematical identity — it is proof. §8 rests on arithmetic reconstruction plus a factorial experiment — it is inference to the best explanation and must be hedged with care that §7 does not need. Housing proof and inference under one heading invites a reader to discount both to the level of the weaker.
3. **Different stage of the pipeline.** §7 concerns *vector* normalisation (L2, within-document, TF-IDF stage). §8 concerns *score* normalisation (min-max vs z-score, cross-sectional, index-assembly stage). These are genuinely different operations that share a word. Folding them into one section would force the paper to use "normalisation" for both within a few paragraphs — the single most likely way to lose the reader on the two points that matter most.
4. **Decisive:** §8 is the paper's strongest result (see central-claim argument below). A result that carries the paper should not be a subsection of a methods critique. Burying it costs it its standing.

**They must nonetheless be adjacent and explicitly bridged.** The required bridge: *at both stages there is a normalisation choice; at both, the normalisation dominates the weighting; the original discloses neither.* §7 establishes that normalisation choices are first-order, which is exactly what makes §8's result predictable rather than coincidental. That is why §7 precedes §8 — without §7, §8 reads as a lucky finding; with it, as an inevitable consequence.

**Must carry, part (a) — establishing the choice, from the original's own tables:**
- The printed formula is a z-score difference; the prose says min-max; the published numbers are unambiguously min-max.
- The arithmetic: Table 2 gives mean ESGSI −0.2179 = 0.3727 − 0.5906 exactly, with both components documented as bounded on [0,1]. `NormSEN − NormSUS` reproduces the reported ESGSI to 4 dp in **14 of 14 rows tested across Tables 2, 3, 4, 5 and 6**. Under a z-score difference the mean is 0 by construction; the reported mean is −0.2179 with sd 0.1614. Present the full row-by-row table — it is short and it is dispositive.
- The consequence: the original's headline conclusion — *"The ESGSI predominantly shows negative values, suggesting a general absence of greenwashing practices among the entities studied"* (VERIFIED, p. 6) — **flips under the formula the paper itself prints**, where roughly half the sample would be flagged.
- The zero threshold inherits the asymmetry between the two components' distributions (Table 2: NormSUS mean 0.5906 vs NormSEN mean 0.3727; the original itself notes the sustainability distribution is *"skewed towards higher values"*, p. 6).
- The category error: the original's only stated defence against document-length bias is that min-max *"address[es] potential biases due to variations in document length"* (p. 4). Min-max is a monotone affine rescale across the cross-section; it is rank-preserving and cannot remove a within-document length effect. The step that could — L2 — is the undisclosed one (§2, §7). This is the cleanest single link between §7 and §8.

**Must carry, part (b) — establishing the consequence, on our corpus:**
- Min-max plus the Lagasio-style SUS reproduces the sign and near-magnitude of the original's headline statistic (−0.237 vs −0.2179, gap 0.019) and drops the flagged share from ~47% to 16%.
- Full 2×2 factorial: changing either choice alone eliminates the effect. Report all four cells, not just the two endpoints.

**Overclaiming guards — mandatory, and the section must be drafted with these in view:**
- **Division of labour must be explicit.** (a) establishes *which* normalisation the original used — and it does so from the original's own published tables, which is strong. (b) establishes *the consequence* of that choice. (b) is **not** additional evidence for (a): a matching mean on a different corpus with a different vocabulary is consistency, not proof, and coincidence is possible. Say so. The claim is "these choices jointly reproduce the statistic and are jointly necessary for it *on our corpus*", never "we have proven Lagasio used min-max" — (a) already carries that, and more cleanly.
- The 47%/16% figures are **our-corpus quantities**. They must not be presented as estimates of the original's flagged share, which is unknown and unknowable: the original never reports it (§2).
- The near-match at 0.019 is suggestive, not probative, and should be presented with that framing rather than as a bullseye.

**Dependencies:** §2 (the silences), §6 (SUS specs and the quoted contradiction), §7 (normalisation is first-order).

---

### 9. Convergent validity and specification choice — **TECHNICAL**

Implementation-side §6, retained. QUANT as an independent yardstick; the structural argument for rejecting sublinear TF.

**Dependencies:** §6, §7.

---

### 10. Robustness of the index to SUS specification — **TECHNICAL**

Implementation-side §7, retained. 28.2% vs 2.0% label flips.

**Cross-reference required:** this is the empirical counterpart to §2's finding that the aggregation is unspecified. A 28.2% label-flip rate across specifications that the original's text does not distinguish between is what makes the §2 silence consequential rather than pedantic. State the link.

**Dependencies:** §2, §6, §8.

---

### 11. Temporal results — **TECHNICAL**

Implementation-side §8, retained. Yearly means, driver decomposition, composition-artefact rebuttals.

**Framing requirement, from this side:** the temporal series must be reported *conditional on* the measurement and normalisation findings. Having argued in §7–§8 that the instrument is unstable under undisclosed choices, the paper cannot then report a time series of that instrument at face value. State which specification the series uses, why (§9), and how it moves under the alternatives (§10). Reference (e) — the original explicitly invited this extension — as the motivation.

**Dependencies:** §1(e), §8, §9, §10.

---

### 12. Limitations — **TECHNICAL**

Ours, plus a short comparison against the original's. The original's entire limitations discussion is two paragraphs in its conclusion (VERIFIED, p. 10) and acknowledges only: cross-sectional design, and that ESGSI measures could be more sophisticated. It does **not** acknowledge the normalisation contradiction, the undisclosed aggregation, the unpublished vocabulary, English-only selection, the loss of 1,251 of 2,000 firms, the arbitrary zero threshold, or the absence of any external validation.

Ours must include, at minimum: vocabulary non-comparability (§5); the country/document-type confound (§3); mixed document types; 49 companies is a small panel; no external greenwashing benchmark for validation on our side either — the same gap we identify in the original, and it must be conceded in the same terms.

**Dependencies:** all.

---

### 13. Reproducibility appendix — **TECHNICAL**

Implementation-side §10, retained, and strengthened: full parameter listings, the vocabulary and hedge lexicon in full, seeds, versions, and the audit protocol — i.e. everything §2 shows the original withheld. The contrast is the point and should be made structurally, by completeness, rather than argued in prose.

**Dependencies:** §2 and all implementation sections.

---

### 14. Appendix B: internal inconsistencies in the original — **TECHNICAL, gated**

Kept out of the main critique deliberately. These are verified but peripheral; leading with them would read as pile-on and would dilute §7–§8, which stand on their own. Recorded because they bear on how much weight the original's reported results can carry, and because anyone building on the original needs them.

- Energy sector: §4.2.5 prose states Energy has *higher* ESGSI than Health Care and Industrials; Table 4 shows Energy lowest of the three (−0.3117 vs −0.1943, −0.1893) — correct magnitudes, opposite sign. The conclusion builds on the wrong sign.
- §4.2.2 states Energy and Materials have *lowest* sustainability scores; Table 4 gives them the highest and second-highest.
- §4.2.2 reports a "Consumer Goods" sector with mean ESGSI −0.3442 and NormSUS 0.5730; **no such row exists in Table 4** and neither figure appears in it. Sector labels in that section do not match the table's GICS labels.
- §4.2.4's narrative reverses the sign convention relative to §4.2.5, and Table 6 supports neither.
- Table 7: Assets Category reported at F = 6.165754 with p = 3.86E-01, while the text claims a significant post-hoc difference. An F of 6.17 across four quartiles at n = 749 cannot give p = 0.386. F-statistics throughout are printed with mangled decimal separators, the same fault that renders 1.0000 as "10.000" in Table 2.

**Hard gate — do not breach.** A set of reference entries in the original could not be matched to real publications and have generic author names and titles. This is flagged **UNVERIFIED** in the companion note and was deliberately not investigated. It is a serious allegation, it is not necessary to any argument the paper makes, and it must **not** appear in any draft unless independently confirmed title-by-title against publisher databases and the author explicitly decides to raise it. Default: omit.

**Dependencies:** §2.

---

## Placeholders — not to be written

| # | Section | Note for the human author |
|---|---|---|
| P1 | Abstract | Must carry the central claim (below). This — not section order — is where the headline gets front-loaded. |
| P2 | Introduction | Must establish the ESG-washing measurement problem and state the contribution. §1 deliberately does not do this. |
| P3 | Literature review | Greenwashing measurement, NLP in accounting/finance, prior ESG-washing indices. |
| P4 | Discussion / implications | Policy, regulatory and practitioner implications. |
| P5 | Conclusion | |

**Structural warning:** as ordered, the paper opens on §1 (scope delta) and §2 (an audit of another paper's silences). Between P2 and §3 the reader meets a critique of the original before learning what the original is for. That is a consequence of the constraint, not a flaw in the structure, but the author should know P2 carries more than usual weight — it is the only place the object of study gets introduced.

---

## Order

1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11 → 12 → 13 → 14

Two ordering choices worth recording, both contested and both resolved deliberately:

**Critique after implementation, not before.** The empirical critique cannot be stated before the machinery exists: the L2 invariance demonstration runs on our corpus and the §8 factorial uses our SUS specifications and lexicon. The implementation side's instinct here was right. The exception is §2, which is a documentary audit needing none of our machinery and which motivates design choices in §6 and §10 — hence early.

**§7 before §8.** §7 establishes that normalisation choices are first-order; §8 then shows a first-order choice left undisclosed, driving a published conclusion. General → specific. Reversing them would put the headline first but would leave §8 looking like a coincidence rather than a consequence. If the author wants the headline earlier, that is solved in the abstract (P1), not by reordering.

---

## The central claim

Three candidates were in contention. Argued, not asserted:

**Rejected — the temporal finding.** It is the project's stated purpose and it is the weakest candidate, for a reason that is itself one of our results. If §7 shows the instrument is a retrieval representation rather than a measurement instrument, and §8 shows its published calibration is an artefact, then a time series of that index inherits every one of those defects. The paper cannot lead with "ESG-washing moved thus over 2018–2024" while simultaneously arguing the instrument is unsound. Leading with it would be self-undermining. §11 is the demonstration of consequences, not the thesis.

**Rejected — the reproducibility failure.** Real and damning, but it is a claim about documentation, not substance, and readers discount procedural findings. On its own it establishes only that the original cannot be checked — necessary scaffolding, not a thesis.

**Rejected as sole claim — the measurement critique.** The strongest *general* contribution: analytic, proof-backed, generalises well beyond Lagasio. But alone it is a methods note. It establishes that an approach is ill-founded, not that any published finding is wrong.

**Adopted — the artefact claim (§8), with §7 as mechanism and §2 as enabling condition:**

> A published empirical conclusion — that ESG-washing is largely absent from corporate sustainability disclosure — is an artefact of two undisclosed pipeline choices rather than a property of the disclosure. We establish the choice from the original's own published tables, explain why such choices are first-order through the measurement properties of TF-IDF, and demonstrate the consequence by reproducing the original's summary statistic on an independent corpus and showing it disappears when either choice is varied.

This is the only candidate with all three properties: **substantive** (a published conclusion is wrong, not merely undocumented), **quantitative** (14/14 exact row reconstruction; −0.237 vs −0.2179; 47% → 16%; a full factorial), and **general** (it converts the measurement critique from pedantry into demonstrated consequence).

The other two candidates are not discarded — they are subordinated into supporting roles, which is what makes the claim defensible: §7 supplies the mechanism that makes the artefact predictable, §2 supplies the condition under which it went undetected, §11 supplies what the corrected instrument shows. The claim is a conjunction, and it should be stated as one.
