# Cross-section issues for the unifier

Collected from the drafting and review agents' report-backs. These are the
things no single section could fix from inside itself. Each needs a decision
or an edit at assembly time.

## Factual errors to fix

1. ~~**§3 misdescribes the deduplication basis.**~~ **RESOLVED.** The code
   (`main.py:201`) hashes `clean_text`, the preprocessed representation, not
   the raw extraction. §3 now says so. Verified separately that for this
   particular pair the extracted text is identical too — same length, same
   MD5, same 224 zones — so the section's parenthetical claim to that effect
   is accurate and can stand.

2. **Possible write race on §13.** The reproducibility appendix was rewritten
   at 18:54:09, after its adversarial reviewer had already started. Verify the
   file contains the reviewer's corrections and not a clobbered earlier state.
   If in doubt, re-review §13 alone.

## The one cross-section argument that must be fixed together

1b. **"You want the default convention both ways."** §2's reviewer identified
    this as the original author's best available counter, and it cannot be
    answered inside either section alone.

    THE CHARGE: §2 treats the unstated `norm` as a reproducibility defect,
    while §7 builds its whole L2-invariance demonstration on the inference
    that `norm='l2'` was in force — an inference licensed by the convention
    that only non-default hyperparameters get reported. If that convention is
    good enough to run our experiment on, it was good enough disclosure.

    THE RESOLUTION, to be applied identically in both sections: **concede the
    `norm` inference and move the reproducibility charge off it.** All four
    disclosed settings are indeed non-defaults, so `norm='l2'` is reasonably
    inferable and we should say so plainly rather than counting it as a
    silence. The charge does not need it, because the genuinely unanswerable
    gap is elsewhere and is not a hyperparameter at all:

    - **The aggregation from per-term TF-IDF values to a single $\SUS_i$ is
      never stated.** Sum, mean, weighted — nothing. This is a modelling step,
      not a library setting, so no default convention can supply it. It is the
      strongest instance of the reproducibility charge and should carry it.
    - The keyword list is unpublished and its size unstated.
    - §3.3's prose and printed formula contradict each other on score
      normalisation, and no convention resolves which was run.

    Result: §7 proceeds on an inference it openly concedes is reasonable, and
    §2's charge rests on gaps that no reader could close from any convention.
    The two are then consistent. Both sections must use the same wording for
    the concession — a referee comparing them will look for daylight.

## Factual errors found in the review pass — VERIFIED, must be fixed

1c. **§5 claims a guarantee the code does not provide.** It states that terms
    collapsing to a single generic word under preprocessing are withheld from
    the TF-IDF vocabulary. They are not. `scripts/build_lexicons.py:79-80`
    appends the term to a `collapsed` list for a warning and then runs
    `lemmatized.setdefault(lemma, term)` **unconditionally**. Verified: the
    generated vocabulary has 154 entries and `diligence` (from *due
    diligence*) is one of them. `agreement` and `transition` are absent only
    because a human deleted *paris agreement* and *just transition* from
    `esg_terms.txt` after reading the warning.

    So the guard is advisory plus a manual decision, not automatic. Describe
    it that way. In a paper whose central claim is about undisclosed
    methodological choices, advertising an automatic safeguard that is
    actually a human judgement call is the worst possible place to be
    imprecise. (Fixing the code to match the prose is the wrong repair: the
    collapse of *due diligence* to *diligence* is benign — 4,330 occurrences,
    almost all from the source phrase — and an automatic rule would silently
    discard it.)

1d. **§13 misreads its own results file.** It claims one document's $\ESGSI$
    rounds to 0.0000 and that "a naive count of strictly positive rounded
    values (171) differs by one from the published label count (172)".
    Verified against `results.csv`: `Etiqueta` is *Potential ESG-washing* 171
    and *Likely Genuine* 172 — 172 is the count of the OTHER label — and no
    document has $|\ESGSI| < 0.0005$. There is no discrepancy and no rounding
    quirk. Delete the passage.

1e. **The deduplication is described imprecisely in two places.** Both
    `conventions.md` §5 and §3 imply the hash is over extracted text. The code
    hashes `clean_text`, the preprocessed representation. §3 has been fixed;
    check nothing else inherited the loose wording.

## Redundancy to resolve

3. **Two parameter tables.** §6 carries `tab:index-parameters` and §13 carries
   a consolidated parameter table. §13's author added a sentence framing its
   table as a consolidation rather than an independent source. Decide whether
   both survive; if they do, they must not disagree on a single value.

4. **§4/§5 boundary.** §4 owns the raw-text regex and extraction; §5 owns the
   lemmatised representation and the TF-IDF vocabulary. Both describe the
   154-term list from different sides. Check no value is stated twice with
   different rounding, and that neither claims ownership of the other's side.

## Claims that must stay consistent across sections

5. **The 878-of-880 external audit reproduction is NOT usable.** The
   supporting workbooks live in `feedback/`, which is git-ignored, so the claim
   cannot be verified by a reader or a referee. §4 and §13 both correctly
   omitted it. Ensure it has not crept into §5, which discusses the same audit
   for the 147-term gap — that gap IS verifiable from the spreadsheet and is
   fine to keep, but the zone-count reproduction is not.

6. **`implementation.md` is stale throughout.** Every table in it is on the
   pre-deduplication n = 344 basis. `conventions.md` is binding. Any figure in
   any section traceable only to `implementation.md` needs checking against
   `results/metrics/results.csv` before it survives assembly.

7. **The contamination concession.** §12 argued, correctly, that the
   SUS/QUANT overlap is not merely a limitation but undermines the yardstick
   on which §9 selects the specification. §9 has been instructed to concede it
   in full. Verify that §9 and §12 now say the same thing at the same
   strength, and that §9 does not defer it to §12.

## Editorial

7b. **Cross-reference style is inconsistent paper-wide.** Some sections write
    `\S\ref{sec:X}`, others bare `\ref{sec:X}`. Currently `\S\ref` in §1, §9,
    §10, §11 (and now §3's new sentence); bare `\ref` in §4, §6, §8, §12, §13.
    No single-section reviewer could fix this without creating a new
    inconsistency, so it is explicitly the unifier's. Pick one and apply it
    everywhere.

8. **Terminology.** "Vector normalisation" (L2, within-document, TF-IDF stage)
   vs "score normalisation" (min–max/z-score, cross-sectional, index
   assembly). No bare "normalisation" anywhere. This was mandated; verify it
   held across all thirteen sections, especially at the §7/§8 boundary where
   the two meanings sit adjacent.

9. **Placeholders.** Abstract, introduction, literature review, discussion and
   conclusion are deliberately unwritten and commented out in `main.tex`. Do
   not write them. Do not let a technical section drift into their territory
   to compensate.

10. **Hard gate.** The possible reference-integrity problems in the original
    stay out of every draft, in every form, including allusion.

## Evidence found after drafting — use it

13. **The feasibility of a standalone-report panel, partially answered.** §1's
    reviewer named this the strongest surviving objection: the corpus
    divergence is convenient rather than forced unless we can show a balanced
    panel of standalone sustainability reports was infeasible. Measured from
    `results.csv`: of the 49 firms, **exactly one — Sanofi-Aventis — is
    represented by standalone sustainability reports, and it is so in all
    seven years**. A panel restricted to that document type would have had one
    firm. Separately, **43 of the 49 firms keep a single document type across
    the whole window**; the six that switch are one-off annual-report-to-URD
    transitions consistent with the French URD regime.

    WORDING CONSTRAINT, and it is not negotiable: our data record the document
    *selected* per firm-year, not what was *available*. Write "exactly one
    firm is represented by standalone sustainability reports across the
    window", never "only one firm published one". The distinction is exactly
    the kind this paper accuses the original of blurring.

## Open decisions for the human author

10b. **The sampling frame has no stated provenance.** Nothing in the repo or
    the research notes explains why these 49 firms and these seven countries.
    §1's reviewer searched and could only label it an unexplained decision.
    A human must supply the rationale; it cannot be inferred and must not be
    invented.

11. **Topic modelling.** `results/lda/coherence.csv` is stale and the LDA stage
    is disabled in `main.py`. Either it is rerun and gets a section, or it is
    omitted entirely. Currently no section claims it. Leave it out unless the
    author says otherwise.

12. **The `esr` artefact.** The generated vocabulary contains `esr`, a
    lemmatisation artefact of "ESRS", with 6,282 occurrences and idf 2.53 — so
    it is heavily up-weighted under both idf specifications. It does not
    affect `density`, the preferred specification, but it does affect the two
    robustness variants. §12 or §13 should carry it; check one of them does.
