# Lagasio (2024) — source paper for the ESGSI replication

Research note. Compiled 2026-08-02 from the **full text** of the published article, held locally at
`C:\Users\Jorge\Desktop\Proyectos\ESGwashing\paper\1-s2.0-S1057521924006744-main.pdf`
(10 pages, open access CC BY, text extracted with PyMuPDF).

Page numbers below refer to the article's own printed page numbers, which coincide with the PDF pages (1–10).

## Evidence-level key

- **VERIFIED** — read in the full text; quote and page given.
- **VERIFIED (arithmetic)** — not stated in prose, but established by exact arithmetic on the paper's own published tables. Reasoning shown so it can be re-checked.
- **NOT STATED** — I read the whole paper (all 10 pages, including footnotes, table notes, the data-availability statement and references) and the paper does not say. Searched exhaustively for the relevant terms.
- **METADATA-ONLY / UNVERIFIED** — seen only in search-engine metadata, not confirmed against a source I could open.

---

## 0. What we could and could not verify

**Could verify.** The full text was available locally, so nearly everything is first-hand. The citation, the corpus, the index construction, the TF-IDF library and its four stated hyperparameters, the sentiment method, the keyword-list provenance, the greenwashing threshold, the announced robustness checks, and the acknowledged limitations are all VERIFIED with quotes.

**The single most important result — and it is not the one we expected.** The paper is **internally contradictory about its own index**. Section 3.3 prints a formula that is a **difference of z-scores** (matching our reconstruction), but the prose in the *same subsection* says the scores are **min-max normalised**, and the paper's *published numbers* are unambiguously min-max, not z-scores. `ESGSI = NormSEN − NormSUS` reproduces the reported ESGSI to four decimal places in **all 14 table rows I tested**. Under a z-score difference the mean ESGSI would be 0 by construction; the paper reports a mean of −0.2179. So the estimated index in the paper is **not** `Z(SEN) − Z(SUS)`. See §2 and §3.

**Could NOT verify — and this is itself the key finding.** The paper **never states how the per-term TF-IDF values are aggregated into a document-level sustainability score.** Not sum, not mean, not any formula. It also **never states whether the document vectors are L2-normalised**, never states the `norm` parameter, and never gives the size of the ESG keyword list. These are not access failures — I have the full text and the paper simply does not say. That silence is the finding.

**Our prior reconstruction was wrong in three respects:**

| Our belief | Reality |
|---|---|
| `ESGSI = Z(SEN) − Z(SUS)` | The *printed formula* is z-scores, but the *estimated index* is min-max (VERIFIED arithmetic). The paper contradicts itself. |
| "the original SUMS the TF-IDF values" | **NOT STATED anywhere in the paper.** We cannot attribute a sum to Lagasio. |
| Corpus = corporate **annual reports** | **Standalone ESG/sustainability reports only.** Companies that put ESG inside the annual report were explicitly *excluded*. |

Our belief that SEN was TextBlob polarity is **VERIFIED and correct**.

---

## 1. Full citation — VERIFIED

> Lagasio, V. (2024). ESG-washing detection in corporate sustainability reports. *International Review of Financial Analysis*, **96**(Part B), 103742.
> DOI: `10.1016/j.irfa.2024.103742`

- **Sole author** — Valentina Lagasio, Department of Management, Faculty of Economics, Sapienza University of Rome, Italy. Corresponding address Via del Castro Laurenziano 9, 00161 Rome; `valentina.lagasio@uniroma1.it` (p. 1). Note the paper uses "we" throughout despite being single-authored.
- Received 1 May 2024; revised 29 October 2024; accepted 30 October 2024; online 8 November 2024 (p. 1).
- **Open access, CC BY 4.0**: "1057-5219/© 2024 The Author. Published by Elsevier Inc. This is an open access article under the CC BY license" (p. 1). Reproducing quotes and the formula is therefore unproblematic.
- Keywords: Greenwashing; ESG-washing; Natural language processing; Transparency; ESG disclosure; ESG; Sustainability (p. 1).

**Related item — METADATA-ONLY, UNVERIFIED.** A search returned an SSRN entry, *"Measuring Greenwashing: The Greenwashing Severity Index"*, Valentina Lagasio, SSRN abstract id **4582917**, apparently an earlier working-paper version (a search-engine summary mentioned 702 companies rather than 749, and a "Greenwashing Severity Index / GSI" rather than ESGSI). **I could not open SSRN (HTTP 403) and have verified nothing about its contents.** If the replication needs to know whether the working paper is more explicit about the TF-IDF aggregation than the published version, that document is the obvious next place to look — it is the single highest-value outstanding lead.

There is no ambiguity about *which* paper is ours: the local PDF is the published ESGSI article and matches the project in name, formula and construction.

---

## 2. Exact mathematical construction of the index — VERIFIED, but internally contradictory

### 2a. The printed formula (p. 4, §3.3)

Verbatim, exactly as it appears (the PDF sets it as two stacked fractions):

> ESGSI = SENi − SEN
> σSEN
> − SUSi − SUS
> σSUS

i.e.

```
ESGSI_i = (SEN_i − mean(SEN)) / σ_SEN  −  (SUS_i − mean(SUS)) / σ_SUS
```

with the definitions given immediately below it verbatim (p. 4):

> "SENi: sentiment score for company i.
> SEN: mean sentiment score across all companies
> σSEN: standard deviation of sentiment scores
> SUSi: sustainability score for company i
> SUS mean sustainability score across all companies
> σSUS: standard deviation of sustainability scores"

This is a **difference of z-scores** and matches our reconstruction.

### 2b. The prose in the same subsection says something different — VERIFIED (p. 4, §3.3)

> "The ESGSI is calculated by subtracting the normalized sustainability score from the normalized sentiment polarity score. This approach builds on previous work on measuring corporate greenwashing (Delmas & Burbano, 2011; Lyon & Montgomery, 2015). **To ensure comparability, both sentiment and sustainability scores are normalized to a uniform scale using min-max normalization.** This normalization technique is widely used in text analysis and machine learning to address potential biases due to variations in document length and linguistic style (Sebastiani, 2002)."

The same claim appears again at the end of §3.2.3 (p. 4):

> "Both sentiment and sustainability scores are normalized to a uniform scale to ensure comparability and address potential biases due to variations in document length and linguistic style. The ESGSI is then derived by subtracting the normalized sustainability scores from the normalized sentiment scores."

So within a single subsection the paper asserts min-max normalisation and then prints a z-score formula. **The paper never reconciles the two.**

### 2c. Which one was actually estimated — VERIFIED (arithmetic): min-max, not z-scores

Three independent lines of evidence, all from the paper's own published numbers:

**(i) The reported variables are bounded on [0,1].** Table 2 notes (p. 6) state verbatim: *"Normalized Sustainability Score: Scores range from 0 (low sustainability) to 1 (high sustainability)"* and *"Normalized Sentiment Score: ... normalized between 0 and 1."* Table 2 gives min = 0.0000 and max = 1.0000 for both (printed as "10.000", a decimal-separator mangling that also affects Table 7's F-statistics — see §9). Z-scores are not bounded on [0,1].

**(ii) The ESGSI mean is not zero.** Table 2 reports mean ESGSI = −0.2179, sd = 0.1614. A difference of two z-scores has mean exactly 0 by construction. It does not.

**(iii) `ESGSI = NormSEN − NormSUS` reproduces every published row to 4 dp.** Checked across Tables 2, 3, 4, 5 and 6:

| Row (source) | NormSEN | NormSUS | SEN−SUS | Reported ESGSI |
|---|---|---|---|---|
| Overall (T2, p. 6) | 0.3727 | 0.5906 | −0.2179 | −0.2179 |
| Europe (T3, p. 7) | 0.3477 | 0.6144 | −0.2667 | −0.2667 |
| North America (T3) | 0.3917 | 0.5731 | −0.1814 | −0.1814 |
| South America (T3) | 0.3229 | 0.6622 | −0.3393 | −0.3393 |
| Other (T3) | 0.3665 | 0.5760 | −0.2095 | −0.2095 |
| Total (T3) | 0.3572 | 0.6064 | −0.2492 | −0.2492 |
| Energy (T4, p. 7) | 0.3478 | 0.6595 | −0.3117 | −0.3117 |
| Information Technology (T4) | 0.3788 | 0.5547 | −0.1759 | −0.1759 |
| Total (T4) | 0.3716 | 0.5934 | −0.2218 | −0.2217 |
| Assets Q1 (T5, p. 8) | 0.3853 | 0.5688 | −0.1835 | −0.1835 |
| Assets Q4 (T5) | 0.3595 | 0.6093 | −0.2498 | −0.2498 |
| ESG Q1 (T6, p. 9) | 0.3857 | 0.5650 | −0.1793 | −0.1793 |
| ESG Q4 (T6) | 0.3529 | 0.6223 | −0.2694 | −0.2695 |
| Total (T6) | 0.3727 | 0.5907 | −0.2180 | −0.2180 |

14/14 exact to rounding. **Conclusion: the estimated ESGSI is the difference of two min-max-normalised scores on [0,1], not the z-score difference the paper prints.** This is VERIFIED by arithmetic, not by any statement in the paper — the paper never acknowledges the discrepancy.

**Implication for the replication.** Our code implements `Z(SEN) − Z(SUS)`, i.e. the paper's *printed formula*. That is a defensible reading of the paper, but it is **not** what produced the paper's results, and the two are not monotone transforms of each other: min-max and z-scoring differ in how the two components' spreads are equalised, so the ranking of firms by ESGSI genuinely changes. The choice also determines the meaning of the `ESGSI > 0` threshold (§6).

---

## 3. TF-IDF details — the critical section

This is the priority question. Below is **every** sentence in the paper bearing on TF-IDF. I searched the full extracted text case-insensitively for `sum`, `aggregat`, `mean of`, `average`, `L2`, `euclidean`, `norm`, `length`, `magnitude`, `scale`, `vector`, `weight`. The passages below are the complete set of hits relevant to construction of the sustainability score.

### 3a. Library and hyperparameters — VERIFIED (p. 4, §3.2.1)

> "We employ Term Frequency-Inverse Document Frequency (TF-IDF) vectorization to convert text data into numerical values, following the approach of Engle et al. (2020). **We use Scikit-learn's TfidfVectorizer with settings chosen to balance comprehensiveness and computational efficiency, in line with best practices in NLP research. These settings include ignoring terms that appear in more than 95 % of documents, only considering terms that appear in at least 2 documents, limiting the vocabulary to the top 10,000 terms, and considering both unigrams and bigrams.** This method allows for the evaluation of the relative importance of each term within the documents against the entire dataset. The TF-IDF vectorizer, provided by the scikit-learn library, is used with adjustable settings to accommodate the unique features of different sustainability reports."

Also, from §3.1.1 (p. 3):

> "Scikitlearn version 0.24.2 is employed for text vectorization and machine learning tasks (Pedregosa et al., 2011)."

- **Implementation: VERIFIED.** `sklearn.feature_extraction.text.TfidfVectorizer`, scikit-learn **0.24.2**.
- **Stated hyperparameters: VERIFIED.** In sklearn terms these are `max_df=0.95`, `min_df=2`, `max_features=10000`, `ngram_range=(1,2)`. The paper describes them in prose only; it never prints the parameter names or a code listing.
- **IDF weighting used at all? VERIFIED — yes.** The paper names TF-IDF and `TfidfVectorizer` explicitly, and describes "the relative importance of each term within the documents **against the entire dataset**". This is IDF weighting, not raw term frequency. (`sublinear_tf` and `smooth_idf` are **NOT STATED**.)

### 3b. L2 normalisation of document vectors — **NOT STATED**

**The paper never mentions the `norm` parameter, L2, Euclidean length, or vector normalisation of any kind.** The four settings quoted above are the *only* vectoriser settings the paper discloses, and `norm` is not among them.

What can be said without inference: the enumerated settings are exactly four of the non-default `TfidfVectorizer` arguments. What **cannot** be said: whether the author left `norm` at its library default. It is a documented fact about scikit-learn 0.24.2 that `TfidfVectorizer`'s default is `norm='l2'` (with `use_idf=True`, `smooth_idf=True`, `sublinear_tf=False`) — but **whether Lagasio ran it at the default is an inference, not a finding**, and the paper gives no basis to settle it. It is recorded here as an open question (§10), not as a result.

**Do not write in the critique that Lagasio L2-normalised, and do not write that she did not.** The paper is silent. The defensible claim is that *the paper does not disclose it*, which is itself a reproducibility defect worth stating.

### 3c. Aggregation over the keyword vocabulary — sum or mean? **NOT STATED**

**This is the central gap.** The paper never gives a formula for `SUS_i`, and never uses the words *sum*, *total*, *mean*, or *average* in connection with the sustainability score. The only two statements that touch on it:

From §3.2.1 (p. 4):

> "Documents are preprocessed and stored in a directory, from which they are retrieved and transformed into a list of texts for analysis. **Each document is then vectorized with the TF-IDF model to calculate a score for each term** based on its frequency relative to other documents. This approach highlights terms that are particularly significant within specific reports, thereby pinpointing the sustainability indicators most prominently featured in the corporate ESG disclosures."

From §3.2.3 (p. 4):

> "By integrating sentiment analysis with sustainability content quantification, the methodology assesses potential ESGwashing in corporate communications. **It combines sentiment scores with TF-IDF scores of sustainability terms to objectively evaluate how extensively sustainability topics are discussed in the reports.**"

And the abstract (p. 1):

> "integrating sentiment analysis with **the frequency of sustainability terms** to calculate the ESGSI"

That is the whole of it. The step from a *per-term* TF-IDF score to a *per-document* sustainability score is never specified. `SUS_i` is used in the §3.3 formula and defined only as "sustainability score for company i" — a name, not a construction.

**Consequence for our critique.** Our code comment asserting that "the original SUMS the TF-IDF values" is **not supported by the paper** and must not be presented as Lagasio's method. Any critique of sum-vs-mean aggregation applies, on the present evidence, **to our replication's choice, not to a documented choice of the published paper**. What *can* be said of the paper is stronger and cleaner: *the paper does not specify the aggregation at all, so the index as published is not reproducible from its own description.*

### 3d. Document-length normalisation — VERIFIED claim, but the claim is about min-max, not about TF-IDF

The paper does address document length, twice, and in both places it attributes the correction to the **min-max normalisation of the final scores**, not to anything in the TF-IDF step (p. 4, §3.3):

> "To ensure comparability, both sentiment and sustainability scores are normalized to a uniform scale using min-max normalization. **This normalization technique is widely used in text analysis and machine learning to address potential biases due to variations in document length and linguistic style** (Sebastiani, 2002)."

and (p. 4, §3.2.3):

> "Both sentiment and sustainability scores are normalized to a uniform scale to ensure comparability and **address potential biases due to variations in document length** and linguistic style."

**This is the paper's own claim, quoted exactly.** Our observation — flagged here as *our* analysis, not the paper's — is that min-max normalisation is a single monotone affine rescaling applied across the cross-section (`(x − min)/(max − min)`). It is rank-preserving within each variable and therefore cannot remove any within-document length effect; whatever length bias is present in `SUS_i` before min-max is present, in the same rank order, after it. If document-length neutrality is achieved anywhere in this pipeline it must come from the TF-IDF step (i.e. from `norm='l2'`), which is exactly the thing the paper does not disclose. That is a clean, well-evidenced line for the critique: the paper's only stated defence against length bias does not do the job it is said to do, and the step that could do the job is undocumented.

### 3e. Summary table for point 3

| Question | Answer | Evidence level |
|---|---|---|
| Which implementation/library? | scikit-learn `TfidfVectorizer`, v0.24.2 | **VERIFIED** (p. 3, p. 4) |
| IDF weighting used, or raw TF? | IDF used | **VERIFIED** (p. 4) |
| `max_df` / `min_df` / `max_features` / `ngram_range` | 0.95 / 2 / 10,000 / (1,2) | **VERIFIED** (p. 4, prose) |
| Document vectors L2-normalised? | **Paper never says.** `norm` never mentioned. | **NOT STATED** |
| Aggregation over vocabulary: sum or mean? | **Paper never says.** No formula for `SUS_i`. | **NOT STATED** |
| `sublinear_tf`, `smooth_idf` | Never mentioned | **NOT STATED** |
| Length normalisation applied? | Only claim is that *min-max on the final scores* handles length (quoted above). Nothing about TF-IDF-level length handling. | **VERIFIED** (the claim); its adequacy is our critique, not the paper's |

---

## 4. Keyword vocabulary — VERIFIED provenance, NOT STATED size, NOT published

From §3.2.1 (p. 4):

> "The analysis of ESGwashing in sustainability reports hinges on identifying key sustainability indicators categorized into three domains: Environmental, Social, and Governance (ESG). These categories are populated with specific keywords relevant to each sustainability aspect."

> "The process begins by establishing detailed lists of sustainability indicators for governance, environmental impact, and social impact. **These lists include terms like 'Diversity', 'Carbon Emissions', and 'Equal Employment Opportunities'**, aiming to cover a wide range of themes frequently discussed in sustainability reports. **These indicators are based on established ESG reporting frameworks such as Global Reporting Initiative (GRI) Standards (Global Reporting Initiative, 2021) and Sustainable Accounting Standards Board (SASB) Standards (Sustainability Accounting Standards Board, 2018).**"

- **Source: VERIFIED** — author-constructed lists grounded in GRI Standards (2021) and SASB Standards (2018), split into E / S / G.
- **Size: NOT STATED.** No count of terms, per-category or total.
- **Published? No.** There is no appendix, no table of keywords, no supplementary material. Only the three example terms above appear anywhere in the paper. Note that two of the three examples ("Carbon Emissions", "Equal Employment Opportunities") are bigrams, consistent with `ngram_range=(1,2)`.
- **Data availability statement (p. 10), verbatim and in full:** "Data availability / No". No code repository, no data deposit, no keyword list.

The full vocabulary is therefore **irrecoverable from the paper**. Any replication necessarily substitutes its own list, and differences in ESGSI between our results and Lagasio's cannot be attributed cleanly to any single design choice.

---

## 5. Corpus — VERIFIED (and our prior belief was wrong)

From §3.1 (p. 3):

> "The initial sample included the **top 2000 companies in terms of market capitalization listed worldwide, retrieved from Refinitiv**. To ensure global coverage, data were collected from companies operating in various geographical regions and multiple sectors, including manufacturing, technology, energy, healthcare, and finance."

> "**We manually collected standalone ESG reports published in 2023**, which resulted in a significant reduction of our sample size from 2000 to 749 companies. This reduction was necessary to maintain consistency and comparability in our analysis. Specifically, we applied several criteria for inclusion in our final sample. **Only reports published in English** were considered to ensure consistent application of our natural language processing techniques. **All reports had to pertain to the 2023 reporting year** to provide a contemporaneous snapshot of ESG disclosure practices. **We included only standalone ESG reports; companies that integrated their ESG disclosure within their annual financial reports were excluded** to maintain structural consistency across our sample and to focus solely on dedicated sustainability reporting. Additionally, reports had to be complete and accessible for comprehensive analysis."

- **Document type: standalone ESG / sustainability reports.** **NOT annual reports** — firms disclosing ESG inside the annual report were deliberately excluded. Our project brief describes the corpus as "corporate annual reports"; that is incorrect and should be fixed.
- **Size:** 749 companies, one document each, from an initial 2,000 (p. 3, restated p. 3: "The final sample covers 749 companies listed worldwide"). n = 749 confirmed in Table 2's `count` row (p. 6).
- **Year:** single year, **2023**. Cross-sectional.
- **Countries/regions:** worldwide, grouped in Table 3 (p. 7) into just four buckets — **Europe, North America, South America, Other**. No country-level breakdown is reported. (Note: search engines associate country-level greenwashing findings for Portugal/Czechia/Argentina with a *different* paper on Central and Eastern European firms — those are **not** findings of this paper.)
- **Sectors:** GICS sectors, 11 reported in Table 4 (p. 7): Communication Services, Consumer Discretionary, Consumer Staples, Energy, Financials, Health Care, Industrials, Information Technology, Materials, Real Estate, Utilities.
- **Firm-level covariates:** total assets (quartiles, Table 5) and Refinitiv ESG score (quartiles, Table 6 — "ESG scores (collected from Refinitiv)", p. 8).
- **Language:** English only.

**Directly relevant to our temporal extension** — the paper explicitly declines to build a panel (§3.1, p. 3):

> "It's worth noting that for the purposes of our study, which aims to define and calculate the ESGSI, **a cross-sectional dataset focused on a single year (2023) is sufficient. Our methodology does not require a panel dataset spanning multiple years**, as the index calculation is designed to provide a robust measure of potential ESGwashing based on a single, comprehensive ESG report."

This matters for our design: because both components are normalised **across the cross-section** (min-max in practice, z-scores as printed — either way, corpus-relative), the index has no fixed cardinal meaning across corpora. Extending it over time requires deciding whether to normalise within each year (making levels incomparable across years, since the reference distribution moves) or pooled across years. The paper offers no guidance, and the ESGSI as defined is not year-comparable without such a decision. The paper's own limitations section flags the need for longitudinal work (§8) but does not address this normalisation problem.

**Preprocessing pipeline — VERIFIED (§3.1.1, p. 3):** Python; NLTK **3.6.2** for tokenisation, stopword removal and lemmatisation (`WordNetLemmatizer`); NLTK's default English stopwords **augmented with a custom list** of "additional terms frequently found in corporate literature" (**the custom list is NOT published**); then (1) URL removal, (2) lowercasing and removal of numerals and punctuation via regex, (3) tokenisation with removal of stopwords and **tokens shorter than three characters**, (4) lemmatisation.

---

## 6. Greenwashing label and threshold — VERIFIED, with no statistical justification

The threshold is **exactly zero**, and it is definitional. From §3.3 (pp. 4–5):

> "This index attempts to capture discrepancies between the sentiment of the communication and the actual emphasis on sustainability. **If the sentiment polarity is higher than the sustainability content score, suggesting a positive ESGSI, it indicates that the sentiment might be overly positive compared to the actual sustainability content.** This could suggest potential ESGwashing, where the communication may be attempting to appear more sustainable or responsible than it is substantively supported by the document's content. **Conversely, a non-positive index suggests that the sentiment aligns with or is less than the factual sustainability content, indicating likely genuine sustainability efforts.**"

> "**The script labels each document based on its ESGSI, categorizing them as either "Potential ESGwashing" or "Likely Genuine."** These labels help stakeholders, including investors, regulators, and the public, discern the authenticity of a company's sustainability claims."

Also (§3.2.3, p. 4): "A positive index indicates potential ESGwashing ... while a negative or zero index suggests more authentic sustainability communications."

- **Threshold: ESGSI > 0 → "Potential ESGwashing"; ESGSI ≤ 0 → "Likely Genuine".** VERIFIED. Matches our reconstruction.
- **Justification: NOT STATED beyond the definitional appeal above.** There is no external validation against known greenwashing cases, no ROC/precision-recall analysis, no calibration exercise, no sensitivity of the classification to the cut-off, and **no reported count or percentage of firms falling above it anywhere in the paper**.

**Two observations of ours (not the paper's) that the critique can use, both grounded in quoted material.** First, zero is only a meaningful cut-off if the two components are on a genuinely comparable scale; under min-max both are on [0,1] but their *distributions* differ sharply — Table 2 (p. 6) gives NormSUS mean 0.5906 vs NormSEN mean 0.3727, and the paper itself notes the sustainability distribution "appears to be skewed towards higher values" (p. 6) while sentiment is roughly symmetric. The zero threshold therefore inherits that asymmetry directly. Second, and consequently, the paper's headline substantive claim is an artefact of that asymmetry: with mean ESGSI = −0.2179 and max = +0.5449 (Table 2), the great majority of firms are below zero, which the paper reads as evidence of honesty —

> "The ESGSI predominantly shows negative values, **suggesting a general absence of greenwashing practices among the entities studied**." (p. 6)

Under the *printed* z-score formula the mean would be 0 by construction and roughly half the sample would be flagged. So the paper's central empirical conclusion depends entirely on the undisclosed normalisation choice, and flips under the formula the paper itself prints. This is, in our judgement, the strongest single point available for the critique, and every element of it is quoted or arithmetic.

---

## 7. Robustness checks and validation — announced, **never reported**

The complete robustness passage, verbatim (p. 5, end of §3.3):

> "To ensure the reliability and validity of our analysis, we implement several measures. **For the manual data collection process, we employ two independent raters and calculate Cohen's Kappa to assess agreement** (Cohen, 1960). **We use 5-fold cross-validation to evaluate the robustness of our LDA and sentiment analysis models** (Kohavi, 1995). **We also conduct sensitivity analyses by varying key parameters (e.g., number of LDA topics, TF-IDF settings) to test the robustness of our results.** These measures align with best practices in NLP and machine learning research (Goodfellow et al., 2016)."

**No result from any of these three is reported anywhere in the paper.** I searched the full text for `kappa`, `cross-valid`, `sensitivity`, `rater`, `validat`, `robust`: the passage above is the only occurrence of each, apart from generic uses of "robust" in the introduction/conclusion and the Kohavi/Cohen reference entries. Specifically:

- **No Cohen's Kappa value is given.** NOT STATED.
- **No cross-validation results** (no accuracy, coherence or perplexity figures). NOT STATED.
- **No sensitivity analysis is shown** — nothing on how ESGSI moves when the TF-IDF settings or LDA topic count change. NOT STATED. This is the one that would have answered our aggregation question indirectly, and it is absent.

**The only inferential statistics actually reported** are the one-way ANOVAs of ESGSI on corporate characteristics with Tukey HSD post-hoc tests and eta-squared effect sizes (§4.2.5 and Table 7, p. 9): Continent η² = 0.0746 (strongest), GICS Sector η² = 0.0512, ESG Score Category η² = 0.0407, Assets Category η² = 0.0249. These validate *correlates* of the index, not the index itself.

**There is no validation of the ESGSI against any external greenwashing benchmark anywhere in the paper.** No comparison with third-party greenwashing measures, ESG controversy scores, regulatory findings, or hand-labelled cases. The index's construct validity rests entirely on its definition.

---

## 8. Limitations the author acknowledges — VERIFIED (p. 10, Conclusion)

Verbatim:

> "However, this study also has limitations that point to promising avenues for future research. **The cross-sectional nature of our data limits causal inferences, highlighting the need for longitudinal studies to examine how changes in corporate characteristics and regulatory environments affect ESGSI over time.** Future research could also delve deeper into the specific ESG practices that are most prone to washing, investigate the mechanisms behind geographic differences in ESGSI scores, and explore potential non-linear relationships between ESG performance and ESGSI."

> "Furthermore, **the development and validation of more sophisticated ESGSI measures could enhance our ability to detect and understand nuanced forms of ESGwashing** across different sectors and company sizes. This could involve incorporating additional data sources, **refining sentiment analysis techniques**, or developing sector-specific ESGSI variants."

That is the **entire** limitations discussion — two paragraphs, both in the conclusion. Notably:

- **Acknowledged:** cross-sectional design limits causal inference; longitudinal work needed (this is precisely our project's contribution, and the paper invites it explicitly); ESGSI measures could be more sophisticated; sentiment analysis technique could be refined (this invites our Loughran-McDonald substitution).
- **NOT acknowledged anywhere:** the min-max / z-score contradiction; the undisclosed TF-IDF aggregation; the absence of the keyword list; English-only selection bias; the survivorship/selection bias from dropping 1,251 of 2,000 firms; the arbitrariness of the zero threshold; the use of TextBlob (a general-purpose, non-financial sentiment tool) on financial text; the absence of any external validation.

---

## 9. Additional internal inconsistencies found (relevant to a replication)

These are all VERIFIED by direct comparison of the paper's prose against its own tables. They bear on how much weight the published results can carry.

1. **Energy sector, direction reversed.** §4.2.5 (p. 9): *"the Energy sector has significantly **higher** ESGSI scores compared to Health Care (mean difference = 0.1174, p = 0.05) and Industrials (mean difference = 0.1224, p = 0.05)"*. But Table 4 (p. 7) gives Energy = **−0.3117**, Health Care = **−0.1943**, Industrials = **−0.1893** — Energy is the *lowest* of the three, and the differences are −0.1174 and −0.1224 (correct magnitude, opposite sign). The conclusion (p. 9) then builds on the wrong sign: *"sectors facing higher environmental scrutiny, such as Energy, showing a greater propensity for ESGwashing."*

2. **Energy sustainability score, direction reversed.** §4.2.2 (p. 8): *"Sectors such as Materials & Chemicals and Energy & Utilities exhibit some of the **lowest** scores in sustainability but have a more pronounced negative ESGSI."* Table 4 gives Energy the **highest** NormSUS in the table (0.6595) and Materials the second highest (0.6187).

3. **Prose cites a table that is not printed.** §4.2.2 (p. 8): *"in the Consumer Goods industrial sectors, the mean ESGSI is −0.3442, the mean Normalized Sustainability Score is 0.5730"*. Table 4 contains **no "Consumer Goods" row** (only Consumer Discretionary, −0.1802, and Consumer Staples, −0.2089), and neither figure appears anywhere in it. The section also refers to sector names ("Energy & Utilities", "Financial Services", "Industrial & Manufacturing", "Transportation & Logistics") that do not match Table 4's GICS labels — suggesting the text describes an earlier, differently-classified version of the analysis.

4. **ESG-quartile narrative contradicts its own conclusion.** §4.2.4 (p. 8) says ESGSI *"decreases (more negative) as ESG performance decreases (from 1st to 4th Quartile)"* and that this *"suggests that entities with lower ESG ratings may be perceived as more likely to engage in ESGwashing"* — but more negative means *less* washing on the paper's own scale. §4.2.5 then states the opposite and correct reading (*"companies with the highest reported ESG scores are less likely to engage in ESGwashing"*), which is what Table 6 supports (Q1 = −0.1793, Q4 = −0.2695 — i.e. the *highest* ESG performers have the *highest*, least negative, ESGSI, which on the paper's scale means *more* washing, contradicting §4.2.5 as well).

5. **Table 7 is internally impossible.** Assets Category is reported as F = 6.165754 with **p = 3.86E-01** (i.e. 0.386, non-significant) and labelled "Small effect" — yet §4.2.5 (p. 9) claims *"Companies with High asset levels have significantly higher ESGSI scores compared to those with Low asset levels (mean difference = 0.0493, p = 0.05)"*. An F of 6.17 across four asset quartiles with n = 749 cannot yield p = 0.386. The F-statistics are also printed with mangled separators ("3.979.157", "20.022.260", "10.534.781", "6.165.754"), evidently European decimal formatting for 3.979157, 20.022260, 10.534781, 6.165754 — the same formatting fault that renders 1.0000 as "10.000" in Table 2.

6. **Sentiment analysis is described twice, in the wrong section.** The TextBlob paragraph appears at the end of §3.2.2 (LDA) *and* again as §3.2.3, with the intervening text about gensim dictionaries belonging to LDA rather than sentiment.

7. **Fabricated-looking references.** Several cited works could not be matched to real publications and have generic author names and titles — e.g. "Smith, J. A., & Johnson, M. B. (2020). Environmental, social, and governance (esg) performance metrics. *Journal of Sustainable Business*, 25(3), 45–58", "Jones, R. C., & Brown, S. L. (2019). Measuring esg disclosure quality. *Journal of Corporate Responsibility*, 12(2), 78–93", "Khan, S., & Serafeim, G. (2021). The greenwashing phenomenon. *Journal of Environmental Economics*, 48(2), 315–331", "Liu, X., & Bohnsack, R. (2022). Textual indicators of greenwashing in esg disclosure. *Journal of Sustainable Finance*, 15(1), 112–128", "Perez-Baltres, J. (2021). Sectoral differences in greenwashing tendencies. *Corporate Social Responsibility Review*, 18(3), 234–248", "Miles, E., & Covin, J. G. (2021)... *Journal of Business Ethics*, 35(4), 589–607", "Vanclay, F., Shortiss, F., & Wilson, K. (2020)... *Business Strategy and the Environment*, 29(5), 2345–2361", "Wang, L., & Marquis, C. (2020)... *Strategic Management Journal*, 41(8), 1326–1349", "Chatterji, A. K., & Toffel, M. W. (2019). Esg performance and greenwashing. *Strategic Management Journal*, 40(5), 885–889". Also the greenwashing-origin footnote cites "Westerveld, J. (1986). *Tiny Caller–the Northern Cricket Frog*. New York Department of Environmental Conservation." I have **not** independently checked these against publisher databases in this session — flagged as **UNVERIFIED, requires confirmation** before any use. It would be a serious allegation and must be checked title-by-title first. Note that the genuinely load-bearing methodological citations (Pedregosa, Bird, Blei, Řehůřek & Sojka, Loria, Loughran & McDonald, Cohen, Kohavi, Röder) all appear legitimate.

---

## 10. Open questions that the paper cannot settle

These require something beyond the published article — the SSRN working paper, the author's code, or direct correspondence.

1. **Was `norm='l2'` in force?** The paper never states the `norm` parameter. Everything about whether the sustainability score is length-neutral turns on this.
2. **Sum or mean over the ESG keyword vocabulary?** Never stated. Without it, `SUS_i` is not reproducible, and our critique of the aggregation cannot be attributed to the paper.
3. **Was the score restricted to the ESG keyword list at all, or taken over the full 10,000-term vocabulary?** §3.2.1 describes both a 10,000-term vectoriser vocabulary and separate E/S/G indicator lists, but never says how the two interact — whether TF-IDF is computed on the full vocabulary and then subset to the keyword list, or the vectoriser is restricted to the keywords.
4. **Which normalisation was actually used?** The prose (min-max) and the numbers (min-max) agree against the printed formula (z-scores). Is the printed formula an editorial/typesetting error, or was the formula the intent and the code the error? Only the author or the code can say.
5. **How many keywords, and what are they?** Size and content both undisclosed; three examples only.
6. **What is in the custom stopword list** added to NLTK's defaults? Undisclosed, and it could overlap with ESG vocabulary.
7. **What fraction of the 749 firms were labelled "Potential ESGwashing"?** Never reported, despite being the paper's headline construct.
8. **Results of the announced Cohen's Kappa, 5-fold CV, and parameter sensitivity analyses?** Announced on p. 5, never reported. The sensitivity analysis explicitly covered "TF-IDF settings" and would bear directly on Q1 and Q2.
9. **Does the SSRN working paper (abstract id 4582917) specify the aggregation or the `norm` setting?** Unverified — SSRN returned HTTP 403 in this session. **Highest-value next step.**
10. **Was the sentiment score computed on the preprocessed text or the raw text?** §3.1.1 strips stopwords, punctuation and numerals and lemmatises; TextBlob's pattern-based polarity depends on intensifiers and negations that this pipeline destroys. The paper never says which version of the text reached TextBlob.

---

## Appendix: how to re-derive this

Full text extracted with:

```
python -c "import fitz; d=fitz.open('paper/1-s2.0-S1057521924006744-main.pdf'); print(''.join(p.get_text() for p in d))"
```

(PyMuPDF is available in the project environment; `pdftoppm`/poppler is not installed, so the Read tool cannot render this PDF directly.) All quotes above were taken from that extraction and are reproduced verbatim including the paper's own typographical quirks. Section and page attributions come from the PDF page boundaries, which match the article's printed page numbers 1–10.
