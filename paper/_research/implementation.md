# Implementation reference — ESGSI replication and extension

Reference material for the methods/results sections. Every number below was re-derived from
the repository files on 2026-08-02 unless explicitly marked. Verification scripts are
throwaway (scratchpad); the numbers, not the scripts, are the deliverable.

**Conventions used in this document**
- `[V]` verified against repo files/data in this pass.
- `[!]` discrepancy with the briefed figure — see the flag text.
- `[NV]` claim present in the repo (commit message / comment) that could **not** be
  independently verified because the supporting file is not in the repository.

---

## 0. Repository state at time of writing

| Item | Value |
|---|---|
| Branch | `fix/lexicon-pipeline-defects` |
| HEAD | `44114d0` *fix: repair ESG lexicon pipeline and restore audited vocabulary* |
| Uncommitted modifications | `main.py`, `src/config.py`, `src/esgsi_analyzer.py`, `.gitignore` — this is the three-SUS-specification work; `results.csv` was regenerated after it |
| `results/metrics/results.csv` mtime | 2026-08-02 16:43 |
| Runtime | conda env `esgwashing`; deps (`pyproject.toml`): pandas, openpyxl, pymupdf, spacy, scikit-learn, loguru, gensim, pysentiment2 |
| Determinism | No RNG anywhere on the ESGSI path; no network calls; no external API. `LexicalDocumentFilter` → `TextProcessor` → `ESGSIAnalyzer` is a pure function of the PDFs + the two lexicon files. |

### Pipeline evolution (git)

| Commit | Change |
|---|---|
| `708b9f8` … `813005c` | Original pipeline: fixed-size chunking (1,200 chars, 300 overlap) + **semantic** filter using Vertex AI `text-embedding-004`, cosine similarity against hand-written "anchor queries" |
| `64393f2` | `SemanticDocumentFilter` **deleted**; replaced by `LexicalDocumentFilter` (deterministic, no GCP). `LM_dictionary_lemmatized.csv` (6,000 rows) dropped in favour of `esg_terms.txt` as single source of truth for the ESG vocabulary. All GCP deps removed. |
| `44114d0` | Three silent zero-measurement defects fixed (see §6.5). Vocabulary restored 32 → 154 terms. `scripts/build_lexicons.py` introduced. |
| working tree | Three SUS specifications, `Breadth`, `calculate_sus_variants()` |

Outputs of the retired semantic pipeline are preserved at
`data/_backup_pipeline_semantico/{processed_texts_semantico.csv, results_semantico.csv}`
(343 data rows; only `SUS_Score`, `SEN_Score`, `ESGSI`, `Etiqueta` — no QUANT/HEDGE). Usable
as a "filter-choice" robustness comparison if wanted.

---

## 1. Corpus

### 1.1 Composition [V]

| Dimension | Value |
|---|---|
| PDFs under `data/pdf/` | 344 |
| Extraction JSONs under `data/chunks_lexical/` | 344 |
| Rows in `results/metrics/results.csv` | 344 |
| Companies | 49 |
| Countries | 7 |
| Years | 2018–2024 (7) |
| Unique (company, year) pairs | **343** |

Country folder names are Spanish; `metadata_loader.py` maps them
(ALEMANIA→GERMANY, BELGICA→BELGIUM, ESPAÑA→SPAIN, FINLANDIA→FINLAND,
FRANCIA→FRANCE, ITALIA→ITALY, PAISES BAJOS→NETHERLANDS).

**Documents and companies per country** [V]

| Country | Documents | Companies |
|---|---:|---:|
| FRANCIA | 119 | 17 |
| ALEMANIA | 98 | 14 |
| PAISES BAJOS | 43 | 6 |
| ITALIA | 35 | 5 |
| ESPAÑA | 28 | 4 |
| FINLANDIA | 14 | 2 |
| BELGICA | 7 | 1 |

**The panel is exactly balanced**: each country contributes the same number of documents in
every year (14/1/4/2/17/5/6), except Netherlands = 7 in 2019. Docs per year: 49 for every
year except 2019 = 50.

### 1.2 `[!]` Duplicate document — the 344th row

The extra 2019 Netherlands document is a **duplicate of the same report**:

```
2019-Annual-Report-Print-version.json           PAISES BAJOS  Wolters Kluwer NV  2019
2019-Annual-Report-Print-version_repaired.json  PAISES BAJOS  Wolters Kluwer NV  2019
```

Both rows carry **identical values in every column** (SUS_density 7.763544, SUS_lagasio
0.019365, Breadth 16.3813, SEN 0.1671, ESGSI 0.1946, ESGSI_ext 0.0991). The `_repaired`
filename indicates a re-saved copy of the same PDF that was never removed from `data/pdf/`.

Consequence: N is effectively **343 unique reports**; one report is double-weighted in every
mean, correlation and z-score normalisation. Effect size is negligible (1/344), but the
paper should state N = 344 documents / 343 unique reports, or drop the duplicate and rerun.

### 1.3 Document type [V]

`TipoDocumento` comes from `metadata/muestras_informes.xlsx` (sheet index 1), joined on
(country, normalised company name, year) by `src/metadata_loader.py`. Company names differ
between folder tree and Excel, so a 45-entry manual map (`_COMPANY_MAP`) plus Unicode-fold /
punctuation-strip normalisation is applied. No document resolves to `Unknown`.

| Type | N |
|---|---:|
| Annual report | 236 |
| URD (*Document d'enregistrement universel*) | 101 |
| Informe de sostenibilidad | 7 |

**Type × country** [V] — near-total confound:

| | Annual report | Sustainability | URD |
|---|---:|---:|---:|
| ALEMANIA | **98** | 0 | 0 |
| BELGICA | 7 | 0 | 0 |
| ESPAÑA | 28 | 0 | 0 |
| FINLANDIA | 14 | 0 | 0 |
| FRANCIA | 13 | **7** | **99** |
| ITALIA | 35 | 0 | 0 |
| PAISES BAJOS | 41 | 0 | 2 |

URD is 99/101 French; Germany is 98/98 annual reports; all 7 sustainability reports are
French and from a single company-year series.

**Type × year** [V] — composition is stable, which is the composition-artefact rebuttal:

| Year | Annual report | Sustainability | URD |
|---|---:|---:|---:|
| 2018 | 36 | 1 | 12 |
| 2019 | 34 | 1 | 15 |
| 2020 | 33 | 1 | 15 |
| 2021 | 33 | 1 | 15 |
| 2022 | 34 | 1 | 14 |
| 2023 | 33 | 1 | 15 |
| 2024 | 33 | 1 | 15 |

### 1.4 Physical size of source documents [V]

Random 20-PDF sample (seed 7): mean **442 pages**, range 164–1,160; mean 226,196 words of
raw PyMuPDF text per PDF, range 79,144–698,488.

---

## 2. Document extraction — `src/lexical_document_filter.py`

Fully deterministic keyword-driven ESG-zone extraction. No chunking into fixed windows; the
output units are text zones whose length is set by document structure.

### 2.1 PDF → paragraphs

PyMuPDF `page.get_text("blocks")`, text blocks only (`block[6] == 0`), sorted by (y, x).

| Step | Rule |
|---|---|
| Margin strip | Drop block if `y1 < 0.08·page_height` or `y0 > 0.92·page_height` |
| Running header/footer | Digits → `#`, blocks ≤120 chars inside the margin bands, counted across pages; any normalised string appearing in **>30%** of pages is blacklisted corpus-wide for that PDF |
| Financial-table filter | Drop if no ESG keyword present **and** (digit ratio >0.35 with ≤6 lines) **or** (all lines <60 chars, ≥2 lines, first line has no terminal punctuation) |
| Section headers | All-caps, ≥2 letters, ≤10 tokens, <60% single-char tokens, no filename-like token → emitted as a `§ TITLE §` marker, not as a paragraph; used to tag each paragraph with its section |
| Cleaning | Strip URLs; rejoin `word-\nword`; ASCII-fold (`encode('ascii','ignore')`); collapse whitespace |

Character offsets are tracked (`start_char`/`end_char`, `+2` per paragraph separator) so zones
can be located back in the document.

### 2.2 Paragraph scoring and zone construction

- **Keyword regex over raw text** (`_ESG_KW_RE`): the 154 terms of `metadata/esg_terms.txt`,
  each compiled with `[\s\-]+` between words (so "net zero" matches "net-zero") and an
  optional trailing `s?` (plural), sorted longest-first so that alternation does not let
  `carbon` shadow `carbon neutrality`; **plus 24 `_EXTRA_PATTERNS`** [V] that cannot live in
  the flat term file:
  - morphological families: `environ\w+`, `sustainab\w+`, `decarboni\w+`, `recycl\w+`,
    `offset\w+`, `electrif\w+`, `inclusiv\w+`, `circulari\w+`, `cybersecuri\w+`,
    `whistleblow\w+`, `injur\w+`
  - digit-bearing terms: `co2e?`, `scope\s*[123]`, `scope\s+(one|two|three)`,
    `article\s*[689]`, `ifrs\s*s[12]`, `iso\s*(14001|45001|50001|37001)`,
    `tonne\w*\s+(co2|carbon)`
  - precision collocations for polysemous heads: `social + {responsibility, impact, policy,
    pillar, report, performance, value, welfare, audit, capital, sustainability, license,
    licence, dialogue, protection, partner, standard, norm, commitment, compliance}`,
    `water + {usage, consumption, stewardship, management, withdrawal, scarcity, stress}`,
    `energy + {consumption, efficiency, transition, mix, intensity, storage}`,
    `employee well\w*`
  - `paris agreement`, `just transition` — moved here in `44114d0` because preprocessing
    collapses them to the useless unigrams "agreement"/"transition"
- **Hot paragraph**: keyword density ≥ `LEXICAL_KW_THRESHOLD = 1.0` per 100 words
- **Context window**: `LEXICAL_CONTEXT_PARAS = 1` paragraph either side
- **Merge**: overlapping or adjacent windows (`lo <= prev_hi + 1`) merged
- **Minimum zone**: `LEXICAL_MIN_ZONE_LEN = 150` characters
- Output: one JSON per PDF mirroring the country/company tree, containing per-zone
  `{text, start_char, end_char, section, kw_count, keywords_found, word_count, kw_density}`
  plus a `relevant_text` field = zones joined by `\n\n\n\n` (this is what the next stage reads)

### 2.3 Extraction yield [V]

Aggregated over all 344 JSONs:

| Statistic | Value |
|---|---|
| Total zones | 74,004 |
| Zones per document | mean 215.1, sd 152.9, median 173, min 9, max 1,148 |
| Words per document (extracted) | mean 76,743, sd 46,321, median 69,960, min 1,619, max 246,666 |
| Total extracted words | 26,399,487 |
| Total keyword matches (raw-text regex) | 965,646 |
| Corpus-wide keyword density | **3.658 per 100 words** of extracted text |
| Mean per-zone keyword density | 3.03 per 100 words |
| Mean words per zone | 396.3 |

**Yield relative to the full PDF** (random 20-PDF sample): mean **35.5%**, median 35.7%,
range 9.0%–59.6% of the PDF's word count survives into the ESG zones.

**Extraction volume by year** [V] — the extraction stage is where the temporal signal originates:

| Year | zones/doc | extracted words/doc | kw matches/doc | words/zone | kw density |
|---|---:|---:|---:|---:|---:|
| 2018 | 178.5 | 51,289 | 1,560.6 | 322.4 | 2.61 |
| 2019 | 179.8 | 54,337 | 1,688.9 | 342.3 | 2.74 |
| 2020 | 199.1 | 64,735 | 2,133.3 | 382.8 | 2.85 |
| 2021 | 207.9 | 75,542 | 2,635.4 | 399.8 | 2.89 |
| 2022 | 216.5 | 84,763 | 3,069.2 | 434.4 | 3.00 |
| 2023 | 255.6 | 92,144 | 3,585.2 | 426.3 | 3.31 |
| 2024 | 275.2 | 115,357 | 5,100.3 | 463.4 | 3.93 |

---

## 3. Preprocessing — `src/text_processor.py`

spaCy `en_core_web_md`, pipeline with `ner` and `parser` disabled, `nlp.max_length = 5_000_000`.

Applied to `relevant_text` from each JSON:
1. strip URLs (`https?://\S+|www\.\S+`)
2. lowercase
3. spaCy parse; keep token iff **all** of:
   - not `is_stop`, not `is_punct`, not `is_space`
   - `_is_content_token`: `text.isalnum()` **and** ≥2 alphabetic characters
     (replaced the old `token.is_alpha` filter, which silently deleted `co2`, `co2e`, `sf6`,
     `iso14001`; the two-letter rule still excludes pure figures `2023`, `14001` and page refs
     `p12`, `q1`)
   - `len(text) > 2`
   - `lemma_` not in `PERSONAL_SW` (189 entries, `metadata/personal_stopwords.txt`)
4. emit `" ".join(lemma_)`

Two text representations are carried forward per document:

| Column | Content | Consumers |
|---|---|---|
| `clean_text` | lemmatised, stopword-filtered token string | SUS, HEDGE, Breadth, LDA |
| `raw_text` | extracted text with whitespace collapsed only (digits intact) | **QUANT** |

Saved to `data/clean/processed_texts.csv` (`;`-separated, 293 MB).

### 3.1 Corpus size after preprocessing [V]

| | Lemmatised (`clean_text`) | Raw (`raw_text`) |
|---|---:|---:|
| Total tokens | 14,404,913 | 26,195,710 |
| Mean per document | 41,874.7 | 76,150.3 |
| Median | 38,050.5 | — |
| Min / Max | 905 / 137,289 | — |

Preprocessing removes ~45% of the extracted tokens.

---

## 4. Lexicon architecture

Four lexicons, three of them generated. The design principle stated in the code: the
vocabulary and the corpus must be in the **same representation by construction**, so every
term is passed through the *same* `TextProcessor` that processes the corpus.

```
metadata/esg_terms.txt  (154 canonical terms, hand-curated, single source of truth)
   ├─→ lexical_document_filter._ESG_KW_RE   (raw-text regex; + 24 _EXTRA_PATTERNS)
   └─→ scripts/build_lexicons.py ──→ metadata/esg_terms_lemmatized.txt  (154 entries)
                                          └─→ config.ESG_KEYWORDS  (SUS vocabulary)

metadata/RAW_LM_dictionary.csv (9,752 rows)
   └─→ scripts/build_lexicons.py ──→ metadata/lm_hedge_lemmatized.txt (1,263 forms)
                                          └─→ config.HEDGE_KEYWORDS

metadata/personal_stopwords.txt (189)   → TextProcessor
metadata/lda_stopwords.txt      (86)    → LDA only, does not touch ESGSI
pysentiment2 bundled LM lexicon         → SEN
```

`config._load_generated_lexicon()` raises `FileNotFoundError` if a generated file is missing
and logs a warning if the source file's mtime is newer than the generated file's — a staleness
guard, since these are build artefacts.

### 4.1 ESG vocabulary [V]

`esg_terms.txt`: **154** canonical terms, natural singular form, in 7 thematic blocks.

| Block | Terms |
|---|---:|
| Ambiental: clima y emisiones | 14 |
| Ambiental: energía | 18 |
| Ambiental: recursos y naturaleza | 17 |
| Sostenibilidad y economía circular | 12 |
| Social: personas y equidad | 34 |
| Gobernanza y ética | 25 |
| Marcos de reporting y ratings | 34 |
| **Total** | **154** |

`esg_terms_lemmatized.txt`: **154** entries (no collisions), n-gram distribution
**85 unigrams / 62 bigrams / 7 trigrams**, so `max_n = 3`. This value is derived at runtime
(`max(len(k.split()) for k in keywords)`) and passed as `ngram_range=(1, max_n)` to both
`CountVectorizer` and `TfidfVectorizer` — without it, sklearn's default `(1,1)` would have
scored every multi-word entry as zero with no error.

**Lemmatisation artefacts visible in the generated vocabulary** (worth acknowledging as a
known cost of the "same representation by construction" rule):
`datum breach`, `datum security`, `force labor`, `employee wellbee`, `ecovadi`, `esr` (from
*esrs*), `live wage`, `nature base solution`, `sustainalytic`, `whistleblowe`,
`science base target`, `supply chain diligence`, `health safety`, `global compact`,
`non financial`, `taxonomy align`, `diligence` (from *due diligence*).
`esr` is not cosmetic: it has idf 2.53 and 6,282 corpus occurrences, i.e. it is one of the
most heavily up-weighted terms in the IDF-weighted specifications.

**Coverage** [V]: every one of the 154 terms appears in ≥1 document (zero dead entries);
1 term (`sustainable`) appears in all 344; a document contains on average **87.63** distinct
vocabulary terms (median 89, min 19, max 133).

### 4.2 Lexicon validation

External audit stored as `feedback/Terminos_ESG_no_detectados.xlsx` [V for the workbook's own
content; the underlying zone workbooks are not in the repo]:

| Item | Value |
|---|---|
| Audited corpus | 880 zones from 4 reports of FY2023: Stellantis 208, adidas 115, Danone 343, Enel 214 |
| Classifier vocabulary as reconstructed by the auditor | 173 terms/roots (from the `Keywords` column) |
| Method | Extended ESG lexicon (GRI, SASB, TCFD, ESRS + literature) unioned with a prior audit glossary (122 terms); word-boundary regex over each zone; candidates already covered by the classifier vocabulary excluded |
| Result | **147 ESG terms present in the zones and absent from the classifier vocabulary; 2,970 total occurrences** (27 overlap with the prior audit, 120 new) |
| Top omissions | `compliance` 436 (G), `social` 275 (S), `supply chain` 241 (G), `raw material` 136 (E) |
| Prior-audit contrast | Prior audit found 30 distinct terms; all confirmed except 3 (`health and safety`, `risk management`, `scope 2`) which are *in* the vocabulary — they were per-zone labelling glitches, not dictionary gaps |
| Internal consistency | Only 3 of 880 zones contain a vocabulary term in the text without recording it in `Keywords` |
| Stated limitation | Restricted to text already inside the exported zones; lexical match only, no paraphrase/synonym detection |

`[NV]` Commit `44114d0` additionally claims the repaired pipeline "reproduces 878 of the 880
zones of the external audit, with adidas (115) and Enel (214) exact". The audit workbooks
(`comparativa_4_archivos_zones.xlsx`, `Auditoria_ESG_zones_2023.xlsx`) are **not in the
repository**, so this specific 878/880 figure could not be re-derived.

`[V]` The 147 additional terms are **deliberately not incorporated** — stated in the header of
`esg_terms.txt`: *"pendiente de análisis de sesgo sectorial"*.

### 4.3 HEDGE lexicon [V]

Source `metadata/RAW_LM_dictionary.csv`, 9,752 rows:
Negative 5,646 · Litigious 1,630 · Positive 1,231 · Uncertainty 767 · Constraining 432 ·
WeakModal 27 · StrongModal 19.

Categories kept (`HEDGE_CATEGORIES` in `build_lexicons.py`): **Uncertainty, WeakModal,
StrongModal, Constraining** = 1,245 rows → 1,196 unique lowercase forms. The generated file
is the **union of the original form and its spaCy lemma** → **1,263 forms** (+67).

Rationale in the code: the original L&M dictionary is in inflected forms, and crossing it
against a lemmatised corpus matched only ~46% of its entries (commit `44114d0`).

### 4.4 SEN lexicon [V]

`pysentiment2.LM()`, bundled Loughran–McDonald list, **Porter-stemmed at load time**:
140 positive stems, 893 negative stems. Note the 6:1 asymmetry.

---

## 5. Index components — exact formulas

Let `d` index documents (n = 344), `j` index the 154 vocabulary entries.
`c_dj` = raw count of vocabulary entry `j` in document `d` (`CountVectorizer`,
`vocabulary=ESG_KEYWORDS`, `ngram_range=(1,3)`, on lemmatised text).
`L_d` = `max(len(clean_text.split()), 1)`; `R_d` = same on `raw_text`.

### 5.1 Z-score

```
z(x)_d = (x_d - mean(x)) / sd(x)      sd = np.std (population, ddof = 0)
z(x)   = 0 vector if sd(x) == 0
```
Corpus-relative by construction: the z-scores are computed over the 344 documents actually
analysed, so all indices are **relative rankings within this corpus**.

### 5.2 SUS — three specifications

**(a) `lagasio`** — replication of the original:
```
idf_j       = ln((1 + n) / (1 + df_j)) + 1                (sklearn smooth_idf default)
w_dj        = c_dj · idf_j
tfidf_dj    = w_dj / ||w_d||_2                            (sklearn norm='l2' default)
SUS_lagasio_d = (1/154) · Σ_j tfidf_dj
```
Implementation: `self.vectorizer.fit_transform(texts).toarray().mean(axis=1)`.
Note: dividing by the constant 154 is irrelevant to the index, because the z-score cancels
any positive affine rescaling. Sum vs mean is a non-issue.

**(b) `tfidf_length`** — Lagasio's weighting, corrected normalisation:
```
SUS_tfidf_length_d = 100 · (Σ_j c_dj · idf_j) / L_d
```

**(c) `density`** — our proposal (default, `config.SUS_MODE = 'density'`):
```
SUS_density_d = 100 · (Σ_j c_dj) / L_d          [ESG mentions per 100 lemmatised words]
```
The only specification that is **corpus-independent**: no idf estimated on the sample, so the
value is comparable across studies. `(a)` and `(b)` both depend on the 344-document sample
through `df_j`.

All three are computed on every run (`calculate_sus_variants`) and written to `results.csv`;
`SUS_MODE` selects which one feeds `ESGSI` / `ESGSI_ext`. `SUS_Score == SUS_density` in the
current results file [V].

### 5.3 Breadth (reported, not in the index)

Effective number of distinct ESG terms — exponential of the Shannon entropy of the
term-share distribution:
```
p_dj      = c_dj / max(Σ_j c_dj, 1)
H_d       = -Σ_{j: p>0} p_dj · ln p_dj
Breadth_d = exp(H_d)                                       ∈ [1, 154]
```
Included because it is the dimension the Lagasio SUS turns out to measure (§6.2).

### 5.4 SEN

`pysentiment2.LM().get_score(LM.tokenize(clean_text))["Polarity"]`, i.e.
```
SEN_d = (s_pos - s_neg) / (s_pos + s_neg + ε)        ε = 1e-6
```
`s_pos` / `s_neg` = counts of Porter-stemmed tokens in the LM positive / negative stem sets.
Input is the **lemmatised** text, which the LM tokenizer then stems — double morphological
normalisation.

### 5.5 QUANT

Regex hit density on **raw** text (`config.QUANT_PATTERNS`, `re.IGNORECASE`):

| Group | Pattern |
|---|---|
| `percentages` | `\b\d+(?:[.,]\d+)?\s*%` |
| `large_numbers` | `\b(?!(?:19\|20)\d{2}\b)\d{4,}\b` — 4+ digits, years excluded |
| `units` | `\b(?:tonne\|ton\|mt\|ktco2\|co2e?\|ghg\|kwh\|mwh\|gwh\|twh\|mw\|gw\|litre\|liter\|m3\|cubic meter)\b` |
| `frameworks` | `\b(?:gri\|tcfd\|sasb\|issb\|sdg\|ungc\|sfdr\|csrd\|un global compact\|paris agreement\|taxonomy)\b` |

```
QUANT_d = (Σ_groups |matches|) / R_d
```

**Corpus totals** [V]: 195,291 hits.

| Group | Hits | Share |
|---|---:|---:|
| percentages | 109,974 | 56.31% |
| units | 38,176 | 19.55% |
| **frameworks** | **33,623** | **17.22%** |
| large_numbers | 13,518 | 6.92% |

Group composition shifts over time [V] — `frameworks` rises 10.15% (2018) → 21.87% (2024)
while `percentages` falls 64.81% → 49.00%:

| Year | percentages | large_numbers | units | frameworks |
|---|---:|---:|---:|---:|
| 2018 | 64.81 | 6.33 | 18.71 | 10.15 |
| 2019 | 67.73 | 7.73 | 15.54 | 8.99 |
| 2020 | 63.02 | 6.83 | 19.34 | 10.81 |
| 2021 | 58.89 | 5.77 | 20.10 | 15.24 |
| 2022 | 56.14 | 6.06 | 20.34 | 17.46 |
| 2023 | 52.90 | 6.20 | 19.30 | 21.61 |
| 2024 | 49.00 | 8.57 | 20.57 | 21.87 |

### 5.6 HEDGE

```
HEDGE_d = |{t ∈ tokens(clean_text_d) : t ∈ HEDGE_KEYWORDS}| / L_d
```
Plain membership test against the 1,263-form lemmatised L&M hedge lexicon.

### 5.7 Composite indices

```
ESGSI_d     = z(SEN)_d - z(SUS)_d
ESGSI_ext_d = z(SEN)_d - z(SUS)_d - 0.5·z(QUANT)_d + 0.5·z(HEDGE)_d
```
`w_quant = w_hedge = 0.5` (`config.ESGSI_EXT_WEIGHTS`). Sign logic in the code: QUANT high →
report carries hard data → less washing → subtract; HEDGE high → report evades concrete
commitment → more washing → add.

Labels: `ESGSI > 0` → "Potential ESG-washing", else "Likely Genuine"; same for `_ext`.

### 5.8 Descriptive statistics of all components (n = 344) [V]

| | mean | sd | min | 25% | 50% | 75% | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| SUS_density | 5.9026 | 1.9402 | 1.9948 | 4.4975 | 5.8126 | 7.1746 | 12.5527 |
| SUS_tfidf_length | 6.7588 | 2.3476 | 2.0937 | 5.0965 | 6.5991 | 8.2056 | 13.9114 |
| SUS_lagasio | 0.031698 | 0.006728 | 0.011662 | 0.027644 | 0.032990 | 0.036929 | 0.044674 |
| Breadth | 35.640 | 10.967 | 6.534 | 27.876 | 37.567 | 43.821 | 58.306 |
| SEN | 0.00515 | 0.14037 | −0.3728 | −0.0910 | −0.0038 | 0.08905 | 0.6495 |
| QUANT | 0.006654 | 0.003455 | 0.0000 | 0.004375 | 0.006550 | 0.008325 | 0.0331 |
| HEDGE | 0.087596 | 0.009610 | 0.0632 | 0.0813 | 0.0858 | 0.092675 | 0.1271 |
| ESGSI | 0.0000 | 1.3752 | −4.2249 | −0.8285 | −0.0039 | 0.7622 | 5.4861 |
| ESGSI_ext | 0.0000 | 1.8221 | −8.5618 | −0.9743 | −0.0949 | 1.2221 | 5.3940 |

Note `QUANT` min = 0: at least one document produces zero regex hits.

Label counts [V]: `Etiqueta` 172 washing / 172 genuine; `Etiqueta_ext` 165 / 179.
Recomputing from the **rounded** `ESGSI` column gives 171 > 0 — one document
(`2019 ANNUAL REPORT.json`) rounds to exactly 0.0000 and was labelled from the unrounded
value. Cosmetic, but the 172/171 gap will otherwise look like an error in a table.

---

## 6. The methodological critique of TF-IDF as a substance measure

### 6.1 L2 normalisation discards magnitude — the identity

With sklearn defaults, `tfidf_d = (c_d ⊙ idf) / ||c_d ⊙ idf||_2`. For any scalar k > 0,
`(k·c_d ⊙ idf) / ||k·c_d ⊙ idf||_2 = (c_d ⊙ idf) / ||c_d ⊙ idf||_2`. The representation is
**exactly invariant to the amount of ESG content**; only the *direction* of the vector —
the mix across terms — survives.

**Demonstration A — amplifying ESG content** [V]

A document is synthesised from the observed vocabulary counts of the corpus's first document
(adidas FY2018 annual report; 695 vocabulary hits) so that the counts can be scaled exactly
by k without perturbing n-gram structure, then embedded in the real 343 remaining documents
and re-vectorised end to end:

| k | tokens | ESG hits | density /100 | **SUS_lagasio** | SUS_tfidf_length |
|---:|---:|---:|---:|---:|---:|
| 1 | 16,556 | 700 | 4.2281 | **0.034912290512** | 5.0166 |
| 2 | 18,112 | 1,400 | 7.7297 | **0.034912290512** | 9.1713 |
| 5 | 22,780 | 3,500 | 15.3644 | **0.034912290512** | 18.2299 |
| 20 | 46,120 | 14,000 | 30.3556 | **0.034912290512** | 36.0172 |

Identical to 10 decimal places; maximum absolute deviation **6.9 × 10⁻¹⁸** (machine epsilon).
Twentyfold more ESG content, same score.

Cross-check on the real data: scaling each of the 344 real count vectors by an arbitrary
per-document integer in [1, 50) changes `SUS_lagasio` by at most **1.4 × 10⁻¹⁷**. The same
routine reproduces `results.csv` `SUS_lagasio` to 5 × 10⁻⁷ (the CSV's rounding).

**Demonstration B — diluting with non-ESG filler** [V]

adidas FY2018, 22,576 lemmatised tokens, 695 vocabulary hits (density 3.0785/100).
Appending 48,000 non-ESG filler tokens drawn from the document's own frequent non-vocabulary
words:

| Variant | tokens | ESG hits | density /100 | SUS_lagasio | SUS_density | SUS_tfidf_length |
|---|---:|---:|---:|---:|---:|---:|
| original | 22,576 | 695 | 3.0785 | 0.0347782747 | 3.0785 | 3.6565 |
| +48,000 filler | 70,576 | 695 | 0.9848 | **0.0347782747** | 0.9848 | 1.1696 |

`SUS_lagasio` ratio padded/original = **exactly 1.0**. The document's ESG density falls to
one third; the score does not move by a single bit.

`[!]` **Discrepancy.** The briefed figure was "density drops from 3.08 to 0.88" at 48,000
filler words. 3.0785 is exact for this document, but 48,000 filler tokens gives **0.9848**,
not 0.88. Reaching 0.88 requires **56,401** filler tokens. Either quote 48,000 → 0.985 (a
3.13× dilution) or 56,400 → 0.880 (a 3.50× dilution). The qualitative claim is untouched;
only the pairing of the two numbers is wrong.

### 6.2 What Lagasio's SUS actually measures [V]

Because only the direction survives, the score tracks how *evenly spread* the mentions are
across the vocabulary, not how many there are:

| Pair | Pearson | Spearman |
|---|---:|---:|
| **SUS_lagasio × Breadth** | **0.9735** | 0.9715 |
| SUS_lagasio × SUS_density | 0.4096 | 0.4504 |
| SUS_lagasio × SUS_tfidf_length | 0.4536 | 0.4894 |
| SUS_tfidf_length × Breadth | 0.5059 | 0.5447 |
| SUS_density × Breadth | 0.4550 | 0.5017 |
| **SUS_density × SUS_tfidf_length** | **0.9925** | 0.9936 |

The original SUS is, to 0.97, a measure of topical breadth.

### 6.3 IDF inverts the weighting [V]

`idf_j = ln(345 / (1 + df_j)) + 1`, n = 344. Observed range **1.0000** (`sustainable`,
df = 344) to **4.8976** (`fair condition`, df = 6). Total vocabulary occurrences in the
corpus: **896,499**.

Concentration of occurrences in low-idf terms:

| Cut | # terms | idf range | share of all ESG occurrences |
|---|---:|---|---:|
| df > 250 | 53 | ≤1.3023 | 84.30% |
| **df > 280** | **43** | ≤1.1946 (= idf < 1.2) | **78.36%** |
| **df > 290** | **39** | 1.0000–1.1668, mean 1.0612 | **76.00%** |
| df > 300 | 35 | ≤1.1298 | 73.96% |
| **idf > 2** | **46** | 2.0056–4.8976 | **2.80%** |

`[!]` **Discrepancy.** The briefed pairing "terms in >290 of 344 documents … hold 78.4% of
all ESG occurrences" mixes two cuts. The 78.36% figure belongs to **df > 280 (43 terms,
idf < 1.2)**; the df > 290 cut (39 terms) holds **76.00%**. Both are usable; pick one and
state it consistently. The "46 rare terms with idf > 2 hold 2.8%" figure is **exact** (46
terms, 2.80%).

Top vocabulary terms by occurrence:

| Term | df | idf | occurrences | share |
|---|---:|---:|---:|---:|
| remuneration | 336 | 1.0235 | 60,513 | 6.75% |
| emission | 339 | 1.0146 | 53,942 | 6.02% |
| governance | 343 | 1.0029 | 46,711 | 5.21% |
| sustainability | 343 | 1.0029 | 46,136 | 5.15% |
| climate | 341 | 1.0087 | 45,333 | 5.06% |
| sustainable | 344 | 1.0000 | 35,095 | 3.91% |
| water | 322 | 1.0659 | 27,007 | 3.01% |
| risk management | 342 | 1.0058 | 26,492 | 2.96% |
| carbon | 338 | 1.0175 | 24,239 | 2.70% |
| human right | 331 | 1.0384 | 21,274 | 2.37% |

Top-20 terms by occurrence hold 62.21%; top-40 hold 81.42%.

**The exchange rate.** `fair condition` occurs **8 times in the entire 26-million-word
corpus** (6 documents) and carries idf 4.8976. `climate` occurs 45,333 times and carries idf
1.0087.

`[!]` **Discrepancy (minor).** The ratio is **4.855**, i.e. one mention of *fair condition*
weighs as much as **≈4.86** mentions of *climate* — not 4.7. Against `sustainable`
(idf = 1.0000) the ratio is 4.90.

**Empirically second-order.** `SUS_tfidf_length` (idf weights, length-normalised) correlates
**0.9925** with `SUS_density` (uniform weights, length-normalised). The rare-term
up-weighting is theoretically wrong but changes almost nothing, because the terms it
up-weights carry 2.8% of the mass. **What matters is the normalisation, not the weighting.**

### 6.4 Convergent validity — the argument for `density` [V]

`QUANT` is built by entirely independent machinery: regexes over **raw** text, counting
digits, units and framework acronyms; it shares no vectoriser, no vocabulary weighting and
no token base with SUS. A substance measure should agree with it.

| SUS specification | Pearson with QUANT | Spearman |
|---|---:|---:|
| `tfidf_length` | **0.6795** | 0.7174 |
| `density` | **0.6627** | 0.7012 |
| `lagasio` | 0.3236 | 0.4329 |
| sublinear-TF (idf, length-normalised) | 0.0526 | — |
| sublinear-TF (no idf, length-normalised) | 0.0026 | — |

`[!]` **Discrepancy.** The briefed value "0.001 for a sublinear-TF variant" was not
reproduced exactly. The closest specification is sublinear TF **without** idf, length-
normalised: **+0.0026**. With idf: **+0.0526**. Both are statistically indistinguishable
from zero and both are an order of magnitude below `lagasio` and two below `density`, so the
argument stands; quote 0.003 (or "≈0") rather than 0.001.

**Additional finding not in the brief, worth reporting because it sharpens the claim:** the
sublinear variant fails only under *length normalisation*. Aggregating sublinear TF as a mean
over the vocabulary instead gives Pearson **0.529** with QUANT (0.482 with L2 norm) and
*preserves* the downward temporal trend. So the collapse is an interaction between
log-damping and the length denominator, not a property of log-damping alone.

**Why sublinear TF fails structurally** [V]. With a curated 154-term vocabulary that
documents cover almost entirely (mean **87.63** distinct terms present, median 89, max 133),
`Σ_j (1 + log tf_dj)` is dominated by the count of *distinct* terms present, which is bounded
by 154, while the denominator `L_d` keeps growing:

- corr( Σ(1 + log tf), number of distinct vocabulary terms present ) = **0.973**
- with idf weighting: **0.9674**

The numerator saturates; the denominator does not. The result is a measure that decreases
with document length regardless of content.

**Temporal reversal** [V]. Mean ESGSI by year under the sublinear specifications:

| Spec | 2018 | 2024 | trend r (year vs ESGSI) |
|---|---:|---:|---:|
| sublinear, no idf, length-norm | −0.183 | +0.234 | +0.098 |
| sublinear, idf, length-norm | −0.097 | +0.158 | +0.051 |
| sublinear, idf, mean over vocab | +0.806 | −0.890 | −0.389 |
| **density (adopted)** | **+0.562** | **−0.675** | **−0.299** |

### 6.5 Three defects repaired in commit `44114d0` (relevant to the "silent zero" argument)

Documented in the commit message and verifiable in the current code:
1. **QUANT was computed on the preprocessed text**, which drops numeric tokens →
   `percentages` and `large_numbers` returned 0 for all 344 documents. After the fix these
   two groups produce 109,974 and 13,518 matches [V — both figures reproduced exactly in §5.5].
   `large_numbers` additionally stopped counting years.
2. **`TextProcessor` filtered on `token.is_alpha`**, deleting `co2`, `co2e`, `sf6`,
   `iso14001` from the corpus entirely.
3. **HEDGE crossed an inflected L&M dictionary against a lemmatised corpus** — 46% coverage.
4. **`TfidfVectorizer` used the default `ngram_range=(1,1)`**, so every multi-word vocabulary
   entry (69 of 154, i.e. 45%) scored zero without raising anything.

All four are failures that produce a plausible-looking number rather than an error. This is
directly usable as a methods-section argument for why measurement instruments need
convergent-validity checks.

---

## 7. Robustness across the three SUS specifications [V]

### 7.1 Component correlations

See §6.2 table. Headline: `density` × `tfidf_length` = **0.9925**;
`density` × `lagasio` = **0.4096**.

### 7.2 Index correlations

| Pair | Pearson | Spearman |
|---|---:|---:|
| ESGSI_density × ESGSI_tfidf_length | 0.9960 | 0.9954 |
| **ESGSI_density × ESGSI_lagasio** | **0.6927** | 0.6715 |
| ESGSI_tfidf_length × ESGSI_lagasio | 0.7162 | 0.6880 |
| ESGSI × ESGSI_ext | 0.9089 | — |

### 7.3 Classification agreement (threshold ESGSI > 0)

| Comparison | Documents changing label | % |
|---|---:|---:|
| density vs lagasio | **97 / 344** | **28.2%** |
| tfidf_length vs lagasio | 94 / 344 | 27.3% |
| density vs tfidf_length | 7 / 344 | 2.0% |
| ESGSI vs ESGSI_ext | 58 / 344 | 16.9% |

Documents flagged (ESGSI > 0): density 171, tfidf_length 168, lagasio 160.

**The result is robust to term weighting (2.0% of labels move) and not robust to
normalisation (28.2% move).** This is the paper's central empirical claim about the index.

---

## 8. Temporal results [V]

### 8.1 Yearly means

| Year | ESGSI (density) | ESGSI tfidf_length | ESGSI lagasio | ESGSI_ext | SUS_density | SUS_tfidf_length | SUS_lagasio | Breadth | SEN | QUANT | HEDGE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2018 | **+0.5623** | +0.6126 | +0.5500 | +0.8074 | 4.8739 | 5.3962 | 0.0282 | 28.99 | +0.0095 | 0.00530 | 0.0885 |
| 2019 | +0.4652 | +0.5206 | +0.4937 | +0.6341 | 5.0365 | 5.5807 | 0.0285 | 30.14 | +0.0077 | 0.00570 | 0.0883 |
| 2020 | +0.1608 | +0.2130 | +0.0964 | +0.3326 | 5.4131 | 6.0441 | 0.0304 | 33.02 | −0.0077 | 0.00570 | 0.0882 |
| 2021 | +0.0061 | +0.0363 | −0.1760 | +0.0988 | 5.7893 | 6.5507 | 0.0325 | 36.70 | −0.0022 | 0.00640 | 0.0885 |
| 2022 | −0.1218 | −0.1250 | −0.2815 | −0.1880 | 6.1258 | 7.0363 | 0.0335 | 38.76 | +0.0042 | 0.00710 | 0.0875 |
| 2023 | −0.4073 | −0.4481 | −0.3782 | −0.7165 | 6.6493 | 7.7581 | 0.0341 | 40.24 | +0.0021 | 0.00840 | 0.0864 |
| 2024 | **−0.6747** | −0.8200 | −0.3143 | −0.9814 | 7.4483 | 8.9695 | 0.0346 | 41.73 | +0.0224 | 0.00810 | 0.0858 |

`[!]` **Discrepancy.** "Falls monotonically … robust across all three specifications" is true
for `density` (strictly monotone decreasing, 7/7) and for `tfidf_length` (strictly monotone),
but **not** for `lagasio`: 2024 (−0.3143) rebounds above 2023 (−0.3782). The lagasio series
is monotone for 2018–2023 only. The *direction and significance* hold for all three (§8.3);
the word "monotonically" must be attached to `density`/`tfidf_length`, not to all three.

### 8.2 Drivers [V]

| Quantity | 2018 | 2024 | Change |
|---|---:|---:|---:|
| Extracted ESG text per report (lemmatised tokens) | **27,905.5** | **63,654.4** | **+128.1%** |
| Extracted ESG text per report (raw tokens) | 51,609.2 | 115,059.1 | +122.9% |
| ESG mentions per report | 1,382.7 | 4,832.0 | +249.5% |
| ESG density (SUS_density, per 100 words) | **4.87** | **7.45** | **+52.8%** |
| Effective distinct terms (Breadth) | **28.99** | **41.73** | **+43.9%** |
| Distinct vocabulary terms present (raw count, /154) | 68.65 | 105.45 | +53.6% |
| Zones per report | 178.5 | 275.2 | +54.2% |

**SEN carries none of the trend**: slope +0.0014/yr, p = 0.712, r = +0.020. The entire ESGSI
decline is the SUS side of the difference — i.e. the finding is "substance grew", not
"tone changed".

### 8.3 Trend regressions (OLS of the metric on calendar year, n = 344) [V]

| Metric | slope/yr | p | Pearson r | Spearman ρ | p (ρ) |
|---|---:|---:|---:|---:|---:|
| ESGSI (density) | −0.2050 | 1.63e−08 | −0.299 | −0.344 | 5.8e−11 |
| ESGSI (tfidf_length) | −0.2348 | 7.90e−11 | — | −0.381 | 2.4e−13 |
| ESGSI (lagasio) | −0.1686 | 6.09e−06 | — | −0.287 | 6.0e−08 |
| ESGSI_ext | −0.3067 | 1.36e−10 | −0.337 | −0.410 | 2.2e−15 |
| SUS_density | +0.4165 | 6.50e−17 | +0.430 | +0.497 | 6.7e−23 |
| SUS_tfidf_length | +0.5739 | 3.88e−22 | +0.490 | +0.542 | 1.1e−27 |
| SUS_lagasio | +0.0012 | 8.68e−12 | +0.357 | +0.345 | 5.1e−11 |
| Breadth | +2.2925 | 4.95e−16 | +0.419 | +0.410 | 2.2e−15 |
| QUANT | +0.0005 | 2.83e−09 | +0.313 | +0.394 | 3.3e−14 |
| HEDGE | −0.0005 | 8.04e−02 | −0.094 | −0.082 | 0.129 |
| SEN | +0.0014 | 0.712 | +0.020 | +0.024 | 0.660 |

### 8.4 The trend is not a composition artefact [V]

Three independent arguments:

1. **The panel is balanced** — identical country counts every year; document types 33–36
   annual reports / 12–15 URD / 1 sustainability report per year (§1.3).
2. **Within a single country with a single document type**: Germany, 98 documents, 14
   companies × 7 years, 98/98 annual reports → ESGSI slope **−0.3015/yr, p = 1.19e−06**
   (steeper than the pooled slope).
3. **Within France-URD only** (99 documents): slope **−0.1575/yr, p = 2.67e−03**.
   **Annual reports only** (236 documents): slope **−0.2337/yr, p = 3.99e−07**.

Germany-only yearly means (clean single-type, single-country series):

| Year | ESGSI | SUS_density | SEN | Breadth |
|---|---:|---:|---:|---:|
| 2018 | +1.6912 | 3.6363 | 0.0782 | 24.06 |
| 2019 | +1.7119 | 3.7567 | 0.0898 | 24.05 |
| 2020 | +1.3199 | 4.4412 | 0.0844 | 27.86 |
| 2021 | +0.9751 | 4.7683 | 0.0598 | 31.92 |
| 2022 | +0.7727 | 5.1442 | 0.0586 | 34.16 |
| 2023 | +0.3533 | 5.9804 | 0.0603 | 35.22 |
| 2024 | −0.0343 | 6.8743 | 0.0706 | 37.22 |

### 8.5 Cross-sectional means (context, not a finding) [V]

By country:

| Country | n | ESGSI | ESGSI_ext | SUS_density | SEN |
|---|---:|---:|---:|---:|---:|
| ALEMANIA | 98 | +0.9700 | +1.2029 | 4.9430 | +0.0717 |
| FINLANDIA | 14 | −0.0352 | +0.4281 | 4.9817 | −0.0664 |
| BELGICA | 7 | −0.2899 | −0.8182 | 5.7894 | −0.0437 |
| PAISES BAJOS | 43 | −0.3086 | −0.2000 | 6.3646 | −0.0047 |
| ESPAÑA | 28 | −0.3166 | −0.5094 | 7.6095 | +0.0843 |
| FRANCIA | 119 | −0.3185 | −0.5218 | 6.1813 | −0.0193 |
| ITALIA | 35 | −0.9284 | −0.9484 | 6.1001 | −0.1107 |

By document type:

| Type | n | ESGSI (mean) | sd | SUS_density | SUS_lagasio | Breadth | SEN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Annual report | 236 | +0.157 | 1.460 | 5.7340 | 0.0296 | 32.00 | +0.0150 |
| URD | 101 | −0.224 | 1.025 | 6.1512 | 0.0363 | 43.52 | −0.0083 |
| Informe de sostenibilidad | 7 | −2.070 | 0.587 | 8.0014 | 0.0372 | 44.72 | −0.1331 |

Because country and type are confounded (§1.3), neither of these tables identifies a
country effect or a type effect. Report them descriptively only.

---

## 9. Limitations

### 9.1 The threshold is arbitrary by construction [V]

`ESGSI = z(SEN) − z(SUS)` is a difference of two mean-zero variables, so it has mean 0 by
construction and the `> 0` rule flags essentially half the corpus (172/344 = 50.0%; 165/344 =
48.0% for the extended index). The index is a **relative ranking within this corpus**, not an
absolute classification. The "Potential ESG-washing" label has no external calibration.

Additionally, both idf-based specifications estimate `df_j` on this same 344-document sample,
so `SUS_lagasio` and `SUS_tfidf_length` are corpus-dependent. `SUS_density` is the only
specification whose value is transferable to another corpus.

### 9.2 Country / document-type confound [V]

URD 99/101 French; Germany 98/98 annual reports; all 7 sustainability reports French. The
country and type factors are not separately identified. Germany simultaneously has the
highest ESGSI (+0.97) and the lowest SUS_density (4.94) and is 100% annual reports; the
sustainability reports have the lowest ESGSI (−2.07) and highest SUS_density (8.00) and are
100% French. Any country or type comparison in this corpus is uninterpretable.

### 9.3 Extended-index weights are arbitrary [V]

`w_quant = w_hedge = 0.5`, hard-coded in `config.ESGSI_EXT_WEIGHTS`, with no sensitivity
analysis anywhere in the repository. The extended index moves 58/344 labels (16.9%) relative
to the base index and correlates 0.909 with it.

### 9.4 `[!]` Double counting between SUS and QUANT — larger than briefed [V]

The brief states 8 double-counted terms (`gri, tcfd, sasb, issb, sdg, sfdr, csrd, taxonomy`).
All 8 are indeed in the SUS vocabulary and all 8 are full matches of the `frameworks` regex —
but the actual overlap is larger:

**Matched by `frameworks` (11 terms):** `csrd, gri, issb, sasb, sdg, sfdr, taxonomy,
taxonomy align, taxonomy alignment, taxonomy eligible, tcfd`
(the `taxonomy` family adds 3 beyond the briefed list).

**Matched by `units` (3 more terms):** `co2, co2e, ghg`.

**Total 14 SUS vocabulary entries also counted by QUANT**, holding **67,197 of 896,499 =
7.50%** of all SUS occurrences:

| Term | occurrences | % of SUS mass |
|---|---:|---:|
| taxonomy | 16,662 | 1.859% |
| co2 | 13,376 | 1.492% |
| ghg | 10,934 | 1.220% |
| gri | 6,754 | 0.753% |
| taxonomy align | 4,778 | 0.533% |
| taxonomy eligible | 3,973 | 0.443% |
| co2e | 3,390 | 0.378% |
| sdg | 2,240 | 0.250% |
| tcfd | 1,713 | 0.191% |
| csrd | 1,446 | 0.161% |
| sasb | 811 | 0.090% |
| sfdr | 788 | 0.088% |
| taxonomy alignment | 249 | 0.028% |
| issb | 83 | 0.009% |

A 15th, conceptual overlap: the SUS vocabulary contains `global compact` while the frameworks
regex contains `un global compact` / `ungc` — not a string full-match, but the same construct.
`paris agreement` is in the frameworks regex but was removed from the SUS vocabulary (it
lemmatises to `agreement`), so that one is *not* double-counted.

`frameworks` = **17.22%** of all QUANT hits [V, matches the briefed 17.2%], and that share
rises from 10.15% (2018) to 21.87% (2024) — so the double-counted component grows over the
study window and pushes SUS and QUANT in the same direction more strongly in later years.
Since ESGSI_ext subtracts z(QUANT) and subtracts z(SUS), the double counting inflates the
temporal slope of the extended index rather than cancelling.

### 9.5 Lexical extraction bounds what can be measured [V]

The extraction stage and the measurement stage share the same vocabulary. A term the
vocabulary does not know is absent from the extracted zones *and* from the SUS count — the
error is not random, it is a systematic floor. The external audit found **147 unrecognised
ESG terms with 2,970 occurrences** inside the already-extracted zones alone (top: `compliance`
436, `social` 275, `supply chain` 241, `raw material` 136), and those 147 terms are
deliberately not incorporated. The audit's own limitation applies too: matching is lexical, so
paraphrase and unlisted synonyms are invisible.

Additionally: mean extraction yield is 35.5% of PDF text, so ~64.5% of each report is
discarded before any measurement occurs. Zones are defined by a 1.0 kw/100-words threshold
and a ±1-paragraph window, both of which are unjustified constants (no sensitivity analysis
in the repo).

### 9.6 Component-specific measurement issues [V]

- **Different token bases.** SUS/HEDGE/Breadth are densities over 14.4M lemmatised tokens;
  QUANT is a density over 26.2M raw tokens. Each component is internally length-normalised,
  so the z-scores are comparable, but no component shares a denominator with any other.
- **`large_numbers` is noisy.** `\b(?!(?:19|20)\d{2}\b)\d{4,}\b` excludes years but still
  counts page numbers, note references, ISINs, share counts and monetary figures with no ESG
  content. It is 6.92% of QUANT.
- **SEN is doubly stemmed.** `pysentiment2.LM` Porter-stems its own dictionary and its input;
  the input is already spaCy-lemmatised. The LM lists are also strongly asymmetric (140
  positive vs 893 negative stems), and `Polarity` is a bounded ratio, not a density — a
  document with 4 positive and 2 negative hits scores the same as one with 400 and 200.
- **Lemmatisation artefacts in the vocabulary** (§4.1). `esr` is the worst case: an artefact
  of `esrs`, but with idf 2.53 and 6,282 occurrences it is one of the most heavily weighted
  entries under both idf specifications.
- **`QUANT` min = 0** for at least one document, so `z(QUANT)` has a hard floor there.
- **Duplicate document** (§1.2) enters every mean, sd and correlation twice.

### 9.7 LDA is implemented but not part of the results [V]

`src/topic_modeler.py` implements C_v / U_mass / C_npmi coherence and an aligned-Jaccard
seed-stability analysis, driven by `K_TOPICS_LIST = [8, 12, 16, 20, 24]`, `ALPHA_LIST =
['auto']`, `K_ITERS = 5`, `LDA_N_SEEDS = 0` (stability disabled). But:

- `main.py` runs with `run_lda = False`.
- `results/lda/coherence.csv` is **stale**: 4 rows, columns
  `Num. topics;Alpha;Iter;Coherence` (a single coherence value), k ∈ {5, 10}, values
  0.2888 / 0.3795 / 0.4007 / 0.3799. It predates the current code, which writes
  `Num_topics;Alpha;Iter;C_v;U_mass;C_npmi` over k ∈ {8…24}.

Either rerun the LDA before citing it or omit topic modelling from the paper entirely.

---

## 10. Quick-reference: what `results.csv` contains

`results/metrics/results.csv`, `;`-separated, 344 rows, 17 columns:

`Documento; País; Compañía; Año; TipoDocumento; SUS_Score; SUS_density; SUS_tfidf_length;
SUS_lagasio; Breadth; SEN_Score; QUANT_Score; HEDGE_Score; ESGSI; ESGSI_ext; Etiqueta;
Etiqueta_ext`

- `SUS_Score` = the specification selected by `config.SUS_MODE` (currently `density`, verified
  identical to `SUS_density`); it is the one that feeds `ESGSI` and `ESGSI_ext`.
- SUS columns are rounded to 6 dp; `Breadth`, `SEN`, `QUANT`, `HEDGE`, `ESGSI`, `ESGSI_ext`
  to 4 dp. Rounding of `QUANT`/`HEDGE` (which have sd 0.0035 / 0.0096) is why recomputing
  `ESGSI_ext` from the published columns reproduces it only to ±0.009; recomputing from the
  unrounded pipeline values is exact.
