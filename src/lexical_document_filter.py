"""
LexicalDocumentFilter — keyword-driven ESG text extraction.

Strategy: ESG-zone extraction — scans the full document paragraph-by-paragraph,
scores each paragraph by keyword density, then expands "hot" paragraphs into
coherent context windows and merges overlapping windows.

No chunking into fixed-size windows: the output units are natural text zones
whose length is determined by the document structure, not by an arbitrary
character budget.

Each zone carries enriched metadata: matched keyword list, word count, and
keyword density (keywords per 100 words).
"""

from __future__ import annotations

import re
import json
import fitz
from pathlib import Path
from loguru import logger

# ── ESG keyword vocabulary ────────────────────────────────────────────────────
# Single-word terms are loaded from metadata/esg_terms.txt (canonical lemma forms).
# _EXTRA_PATTERNS covers prefix matching, multi-word phrases, and numeric patterns
# that only make sense on raw text and have no direct TF-IDF representation.

_METADATA_DIR = Path(__file__).resolve().parent.parent / "metadata"

def _load_esg_terms() -> list[str]:
    path = _METADATA_DIR / "esg_terms.txt"
    with open(path, encoding="utf-8") as fh:
        return [
            line.strip().lower()
            for line in fh
            if line.strip() and not line.startswith("#")
        ]

# Patterns that require prefix or multi-word matching — raw PDF text only.
# 'social' and 'energy' only when accompanied by ESG context to avoid false positives.
_EXTRA_PATTERNS: list[str] = [
    r"sustainab\w+",                                               # sustainability, sustainably…
    r"decarboni\w+",                                               # decarbonize, decarbonization…
    r"recycl\w+",                                                  # recycling, recyclable…
    r"offset\w+",                                                  # offsetting, offsetted…
    r"co2e?",
    r"scope\s*[123]",
    r"scope\s+(?:one|two|three)",
    r"water\s+(?:usage|consumption|stewardship|management)",
    r"social\s+(?:responsibility|impact|policy|pillar|report|performance|value|welfare|audit|capital|sustainability|license)",
    r"energy\s+(?:consumption|efficiency|transition|mix)",
    r"human[\s\-]rights?",
    r"net[\s\-]zero",
    r"circular\s+economy",
    r"paris\s+agreement",
    r"health\s+and\s+safety",
    r"employee\s+well\w*",
    r"supply[\s\-]chain\s+(?:ethics|risk)",
    r"responsible\s+(?:sourcing|investment|business)",
    r"double\s+materiality",
    r"transition\s+plan",
    r"renewable\s+energy",
    r"wind\s+(?:power|energy)",
    r"tonne\w*\s+(?:co2|carbon)",
    r"science[\s\-]based\s+targets?",
    r"un\s+sdg",
]

_ESG_TERMS  = _load_esg_terms()
_terms_pat  = "|".join(f"{t}s?" for t in _ESG_TERMS)
_extra_pat  = "|".join(_EXTRA_PATTERNS)
_ESG_KW_RE  = re.compile(rf"\b(?:{_terms_pat}|{_extra_pat})\b", re.IGNORECASE)

# ── Running header / footer detection ───────────────────────────────────────
_HEADER_RATIO  = 0.08
_FOOTER_RATIO  = 0.92
_REPEAT_THRESH = 0.30
_MAX_HF_LEN    = 120

# ── Section marker ────────────────────────────────────────────────────────────
_SECTION_MARKER    = "§"
_SECTION_MAX_TOKENS = 10

# ── Zone extraction parameters ────────────────────────────────────────────────
_DEFAULT_KW_THRESHOLD  = 1.0   # keywords per 100 words to flag a paragraph
_DEFAULT_CONTEXT_PARAS = 1     # paragraphs of context around each hot paragraph
_DEFAULT_MIN_ZONE_LEN  = 150   # minimum chars for an extracted zone to be kept


class LexicalDocumentFilter:
    """
    Keyword-driven ESG text extractor.  No external API required.

        ldf = LexicalDocumentFilter()
        ldf.process_folder(pdf_data_dir)
    """

    def __init__(
        self,
        kw_threshold:  float = _DEFAULT_KW_THRESHOLD,
        context_paras: int   = _DEFAULT_CONTEXT_PARAS,
        min_zone_len:  int   = _DEFAULT_MIN_ZONE_LEN,
    ):
        self.kw_threshold  = kw_threshold
        self.context_paras = context_paras
        self.min_zone_len  = min_zone_len

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def process_folder(self, root_folder: Path):
        """Extract and save ESG zones for every PDF under root_folder."""
        if not root_folder.exists():
            raise FileNotFoundError(f"Carpeta no encontrada: {root_folder}")
        pdf_files = [p for p in root_folder.rglob("*") if p.suffix.lower() == ".pdf"]
        logger.info(f"LexicalFilter: {len(pdf_files)} PDFs en '{root_folder.name}'.")
        for idx, pdf in enumerate(pdf_files, 1):
            logger.info(f"[{idx}/{len(pdf_files)}] {pdf.name}")
            try:
                result = self.process_document(pdf)
                if result["zones"]:
                    self._save_results(pdf, result)
                else:
                    logger.warning(f"Sin zonas ESG: {pdf.name}")
            except Exception as e:
                logger.exception(f"Error en {pdf.name}: {e}")
        logger.success(f"LexicalFilter completo ({len(pdf_files)} PDFs).")

    def process_document(self, pdf_path: Path) -> dict:
        """
        Full pipeline for a single PDF:
          1. Extract clean paragraphs from the PDF.
          2. Apply keyword-zone extraction on all paragraphs.
          3. Return merged list of relevant text zones.
        """
        paragraphs = self._extract_paragraphs(pdf_path)
        zones = self._keyword_zone_extraction(paragraphs)
        zones.sort(key=lambda z: z["start_char"])

        relevant_text = self._merge_zones_text(zones)

        logger.info(f"{pdf_path.name}: {len(zones)} keyword zones.")
        return {
            "file":          pdf_path.name,
            "path":          str(pdf_path),
            "zones":         zones,
            "relevant_text": relevant_text,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # PDF text extraction
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_for_hf(text: str) -> str:
        return re.sub(r"\d+", "#", text.strip())

    def _detect_running_hf(self, doc: fitz.Document) -> set:
        n_pages = len(doc)
        counts: dict[str, int] = {}
        for page in doc:
            ph = page.rect.height
            for block in page.get_text("blocks"):
                y0, y1, text, btype = block[1], block[3], block[4], block[6]
                if btype != 0:
                    continue
                ts = text.strip()
                if not ts or len(ts) > _MAX_HF_LEN:
                    continue
                if y1 < ph * _HEADER_RATIO or y0 > ph * _FOOTER_RATIO:
                    key = self._normalize_for_hf(ts)
                    counts[key] = counts.get(key, 0) + 1
        return {k for k, v in counts.items() if v / n_pages > _REPEAT_THRESH}

    @staticmethod
    def _is_section_header(text: str) -> bool:
        alpha = re.sub(r"[^a-zA-Z]", "", text)
        if len(alpha) < 2:
            return False
        if alpha != alpha.upper():
            return False
        tokens = text.split()
        if len(tokens) > _SECTION_MAX_TOKENS:
            return False
        single_char_ratio = sum(
            1 for t in tokens if len(re.sub(r"[^a-zA-Z]", "", t)) <= 1
        ) / len(tokens)
        if single_char_ratio >= 0.60:
            return False
        if re.search(r'\b\w[\w-]*\.\w{2,6}\b', text):
            return False
        return True

    @staticmethod
    def _clean_block(text: str) -> str:
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        text = text.encode("ascii", "ignore").decode("ascii")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _is_financial_table(text: str) -> bool:
        if _ESG_KW_RE.search(text):
            return False
        total = len(text)
        if total == 0:
            return False
        digits = sum(1 for c in text if c.isdigit())
        lines_list = text.split("\n")
        n_lines    = len(lines_list)
        if (digits / total) > 0.35 and n_lines <= 6:
            return True
        if n_lines >= 2:
            short_lines = sum(1 for ln in lines_list if len(ln.strip()) < 60)
            first_line  = lines_list[0].rstrip()
            no_terminal = not first_line.endswith((".", "!", "?", ":"))
            if short_lines == n_lines and no_terminal:
                return True
        return False

    def _extract_paragraphs(self, pdf_path: Path) -> list[dict]:
        """Returns list of {text, start_char, end_char, section_idx, section}."""
        with fitz.open(pdf_path) as doc:
            running_hf = self._detect_running_hf(doc)
            units: list[str] = []

            for page in doc:
                ph = page.rect.height
                pw = page.rect.width
                raw_blocks = [b for b in page.get_text("blocks") if b[6] == 0]
                blocks = sorted(raw_blocks, key=lambda b: (b[1], b[0]))

                for block in blocks:
                    y0, y1, text, btype = block[1], block[3], block[4], block[6]
                    if y1 < ph * _HEADER_RATIO or y0 > ph * _FOOTER_RATIO:
                        continue
                    ts = text.strip()
                    if not ts:
                        continue
                    if self._normalize_for_hf(ts) in running_hf:
                        continue
                    if self._is_financial_table(ts):
                        continue
                    if self._is_section_header(ts):
                        header_clean = self._clean_block(ts)
                        if header_clean:
                            units.append(f"{_SECTION_MARKER} {header_clean} {_SECTION_MARKER}")
                        continue
                    cleaned = self._clean_block(ts)
                    if cleaned:
                        units.append(cleaned)

        # Build paragraph list with char offsets
        paragraphs: list[dict] = []
        char_pos = 0
        current_section_idx = -1
        current_section_title = ""

        for unit in units:
            is_marker = unit.startswith(_SECTION_MARKER) and unit.endswith(_SECTION_MARKER)
            unit_len  = len(unit)

            if is_marker:
                current_section_title = unit.strip(_SECTION_MARKER).strip()
                current_section_idx += 1
            else:
                paragraphs.append({
                    "text":        unit,
                    "start_char":  char_pos,
                    "end_char":    char_pos + unit_len,
                    "section_idx": current_section_idx,
                    "section":     current_section_title,
                })

            char_pos += unit_len + 2   # +2 for \n\n separator

        logger.debug(f"{pdf_path.name}: {len(paragraphs)} párrafos.")
        return paragraphs

    # ──────────────────────────────────────────────────────────────────────────
    # ESG-zone extraction (keyword density + context window)
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _kw_density(text: str) -> float:
        """Keywords per 100 words."""
        words = len(text.split())
        if words == 0:
            return 0.0
        hits = len(_ESG_KW_RE.findall(text))
        return hits / words * 100

    def _keyword_zone_extraction(self, paragraphs: list[dict]) -> list[dict]:
        """
        Score every paragraph, expand hot ones into context windows,
        merge overlapping windows, return zone dicts.
        """
        if not paragraphs:
            return []
        scores = [self._kw_density(p["text"]) for p in paragraphs]
        return self._build_zones(paragraphs, scores, source="keyword_zone")

    def _build_zones(
        self,
        paragraphs: list[dict],
        scores:     list[float],
        source:     str,
    ) -> list[dict]:
        n = len(paragraphs)
        ctx = self.context_paras

        # Mark hot paragraphs
        hot = [s >= self.kw_threshold for s in scores]

        # Expand into windows [max(0, i-ctx) .. min(n-1, i+ctx)]
        windows: list[tuple[int, int]] = []
        for i, is_hot in enumerate(hot):
            if not is_hot:
                continue
            lo = max(0, i - ctx)
            hi = min(n - 1, i + ctx)
            windows.append((lo, hi))

        if not windows:
            return []

        # Merge overlapping windows
        windows.sort()
        merged: list[tuple[int, int]] = [windows[0]]
        for lo, hi in windows[1:]:
            if lo <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
            else:
                merged.append((lo, hi))

        # Build zone dicts
        zones: list[dict] = []
        for lo, hi in merged:
            zone_paras = paragraphs[lo : hi + 1]
            zone_text  = "\n\n".join(p["text"] for p in zone_paras)
            if len(zone_text) < self.min_zone_len:
                continue
            kw_matches = _ESG_KW_RE.findall(zone_text)
            kw_count   = len(kw_matches)
            word_count = len(zone_text.split())
            kw_density = round(kw_count / word_count * 100, 4) if word_count else 0.0
            zones.append({
                "text":           zone_text,
                "start_char":     zone_paras[0]["start_char"],
                "end_char":       zone_paras[-1]["end_char"],
                "source":         source,
                "section":        zone_paras[0].get("section", ""),
                "kw_count":       kw_count,
                "keywords_found": sorted(set(m.lower() for m in kw_matches)),
                "word_count":     word_count,
                "kw_density":     kw_density,
            })
        return zones

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _merge_zones_text(zones: list[dict]) -> str:
        if not zones:
            return ""
        return "\n\n\n\n".join(z["text"] for z in zones)

    def _save_results(self, pdf_path: Path, result: dict):
        out_path = Path(str(pdf_path).replace("pdf", "chunks_lexical")).with_suffix(".json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "file":          result["file"],
            "path":          result["path"],
            "zones":         result["zones"],
            "relevant_text": result["relevant_text"],
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=4)
        logger.success(f"JSON guardado: {out_path}")