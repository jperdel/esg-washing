"""
export_vocab_v3.py
------------------
Vuelca la propuesta de vocabulario v3 (metadata/ESG_terms_v3.csv) a los
ficheros que lee el pipeline:

    metadata/esg_terms.txt                 terminos, no sectoriales
    metadata/esg_terms_sectorial.txt       terminos sectoriales
    metadata/esg_patterns.txt              patrones regex, no sectoriales
    metadata/esg_patterns_sectorial.txt    patrones regex sectoriales

    python scripts/export_vocab_v3.py
    python scripts/build_lexicons.py       # despues, siempre

El vocabulario anterior (154 + ampliacion a 289 + 22 sectoriales, con los
patrones escritos en lexical_document_filter.py) queda en el historial de git.

Los patrones solo actuan en la extraccion, sobre texto crudo. Los terminos
alimentan la extraccion y, lematizados, el SUS.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
METADATA_DIR = BASE_DIR / "metadata"
PILLARS = [("E", "Ambiental"), ("S", "Social"), ("G", "Gobernanza"),
           ("TRANS", "Transversal (marcos, sostenibilidad en general)")]

HEADER = """\
# GENERADO POR scripts/export_vocab_v3.py a partir de metadata/ESG_terms_v3.csv.
# Para cambiar el vocabulario, edita las decisiones (ESG_terms_v3_decisiones.csv),
# ejecuta build_vocab_v3.py y despues este script. No editar a mano.
#
# {what}
#
# >>> DESPUES DE REGENERAR, EJECUTA:  python scripts/build_lexicons.py
"""

WHAT = {
    ("termino", "0"): "Terminos en forma natural singular. El regex de extraccion tolera "
                      "plural y guiones; el SUS usa su forma lematizada.",
    ("termino", "1"): "Terminos sectoriales: entran con ESG_INCLUDE_SECTORAL=1.",
    ("patron", "0"): "Patrones regex, uno por linea. Solo actuan en la extraccion (texto crudo).",
    ("patron", "1"): "Patrones regex sectoriales: entran con ESG_INCLUDE_SECTORAL=1.",
}

TARGETS = {
    ("termino", "0"): "esg_terms.txt",
    ("termino", "1"): "esg_terms_sectorial.txt",
    ("patron", "0"): "esg_patterns.txt",
    ("patron", "1"): "esg_patterns_sectorial.txt",
}


def main() -> None:
    with open(METADATA_DIR / "ESG_terms_v3.csv", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh, delimiter=";"))

    for (kind, sec), target in TARGETS.items():
        subset = [r for r in rows if r["tipo"] == kind and r["sectorial"] == sec]
        lines = [HEADER.format(what=WHAT[(kind, sec)])]
        for code, name in PILLARS:
            block = sorted({r["termino"].strip() for r in subset if r["pilar"] == code},
                           key=str.lower)
            if not block:
                continue
            lines.append(f"\n# --- {code}: {name} ({len(block)}) ---")
            for term in block:
                if kind == "patron":
                    re.compile(term)
                lines.append(term if kind == "patron" else term.lower())
        unknown = [r["termino"] for r in subset if r["pilar"] not in dict(PILLARS)]
        if unknown:
            raise ValueError(f"pilar desconocido en {unknown}")
        (METADATA_DIR / target).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"{target:30s} {len(subset):4d} entradas")


if __name__ == "__main__":
    main()
