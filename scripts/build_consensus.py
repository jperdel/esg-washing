"""
build_consensus.py
------------------
Agrega las tres revisiones ciegas del vocabulario y genera el informe de consenso.

    python scripts/build_consensus.py

Entrada
    paper/_research/revision_{1,2,3}.csv   listas independientes, formato
                                           termino;pilar;sectorial;colocacion;justificacion
    paper/_research/muestra_revision.txt   la muestra que los tres leyeron
    metadata/ESG_terms_final.csv           vocabulario vigente (361 entradas)

Salida
    informes/ESG_deep_consensus.md         informe legible
    metadata/ESG_deep_consensus.csv        misma tabla en formato maquina

Los tres revisores trabajaron a ciegas: sin ver el vocabulario del proyecto, sin
ver los ficheros de los otros dos y sin conocer sus resultados. El consenso es
por tanto acuerdo independiente, no convergencia inducida.

DOS DETALLES QUE NO SON OBVIOS Y QUE YA HAN CAUSADO ERRORES:

  1. 33 de las 361 entradas del vocabulario son EXPRESIONES REGULARES
     (tipo == "patron"): `environ\\w+`, `paris\\s+agreement`, `water\\s+(?:...)`.
     Hay que EJECUTARLAS con re.search, no compararlas como texto. Tratarlas
     como literales las cuenta a todas como ausentes e infla los huecos: en la
     primera pasada dio 72 huecos 3/3 en vez de 63, y presentaba toda la familia
     `environmental + X` como descubierta cuando `environ\\w+` ya la captura.

  2. La comparacion es ASIMETRICA. La muestra son 17 informes de 343, asi que
     sirve para PROPONER terminos nuevos pero no para RETIRAR los existentes:
     que una entrada no aparezca puede significar solo que no sale en estos 17
     documentos. Por eso el informe separa "ausente de la muestra" de "presente
     y aun asi nadie la anoto".
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
METADATA_DIR = BASE_DIR / "metadata"
RESEARCH_DIR = BASE_DIR / "paper" / "_research"

# --- normalizacion de terminos -------------------------------------------
# Objetivo: que "GHG Emissions", "ghg emissions" y "ghg emission" cuenten como
# el mismo termino, sin llegar a fusionar terminos que de verdad son distintos.
SYNONYMS = {
    "labour": "labor", "labours": "labor",
    "programme": "program", "programmes": "program",
    "organisation": "organization", "organisational": "organizational",
    "decarbonisation": "decarbonization", "utilisation": "utilization",
    "sulphur": "sulfur", "fibre": "fiber", "centre": "center",
    "behaviour": "behavior", "favour": "favor",
    "ghgs": "ghg", "co2eq": "co2e", "co2-eq": "co2e",
    "wellbeing": "well-being",
}

# Palabras cuya singularizacion mecanica seria erronea.
IRREGULAR = {
    "people": "person", "children": "child", "women": "woman", "men": "man",
    "rights": "right", "goods": "good", "hours": "hour", "goals": "goal",
    "emissions": "emission",
    # invariantes: terminan en -s pero ya son singulares
    "series": "series", "gas": "gas", "analysis": "analysis",
    "species": "species", "process": "process", "business": "business",
    "waste": "waste", "access": "access", "stress": "stress", "loss": "loss",
    "class": "class", "bonus": "bonus", "campus": "campus", "status": "status",
    "focus": "focus", "ethics": "ethics", "politics": "politics",
    "works": "works", "human": "human",
}


def singularize(word: str) -> str:
    if word in IRREGULAR:
        return IRREGULAR[word]
    if len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("ses", "xes", "zes", "ches", "shes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def normalize(term: str) -> str:
    """Clave de agrupacion: minusculas, sin puntuacion, sin plurales."""
    term = term.strip().lower().replace("’", "").replace("–", "-")
    term = re.sub(r"[-_/]+", " ", term)
    term = re.sub(r"[^a-z0-9 ]+", " ", term)
    term = re.sub(r"\s+", " ", term).strip()
    return " ".join(singularize(SYNONYMS.get(w, w)) for w in term.split())


# --- carga -----------------------------------------------------------------
def load_reviews() -> dict[str, dict]:
    """Cruza las tres listas en un diccionario por termino normalizado."""
    agg: dict[str, dict] = {}
    for i in (1, 2, 3):
        path = RESEARCH_DIR / f"revision_{i}.csv"
        seen: set[str] = set()
        with open(path, encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh, delimiter=";"):
                raw = (row.get("termino") or "").strip()
                key = normalize(raw)
                if not key or key in seen:
                    continue          # un revisor no vota dos veces lo mismo
                seen.add(key)
                rec = agg.setdefault(key, {"surface": {}, "pilar": {},
                                           "sect": {}, "coloc": {}, "just": {}})
                rec["surface"][i] = raw
                rec["pilar"][i] = (row.get("pilar") or "").strip().upper()
                rec["sect"][i] = (row.get("sectorial") or "0").strip()
                rec["coloc"][i] = (row.get("colocacion") or "0").strip()
                rec["just"][i] = (row.get("justificacion") or "").strip()
        print(f"revision_{i}.csv: {len(seen)} terminos unicos")
    return agg


def load_vocabulary() -> tuple[list[dict], set[str], list[tuple]]:
    """Devuelve (filas, literales normalizados, patrones compilados)."""
    rows = list(csv.DictReader(
        open(METADATA_DIR / "ESG_terms_final.csv", encoding="utf-8-sig"),
        delimiter=";"))
    literals: set[str] = set()
    patterns: list[tuple] = []
    for row in rows:
        term = row["termino"].strip()
        if row["tipo"].strip() == "patron":
            try:
                patterns.append((re.compile(term, re.I), term))
            except re.error as exc:
                print(f"  AVISO: regex invalida en el vocabulario: {term} ({exc})")
        else:
            literals.add(normalize(term))
    print(f"ESG_terms_final.csv: {len(rows)} entradas "
          f"({len(literals)} literales, {len(patterns)} patrones)")
    return rows, literals, patterns


# --- cobertura -------------------------------------------------------------
def classify(key: str, surface: str, literals: set[str],
             patterns: list[tuple]) -> tuple[str, str]:
    """(estado, entrada_que_lo_cubre) para un termino revisado."""
    if key in literals:
        return "en vocabulario", ""
    words = key.split()
    best = None
    for n in range(len(words), 0, -1):          # la raiz mas larga gana
        for i in range(len(words) - n + 1):
            cand = " ".join(words[i:i + n])
            if cand in literals and (best is None or len(cand) > len(best)):
                best = cand
    if best:
        return "cubierto", best
    for regex, source in patterns:              # los patrones se EJECUTAN
        if regex.search(surface) or regex.search(key):
            return "cubierto", source
    return "HUECO", ""


def classify_inverse(rows: list[dict], reviewed: set[str]) -> dict[str, list]:
    """Cada entrada del vocabulario frente a lo que encontraron los revisores."""
    sample = (RESEARCH_DIR / "muestra_revision.txt").read_text(
        encoding="utf-8", errors="ignore").lower()
    sample = re.sub(r"\s+", " ", re.sub(r"[-_/]+", " ", sample))
    words_of = {k: set(k.split()) for k in reviewed}

    out = {"exacto": [], "absorbido": [], "huerfano_pres": [], "huerfano_aus": []}
    for row in rows:
        term, kind = row["termino"].strip(), row["tipo"].strip()
        if kind == "patron":
            try:
                regex = re.compile(term, re.I)
            except re.error:
                continue
            hits = [k for k in reviewed if regex.search(k)]
            present = bool(regex.search(sample))
        else:
            key = normalize(term)
            if key in reviewed:
                out["exacto"].append(term)
                continue
            wanted = set(key.split())
            hits = [k for k in reviewed if wanted and wanted.issubset(words_of[k])]
            pat = r"\b" + r"\s+".join(re.escape(w) + r"s?" for w in key.split()) + r"\b"
            present = bool(re.search(pat, sample))
        if hits:
            out["absorbido"].append((term, sorted(hits)[:4]))
        elif present:
            out["huerfano_pres"].append((term, row.get("pilar", "?"),
                                         row.get("origen", "")))
        else:
            out["huerfano_aus"].append(term)
    for k in out:
        out[k] = sorted(out[k])
    return out


def main() -> None:
    agg = load_reviews()
    rows, literals, patterns = load_vocabulary()

    records = []
    for key, val in agg.items():
        revs = sorted(val["pilar"])
        pillars = Counter(val["pilar"][i] for i in revs)
        sect_yes = sum(1 for i in revs if val["sect"][i] == "1")
        col_yes = sum(1 for i in revs if val["coloc"][i] == "1")
        forms = Counter(val["surface"][i] for i in revs)
        top = forms.most_common(1)[0][1]
        estado, cubierto = classify(
            key, val["surface"][revs[0]], literals, patterns)
        records.append({
            "termino": sorted([f for f in forms if forms[f] == top], key=len)[0],
            "clave": key,
            "pilar": pillars.most_common(1)[0][0],
            "pilar_disenso": "/".join(sorted(pillars)) if len(pillars) > 1 else "",
            "sectorial": int(sect_yes * 2 > len(revs)),
            "sectorial_disenso": int(0 < sect_yes < len(revs)),
            "colocacion": int(col_yes * 2 > len(revs)),
            "consenso": {3: "maximo", 2: "alto", 1: "bajo"}[len(revs)],
            "n_revisores": len(revs),
            "revisores": "+".join(str(i) for i in revs),
            "estado_vocabulario": estado,
            "cubierto_por": cubierto,
            "justificacion": val["just"][revs[0]],
        })
    records.sort(key=lambda r: (-r["n_revisores"], r["pilar"], r["clave"]))

    out_csv = METADATA_DIR / "ESG_deep_consensus.csv"
    with open(out_csv, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0]), delimiter=";")
        writer.writeheader()
        writer.writerows(records)

    inverse = classify_inverse(rows, set(agg))
    total = sum(len(v) for v in inverse.values())
    assert total == len(rows), f"el inverso no cuadra: {total} != {len(rows)}"

    counts = Counter(r["n_revisores"] for r in records)
    gaps = Counter(r["n_revisores"] for r in records
                   if r["estado_vocabulario"] == "HUECO")
    print(f"\nEscrito: {out_csv}  ({len(records)} terminos)")
    for lvl, name in ((3, "maximo"), (2, "alto"), (1, "bajo")):
        print(f"  consenso {name:7s} ({lvl}/3): {counts[lvl]:3d} terminos, "
              f"{gaps[lvl]:3d} huecos")
    print(f"\nInverso sobre las {len(rows)} entradas del vocabulario:")
    for k, label in (("exacto", "registradas tal cual"),
                     ("absorbido", "dentro de un compuesto"),
                     ("huerfano_aus", "ausentes de la muestra"),
                     ("huerfano_pres", "presentes y no anotadas")):
        print(f"  {label:26s}: {len(inverse[k]):3d}")

    json.dump(inverse, open(RESEARCH_DIR / "consensus_inverse.json", "w",
                            encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\nEl informe legible (ESG_deep_consensus.md) se redacta a mano sobre")
    print("estas cifras; este script regenera el CSV y las verifica.")


if __name__ == "__main__":
    sys.exit(main())
