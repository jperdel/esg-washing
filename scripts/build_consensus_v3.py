"""
build_consensus_v3.py
---------------------
Agrega la auditoria de FALSOS POSITIVOS del vocabulario (ronda 3).

    python scripts/build_consensus_v3.py

Entrada
    paper/_research/kwic/bloque_NN.txt       concordancias auditadas
    paper/_research/kwic/veredicto_NN.csv    un veredicto por concordancia
    paper/_research/term_mass_sin_navegacion.csv   masa de cada entrada
    metadata/ESG_terms_final.csv             vocabulario vigente

Salida
    metadata/ESG_deep_consensus_v3.csv       precision por termino y decision propuesta

QUE MIDE

    Las rondas 1 y 2 midieron si un termino es ESG *como tipo*. Esta mide si,
    cuando el termino dispara sobre el corpus, el disparo es una mencion ESG
    real. Es la cifra que entra en las metricas.

TRES DECISIONES DE AGREGACION

  1. La precision GLOBAL se pondera por masa (muestreo estratificado con pesos
     conocidos):  P = sum_t m_t * p_t / sum_t m_t.  Sin ponderar, un termino de
     80.000 disparos pesaria lo mismo que uno de 12.

  2. El RUIDO que aporta cada termino es  m_t * (1 - p_t): disparos falsos
     esperados en el corpus. Ordenar por ahi dice que arreglar primero; ordenar
     por precision diria que terminos son peores, que no es lo mismo.

  3. Los intervalos son de Wilson al 95 %. Con 6 concordancias por termino son
     anchos, y la decision propuesta lo tiene en cuenta: un termino ligero no se
     retira por una sola muestra mala si su intervalo es compatible con una
     precision aceptable.

CORPUS AUDITADO

    Las concordancias se muestrearon al azar dentro de cada informe y sobre las
    zonas SIN parrafos de navegacion (indices, tablas de correspondencia), que
    el extractor ya descarta. Una version anterior de la muestra tomaba las
    primeras apariciones de cada informe y quedo invalidada; se conserva en
    paper/_research/kwic_v1_sesgada solo como material de ajuste del filtro.
"""

from __future__ import annotations

import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
METADATA_DIR = BASE_DIR / "metadata"
RESEARCH_DIR = BASE_DIR / "paper" / "_research"
# --candidatos: agrega la auditoria de los huecos de la ronda 2 en vez del
# vocabulario vigente.
import sys
# --colocaciones: agrega la auditoria de las colocaciones nuevas de la propuesta v3.
CANDIDATES = "--candidatos" in sys.argv
COLLOCATIONS = "--colocaciones" in sys.argv
_MODE = "colocaciones" if COLLOCATIONS else "candidatos" if CANDIDATES else None
KWIC_DIR = RESEARCH_DIR / (f"kwic_{_MODE}" if _MODE else "kwic")
MASS_FILE = RESEARCH_DIR / (f"term_mass_{_MODE}.csv" if _MODE
                            else "term_mass_sin_navegacion.csv")
OUT_FILE = METADATA_DIR / (f"ESG_deep_consensus_v3_{_MODE}.csv" if _MODE
                           else "ESG_deep_consensus_v3.csv")

CATEGORIES = ["ok", "indice", "financiero", "generico", "otro"]

# Umbrales de la decision propuesta.
KEEP_AT = 0.80        # precision puntual a partir de la cual se mantiene
DROP_BELOW = 0.50     # precision puntual por debajo de la cual se propone retirar
DROP_CI_MAX = 0.70    # ... siempre que el limite superior de Wilson no pase de aqui


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_verdicts() -> dict[str, list[dict]]:
    """termino -> lista de veredictos, comprobando que cada fichero casa con su
    bloque linea a linea."""
    by_term: dict[str, list[dict]] = defaultdict(list)
    for block in sorted(KWIC_DIR.glob("bloque_*.txt")):
        num = block.stem.split("_")[1]
        verdict = KWIC_DIR / f"veredicto_{num}.csv"
        if not verdict.exists():
            raise FileNotFoundError(f"falta {verdict.name}")
        expected, term = [], None
        for line in block.read_text(encoding="utf-8").split("\n"):
            if line.startswith("TERMINO: "):
                term = line[9:].strip()
            m = re.match(r"\[(\d+)\]", line)
            if m:
                expected.append((term, int(m.group(1))))
        rows = list(csv.DictReader(open(verdict, encoding="utf-8-sig"), delimiter=";"))
        got = [(r["termino"].strip(), int(r["n"])) for r in rows]
        if got != expected:
            raise ValueError(f"{verdict.name} no casa con {block.name}")
        for r in rows:
            cat = r["categoria"].strip()
            if cat not in CATEGORIES:
                raise ValueError(f"{verdict.name}: categoria desconocida {cat!r}")
            by_term[r["termino"].strip()].append(
                {"ok": cat == "ok", "cat": cat, "nota": (r.get("nota") or "").strip()})
    return by_term


def decide(p: float, lo: float, hi: float) -> str:
    if p >= KEEP_AT:
        return "mantener"
    if p < DROP_BELOW and hi <= DROP_CI_MAX:
        return "retirar"
    return "revisar"


def main() -> None:
    verdicts = load_verdicts()
    mass = {r["termino"]: r for r in csv.DictReader(
        open(MASS_FILE, encoding="utf-8"), delimiter=";")}
    vocab = mass if _MODE else {r["termino"]: r for r in csv.DictReader(
        open(METADATA_DIR / "ESG_terms_final.csv", encoding="utf-8-sig"), delimiter=";")}

    records = []
    for term, vs in verdicts.items():
        n = len(vs)
        k = sum(v["ok"] for v in vs)
        p = k / n
        lo, hi = wilson(k, n)
        m = int(mass[term]["ocurrencias"])
        cats = Counter(v["cat"] for v in vs)
        notes = [v["nota"] for v in vs if not v["ok"] and v["nota"]]
        records.append({
            "termino": term, "tipo": vocab[term]["tipo"], "pilar": vocab[term]["pilar"],
            "sectorial": vocab[term]["sectorial"], "ocurrencias": m,
            "documentos": mass[term]["documentos"], "empresas": mass[term]["empresas"],
            "n_muestra": n, "aciertos": k, "precision": round(p, 3),
            "ic95_inf": round(lo, 3), "ic95_sup": round(hi, 3),
            "disparos_falsos_estimados": round(m * (1 - p)),
            **{f"fp_{c}": cats[c] for c in CATEGORIES[1:]},
            "decision_propuesta": decide(p, lo, hi),
            "notas_auditor": " | ".join(dict.fromkeys(notes))[:300],
        })

    # Entradas no auditadas: en el vocabulario, las que no disparan; en los
    # candidatos, las que no pasan el corte de masa y reparto.
    for term, row in vocab.items():
        if term not in verdicts:
            records.append({"termino": term, "tipo": row["tipo"], "pilar": row["pilar"],
                            "sectorial": row["sectorial"],
                            "ocurrencias": int(mass.get(term, {}).get("ocurrencias", 0)),
                            "decision_propuesta": ("no auditada (menos de 20 disparos)" if COLLOCATIONS
                                                   else "no entra (masa o reparto insuficiente)" if CANDIDATES
                                                   else "retirar (no dispara)")})

    records.sort(key=lambda r: -r.get("disparos_falsos_estimados", -1))
    fields = list(max(records, key=len).keys())
    with open(OUT_FILE, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter=";")
        w.writeheader()
        w.writerows(records)

    audited = [r for r in records if "precision" in r]
    total_m = sum(r["ocurrencias"] for r in audited)
    p_global = sum(r["ocurrencias"] * r["precision"] for r in audited) / total_m
    var = sum((r["ocurrencias"] / total_m) ** 2 * r["precision"] * (1 - r["precision"]) / r["n_muestra"]
              for r in audited)
    unweighted = sum(r["aciertos"] for r in audited) / sum(r["n_muestra"] for r in audited)

    print(f"entradas auditadas: {len(audited)} | concordancias: {sum(r['n_muestra'] for r in audited)}")
    print(f"precision global ponderada por masa: {p_global:.3f}  "
          f"(IC95 {p_global - 1.96 * math.sqrt(var):.3f}-{p_global + 1.96 * math.sqrt(var):.3f})")
    print(f"precision sin ponderar (solo referencia): {unweighted:.3f}")
    print("\nprecision ponderada por pilar:")
    for pil in ("E", "S", "G", "TRANS"):
        sub = [r for r in audited if r["pilar"] == pil]
        mm = sum(r["ocurrencias"] for r in sub)
        print(f"  {pil:5s} {sum(r['ocurrencias'] * r['precision'] for r in sub) / mm:.3f}  "
              f"({len(sub)} entradas, {mm / total_m * 100:.1f} % de la masa)")
    fp_tot = sum(r["disparos_falsos_estimados"] for r in audited)
    print("\ncausa de los falsos positivos (ponderada por masa):")
    for c in CATEGORIES[1:]:
        share = sum(r["ocurrencias"] * r[f"fp_{c}"] / r["n_muestra"] for r in audited) / fp_tot
        print(f"  {c:10s} {share * 100:5.1f} %")
    print("\ndecisiones propuestas:", dict(Counter(r["decision_propuesta"] for r in records)))
    print("\n15 terminos que mas disparos falsos aportan:")
    acc = 0
    for r in audited[:15]:
        acc += r["disparos_falsos_estimados"]
        print(f"  {r['termino'][:28]:28s} p={r['precision']:.2f} [{r['ic95_inf']:.2f}-{r['ic95_sup']:.2f}] "
              f"falsos~{r['disparos_falsos_estimados']:6d}  acumulado {acc / fp_tot * 100:4.1f} %  -> {r['decision_propuesta']}")


if __name__ == "__main__":
    main()
