"""
build_vocab_v3.py
-----------------
Construye la PROPUESTA de vocabulario v3 a partir de la auditoria de falsos
positivos. No toca el vocabulario activo (esg_terms.txt / ESG_terms_final.csv):
eso se hace al cerrar el vocabulario, despues de validarlo con Francisco.

    python scripts/build_vocab_v3.py

Entrada
    metadata/ESG_terms_final.csv                vocabulario vigente (361)
    metadata/ESG_terms_v3_decisiones.csv        decisiones razonadas termino a termino
    metadata/ESG_deep_consensus_v3.csv          auditoria del vocabulario vigente
    metadata/ESG_deep_consensus_v3_candidatos.csv   auditoria de los huecos de la ronda 2
    metadata/ESG_deep_consensus_v3_colocaciones.csv auditoria de las colocaciones nuevas

Salida
    metadata/ESG_terms_v3.csv                   vocabulario propuesto
    paper/_research/vocab_v3_resumen.json       cifras para el informe

REGLAS

  * Los terminos sin decision explicita se mantienen. La decision por defecto
    de la auditoria ("revisar") no retira nada por si sola: con 6 concordancias
    una precision de 0,5-0,67 es muy ruidosa, y un termino ligero solo se
    retira si los auditores identificaron un homografo sistematico.
  * Solo entran los candidatos que la auditoria propone mantener (p >= 0,80),
    y solo si superaron el corte de masa y reparto (>= 50 disparos, >= 5
    empresas).
  * Las colocaciones nuevas heredan pilar y sectorialidad del termino al que
    sustituyen. Se auditan aparte (ESG_deep_consensus_v3_colocaciones.csv):
    las que la auditoria propone retirar se descartan, y si todas las de un
    termino fallan, el termino se retira sin sustituto (familia non-financial).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
METADATA_DIR = BASE_DIR / "metadata"
RESEARCH_DIR = BASE_DIR / "paper" / "_research"

ACTIONS = {"retirar", "sustituir", "restringir", "mantener_aviso"}

# Patrones corregidos despues de auditarlos, con la forma en que se auditaron.
# La auditoria busca cada patron sin \b y el pipeline lo envuelve en \b...\b,
# asi que "...emission" no disparaba en el pipeline ante "emissions".
AUDITED_AS = {
    r"offsett?ing\s+(?:of\s+)?(?:\w+\s+)?emission\w*":
        r"offsett?ing\s+(?:of\s+)?(?:\w+\s+)?emission",
}


def read(path: Path, **kw) -> list[dict]:
    with open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh, delimiter=";", **kw))


def strip_re(term: str) -> tuple[str, str]:
    """'re:environ\\w+' -> ('environ\\w+', 'patron')."""
    return (term[3:], "patron") if term.startswith("re:") else (term, "termino")


def main() -> None:
    current = read(METADATA_DIR / "ESG_terms_final.csv")
    audit = {r["termino"]: r for r in read(METADATA_DIR / "ESG_deep_consensus_v3.csv")}
    cand = read(METADATA_DIR / "ESG_deep_consensus_v3_candidatos.csv")
    coll_path = METADATA_DIR / "ESG_deep_consensus_v3_colocaciones.csv"
    coll = {r["termino"]: r for r in read(coll_path)} if coll_path.exists() else {}

    decisions = {}
    for r in read(METADATA_DIR / "ESG_terms_v3_decisiones.csv"):
        term, _ = strip_re(r["termino"])
        if r["accion"] not in ACTIONS:
            raise ValueError(f"accion desconocida {r['accion']!r} en {term}")
        decisions[term] = r
    by_term = {r["termino"]: r for r in current}
    missing = [t for t in decisions if t not in by_term]
    if missing:
        raise ValueError(f"decisiones sobre terminos que no estan en el vocabulario: {missing}")

    out, removed, keys = [], [], set()

    def add(row: dict) -> None:
        key = row["termino"].lower()
        if key not in keys:
            keys.add(key)
            out.append(row)

    for row in current:
        term = row["termino"]
        a = audit.get(term, {})
        d = decisions.get(term)
        base = {"termino": term, "tipo": row["tipo"], "pilar": row["pilar"],
                "sectorial": row["sectorial"], "origen": row.get("origen", ""),
                "precision": a.get("precision", ""), "ocurrencias": a.get("ocurrencias", ""),
                "aviso": ""}
        if d is None:
            add(base)
            continue
        if d["accion"] == "mantener_aviso":
            base["aviso"] = d["motivo"]
            add(base)
            continue
        removed.append({"termino": term, "accion": d["accion"], "motivo": d["motivo"],
                        "precision": a.get("precision", ""), "ocurrencias": a.get("ocurrencias", "")})
        for sub in filter(None, d["sustituto"].split("|")):
            text, kind = strip_re(sub)
            c = coll.get(AUDITED_AS.get(text, text))
            if c and c["decision_propuesta"] == "retirar":
                removed.append({"termino": text, "accion": "colocacion descartada",
                                "motivo": f"auditada: p={c['precision']}",
                                "precision": c["precision"], "ocurrencias": c["ocurrencias"]})
                continue
            if not c:
                aviso = "colocacion nueva sin auditar"
            elif not c.get("precision"):
                aviso = "colocacion nueva, menos de 20 disparos"
            elif c["decision_propuesta"] == "mantener":
                aviso = "colocacion nueva auditada"
            else:
                aviso = f"colocacion nueva a revisar (p={c['precision']})"
            add({"termino": text, "tipo": kind, "pilar": row["pilar"],
                 "sectorial": row["sectorial"], "origen": f"v3: sustituye a {term}",
                 "precision": c.get("precision", "") if c else "",
                 "ocurrencias": c.get("ocurrencias", "") if c else "", "aviso": aviso})

    added_cand = []
    for r in cand:
        if r["decision_propuesta"] != "mantener":
            continue
        row = {"termino": r["termino"], "tipo": "termino", "pilar": r["pilar"],
               "sectorial": r["sectorial"], "origen": "v3: hueco ronda 2 auditado",
               "precision": r["precision"], "ocurrencias": r["ocurrencias"], "aviso": ""}
        if row["termino"].lower() not in keys:
            add(row)
            added_cand.append(row)

    fields = ["termino", "tipo", "pilar", "sectorial", "origen", "precision", "ocurrencias", "aviso"]
    with open(METADATA_DIR / "ESG_terms_v3.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter=";")
        w.writeheader()
        w.writerows(out)

    # --- precision estimada ------------------------------------------------
    def weighted(rows):
        rows = [r for r in rows if r["precision"] not in ("", None) and r["ocurrencias"] not in ("", None)]
        m = sum(float(r["ocurrencias"]) for r in rows)
        return sum(float(r["ocurrencias"]) * float(r["precision"]) for r in rows) / m, m

    p_before, m_before = weighted([r for r in audit.values() if r.get("precision")])
    kept = [r for r in out if r["origen"] != "v3: hueco ronda 2 auditado" and r["precision"]]
    if not coll:
        # sin auditoria de colocaciones: environ restringido estimado con sus propios veredictos
        kept = kept + [{"precision": "0.833", "ocurrencias": str(round(
            24 / 40 * float(audit["environ\\w+"]["ocurrencias"])))}]
    p_kept, m_kept = weighted(kept)
    p_after, m_after = weighted(kept + added_cand)

    pillars = {}
    for p in ("E", "S", "G", "TRANS"):
        pillars[p] = {"antes": sum(1 for r in current if r["pilar"] == p),
                      "despues": sum(1 for r in out if r["pilar"] == p)}

    summary = {
        "entradas_antes": len(current), "entradas_despues": len(out),
        "retiradas_o_sustituidas": len(removed),
        "colocaciones_nuevas": sum(1 for r in out if r["origen"].startswith("v3: sustituye")),
        "colocaciones_descartadas": sum(1 for r in removed if r["accion"] == "colocacion descartada"),
        "candidatos_incorporados": len(added_cand),
        "precision_antes": round(p_before, 3), "masa_auditada_antes": round(m_before),
        "precision_tras_depurar": round(p_kept, 3), "masa_tras_depurar": round(m_kept),
        "precision_tras_depurar_y_ampliar": round(p_after, 3), "masa_tras_ampliar": round(m_after),
        "pilares": pillars, "retirados": removed,
        "candidatos": [{"termino": r["termino"], "pilar": r["pilar"], "precision": r["precision"],
                        "ocurrencias": r["ocurrencias"]} for r in added_cand],
    }
    (RESEARCH_DIR / "vocab_v3_resumen.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"entradas: {len(current)} -> {len(out)}")
    print(f"  retiradas o sustituidas: {len(removed)}")
    print(f"  colocaciones nuevas: {summary['colocaciones_nuevas']} "
          f"(descartadas tras auditarlas: {summary['colocaciones_descartadas']})")
    print(f"  candidatos de la ronda 2 incorporados: {len(added_cand)}")
    print(f"precision ponderada por masa (parte auditada):")
    print(f"  vocabulario vigente:        {p_before:.3f}  sobre {m_before:,.0f} disparos")
    print(f"  tras depurar:               {p_kept:.3f}  sobre {m_kept:,.0f} disparos")
    print(f"  tras depurar y ampliar:     {p_after:.3f}  sobre {m_after:,.0f} disparos")
    print("entradas por pilar:", {k: f"{v['antes']}->{v['despues']}" for k, v in pillars.items()})


if __name__ == "__main__":
    main()
