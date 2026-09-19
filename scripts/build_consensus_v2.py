"""
build_consensus_v2.py
---------------------
Segunda ronda de la revision ciega del vocabulario, y medida de la ESTABILIDAD
del consenso entre las dos rondas.

    python scripts/build_consensus_v2.py

DISENO DEL EXPERIMENTO

    Ronda 1 (revision_{1,2,3}.csv)   RECUERDO LIBRE
        Tres revisores ciegos leen la muestra y anotan los terminos ESG que se
        les ocurren. Un termino puede faltarle a un revisor por dos motivos
        distintos que la ronda 1 no sabe separar: porque no lo considera ESG, o
        porque simplemente no se le ocurrio anotarlo.

    Ronda 2 (ronda2_{1,2,3}.csv)     RECONOCIMIENTO
        Tres revisores ciegos NUEVOS reciben la union de los 532 terminos de la
        ronda 1, en orden aleatorio y sin ninguna senal de cuantos los habian
        detectado, y juzgan cada uno: valido si / no, pilar, sectorial,
        colocacion. Aqui el olvido ya no existe: si un termino se rechaza, es
        una decision.

    La diferencia entre ambas rondas es, por tanto, el RUIDO DE DETECCION. Lo
    que sobrevive a las dos es acuerdo sustantivo sobre que es un termino ESG.

POR QUE SE PUEDEN COMPARAR

    Las dos rondas se proyectan sobre el MISMO universo de 532 terminos: en la
    ronda 1, "detectado por k revisores"; en la ronda 2, "aprobado por k
    revisores". Eso permite calcular la kappa de Fleiss en ambas y compararlas.

    CAVEAT que hay que declarar: el universo de la ronda 1 es endogeno, porque
    lo definen sus propias detecciones. No existe en ella un termino con 0
    detecciones, mientras que en la ronda 2 si. La kappa de la ronda 1 esta por
    tanto sesgada AL ALZA respecto a la de una tarea con universo fijo, y la
    comparacion entre ambas es indicativa, no un contraste formal.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
METADATA_DIR = BASE_DIR / "metadata"
RESEARCH_DIR = BASE_DIR / "paper" / "_research"

# La normalizacion y la lectura del vocabulario son identicas a la ronda 1.
import sys
sys.path.insert(0, str(BASE_DIR / "scripts"))
from build_consensus import normalize, load_vocabulary, classify  # noqa: E402

PILLARS = ["E", "S", "G", "TRANS"]
PILLAR_NAME = {"E": "Ambiental (E)", "S": "Social (S)",
               "G": "Gobernanza (G)", "TRANS": "Transversal (TRANS)"}
LEVEL = {3: "máximo (3/3)", 2: "alto (2/3)", 1: "bajo (1/3)", 0: "nulo (0/3)"}


# --- estadistica -----------------------------------------------------------
def fleiss_kappa(counts: list[list[int]]) -> float:
    """counts[i][j] = numero de jueces que asignan el item i a la categoria j.
    Todas las filas deben sumar el mismo n."""
    n = sum(counts[0])
    N = len(counts)
    if n < 2 or N == 0:
        return float("nan")
    p_bar = sum((sum(c * c for c in row) - n) / (n * (n - 1)) for row in counts) / N
    totals = [sum(row[j] for row in counts) for j in range(len(counts[0]))]
    p_e = sum((t / (N * n)) ** 2 for t in totals)
    return (p_bar - p_e) / (1 - p_e) if p_e != 1 else float("nan")


def spearman(xs: list[float], ys: list[float]) -> float:
    def rank(vs):
        order = sorted(range(len(vs)), key=lambda i: vs[i])
        r = [0.0] * len(vs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


# --- carga -----------------------------------------------------------------
def load_round1() -> dict[str, dict]:
    agg: dict[str, dict] = {}
    for i in (1, 2, 3):
        seen: set[str] = set()
        with open(RESEARCH_DIR / f"revision_{i}.csv", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh, delimiter=";"):
                key = normalize((row.get("termino") or "").strip())
                if not key or key in seen:
                    continue
                seen.add(key)
                rec = agg.setdefault(key, {"surface": {}, "pilar": {},
                                           "sect": {}, "coloc": {}})
                rec["surface"][i] = row["termino"].strip()
                rec["pilar"][i] = (row.get("pilar") or "").strip().upper()
                rec["sect"][i] = (row.get("sectorial") or "0").strip()
                rec["coloc"][i] = (row.get("colocacion") or "0").strip()
    return agg


def load_round2() -> dict[str, dict]:
    agg: dict[str, dict] = {}
    for i in (1, 2, 3):
        path = RESEARCH_DIR / f"ronda2_{i}.csv"
        seen: set[str] = set()
        n_rows = 0
        with open(path, encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh, delimiter=";"):
                raw = (row.get("termino") or "").strip()
                key = normalize(raw)
                if not key or key in seen:
                    continue
                seen.add(key)
                n_rows += 1
                rec = agg.setdefault(key, {"surface": {}, "valido": {},
                                           "pilar": {}, "sect": {}, "coloc": {}})
                rec["surface"][i] = raw
                rec["valido"][i] = 1 if (row.get("valido") or "0").strip() == "1" else 0
                rec["pilar"][i] = (row.get("pilar") or "").strip().upper()
                rec["sect"][i] = (row.get("sectorial") or "0").strip()
                rec["coloc"][i] = (row.get("colocacion") or "0").strip()
        print(f"ronda2_{i}.csv: {n_rows} filas unicas, "
              f"{sum(1 for k in seen if agg[k]['valido'].get(i))} aprobadas")
    return agg


# Variantes britanico/americano que `normalize` NO unifica. No se anaden a
# SYNONYMS para no alterar las cifras ya publicadas de la ronda 1; se detectan
# y se declaran, que es peor que arreglarlo pero mejor que ocultarlo.
BRITISH = {
    "defence": "defense", "licence": "license", "offence": "offense",
    "practise": "practice", "litre": "liter", "metre": "meter",
    "colour": "color", "honour": "honor", "vapour": "vapor", "odour": "odor",
    "neighbourhood": "neighborhood", "aluminium": "aluminum",
}


def orthographic_duplicates(keys: set[str]) -> list[tuple[str, str]]:
    """Pares de claves distintas que son la misma expresion en otra ortografia."""
    pairs = set()
    for key in keys:
        alt = " ".join(BRITISH.get(w, w) for w in key.split())
        if alt != key and alt in keys:
            pairs.add(tuple(sorted((key, alt))))
    return sorted(pairs)


def main() -> None:
    r1, r2 = load_round1(), load_round2()
    rows, literals, patterns = load_vocabulary()

    dups = orthographic_duplicates(set(r1) | set(r2))
    if dups:
        print(f"\nAVISO: {len(dups)} par(es) de variantes ortograficas contadas "
              f"por separado (inflan el total en {len(dups)} termino/s):")
        for a, b in dups:
            print(f"    {a}  <->  {b}")

    only1 = set(r1) - set(r2)
    only2 = set(r2) - set(r1)
    if only1:
        print(f"\nAVISO: {len(only1)} terminos de la ronda 1 sin fila en la ronda 2")
        for k in sorted(only1)[:10]:
            print(f"    {k}")
    if only2:
        print(f"AVISO: {len(only2)} terminos nuevos aparecidos en la ronda 2")
        for k in sorted(only2)[:10]:
            print(f"    {k}")

    keys = sorted(set(r1) | set(r2))
    records = []
    for key in keys:
        a1, a2 = r1.get(key), r2.get(key)
        det = len(a1["pilar"]) if a1 else 0
        appr = sum(a2["valido"].values()) if a2 else 0
        # pilar de la ronda 2: solo votan los que aprobaron
        p2 = Counter(a2["pilar"][i] for i in a2["valido"]
                     if a2["valido"][i] and a2["pilar"].get(i)) if a2 else Counter()
        p1 = Counter(a1["pilar"].values()) if a1 else Counter()
        surface = ((a2 or a1)["surface"])
        forms = Counter(surface.values())
        top = forms.most_common(1)[0][1]
        display = sorted([f for f in forms if forms[f] == top], key=len)[0]
        sect2 = sum(1 for i in (a2["sect"] if a2 else {})
                    if a2["sect"][i] == "1" and a2["valido"][i])
        col2 = sum(1 for i in (a2["coloc"] if a2 else {})
                   if a2["coloc"][i] == "1" and a2["valido"][i])
        estado, cubierto = classify(key, display, literals, patterns)
        records.append({
            "termino": display, "clave": key,
            "r1_detecciones": det, "r2_aprobaciones": appr,
            "consenso_r1": LEVEL[det].split()[0], "consenso_r2": LEVEL[appr].split()[0],
            "estable": int(det == appr),
            "pilar_r1": p1.most_common(1)[0][0] if p1 else "",
            "pilar_r2": p2.most_common(1)[0][0] if p2 else "",
            "pilar_coincide": int(bool(p1) and bool(p2)
                                  and p1.most_common(1)[0][0] == p2.most_common(1)[0][0]),
            "pilar_disenso_r2": "/".join(sorted(p2)) if len(p2) > 1 else "",
            "sectorial_r2": int(appr and sect2 * 2 > appr),
            "colocacion_r2": int(appr and col2 * 2 > appr),
            "estado_vocabulario": estado, "cubierto_por": cubierto,
        })
    records.sort(key=lambda r: (-r["r2_aprobaciones"], -r["r1_detecciones"],
                                r["pilar_r2"] or "Z", r["clave"]))

    out_csv = METADATA_DIR / "ESG_deep_consensus_v2.csv"
    with open(out_csv, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0]), delimiter=";")
        w.writeheader()
        w.writerows(records)
    print(f"\nEscrito: {out_csv} ({len(records)} terminos)")

    # --- estadisticos --------------------------------------------------
    k1 = fleiss_kappa([[r["r1_detecciones"], 3 - r["r1_detecciones"]] for r in records])
    k2 = fleiss_kappa([[r["r2_aprobaciones"], 3 - r["r2_aprobaciones"]] for r in records])
    rho = spearman([r["r1_detecciones"] for r in records],
                   [r["r2_aprobaciones"] for r in records])
    unan1 = sum(1 for r in records if r["r1_detecciones"] in (0, 3))
    unan2 = sum(1 for r in records if r["r2_aprobaciones"] in (0, 3))
    print(f"\nkappa de Fleiss ronda 1: {k1:.3f}  (unanimidad {unan1}/{len(records)})")
    print(f"kappa de Fleiss ronda 2: {k2:.3f}  (unanimidad {unan2}/{len(records)})")
    print(f"Spearman entre niveles:  {rho:.3f}")
    stable = sum(r["estable"] for r in records)
    print(f"Mismo nivel en las dos rondas: {stable}/{len(records)} "
          f"({stable / len(records) * 100:.1f} %)")

    xtab = Counter((r["r1_detecciones"], r["r2_aprobaciones"]) for r in records)
    print("\nTabla cruzada (filas = detecciones ronda 1, cols = aprobaciones ronda 2)")
    print("        r2=0  r2=1  r2=2  r2=3   total")
    for d in (3, 2, 1):
        cells = [xtab[(d, a)] for a in range(4)]
        print(f"  r1={d} " + "".join(f"{c:6d}" for c in cells) + f"{sum(cells):8d}")
    print("  total" + "".join(
        f"{sum(xtab[(d, a)] for d in (1, 2, 3)):6d}" for a in range(4)))

    write_report(records, k1, k2, rho, xtab, dups)


# --- informe ---------------------------------------------------------------
def chi2_independence(table: list[list[int]]) -> tuple[float, int, float]:
    """chi2 de independencia y V de Cramer. Sin dependencias externas."""
    n = sum(sum(row) for row in table)
    rows_t = [sum(row) for row in table]
    cols_t = [sum(row[j] for row in table) for j in range(len(table[0]))]
    chi2 = sum((table[i][j] - rows_t[i] * cols_t[j] / n) ** 2
               / (rows_t[i] * cols_t[j] / n)
               for i in range(len(table)) for j in range(len(table[0])))
    dof = (len(table) - 1) * (len(table[0]) - 1)
    k = min(len(table), len(table[0])) - 1
    return chi2, dof, (chi2 / (n * k)) ** 0.5


def num(x: float, d: int = 3) -> str:
    return ("%%.%df" % d % x).replace(".", ",")


def write_report(records, k1, k2, rho, xtab, dups) -> None:
    total = len(records)
    by_a = Counter(r["r2_aprobaciones"] for r in records)
    gaps = [r for r in records
            if r["r2_aprobaciones"] == 3 and r["estado_vocabulario"] == "HUECO"]
    rejected = [r for r in records if r["r2_aprobaciones"] == 0]
    split = [r for r in records if r["r2_aprobaciones"] in (1, 2)]
    drop = sorted((r for r in records if r["r2_aprobaciones"] <= 1
                   and r["estado_vocabulario"] == "en vocabulario"),
                  key=lambda r: (r["r2_aprobaciones"], r["clave"]))
    both = [r for r in records if r["pilar_r1"] and r["pilar_r2"]]
    same = [r for r in both if r["pilar_coincide"]]
    chi2, dof, cramer = chi2_independence(
        [[xtab[(d, 3)], sum(xtab[(d, a)] for a in (0, 1, 2))] for d in (3, 2, 1)])

    def tbl(rows, cols=("termino", "pilar_r2", "sectorial_r2", "r1_detecciones",
                        "r2_aprobaciones", "estado_vocabulario")):
        head = {"termino": "Término", "pilar_r2": "Pilar",
                "sectorial_r2": "Sect.", "r1_detecciones": "Ronda 1",
                "r2_aprobaciones": "Ronda 2", "estado_vocabulario": "Vocabulario"}
        out = ["| " + " | ".join(head[c] for c in cols) + " |",
               "|" + "|".join("---" if c == "termino" else ":--:" for c in cols) + "|"]
        for r in rows:
            cells = []
            for c in cols:
                v = r[c]
                if c == "termino":
                    v = "`%s`" % v
                elif c == "sectorial_r2":
                    v = "sí" if str(v) == "1" else "no"
                elif c in ("r1_detecciones", "r2_aprobaciones"):
                    v = "%s/3" % v
                elif c == "estado_vocabulario":
                    v = {"en vocabulario": "**ya incluido**",
                         "cubierto": "cubierto por `%s`" % r["cubierto_por"],
                         "HUECO": "hueco"}[v]
                cells.append(str(v))
            out.append("| " + " | ".join(cells) + " |")
        return "\n".join(out)

    L = []
    A = L.append
    A("# Vocabulario ESG · ronda 2: ¿es estable el nivel de consenso?\n")
    A("La ronda 1 (`ESG_deep_consensus.md`) midió el acuerdo de tres revisores ciegos")
    A("que leían la muestra y **anotaban los términos que se les ocurrían**. Este")
    A("documento repite el ejercicio con tres evaluadores nuevos, también ciegos, a los")
    A("que se les da **la lista completa de los %d términos** de la ronda 1 en orden" % total)
    A("aleatorio y sin ninguna señal de cuántos los habían detectado, para que juzguen")
    A("cada uno: válido sí/no, pilar, sectorial, colocación.\n")
    A("El cambio de tarea es deliberado. La ronda 1 era **recuerdo libre**, donde que a")
    A("un revisor le faltara un término era ambiguo: podía ser que no lo considerase ESG")
    A("o simplemente que no se le ocurriera anotarlo. La ronda 2 es **reconocimiento**:")
    A("el olvido desaparece y un rechazo es una decisión. La diferencia entre ambas")
    A("rondas mide, por tanto, **cuánto del \"consenso\" de la ronda 1 era ruido de")
    A("detección** y cuánto era acuerdo real sobre qué es un término ESG.\n")
    A("---\n")
    A("## 1. El resultado en una línea\n")
    A("**El nivel de consenso de la ronda 1 NO es estable, y la causa es identificable:")
    A("medía sobre todo si un revisor se acordaba de anotar el término, no si lo")
    A("consideraba válido.** Cuando se les pregunta explícitamente, los evaluadores")
    A("aprueban el %.1f %% de los términos y coinciden casi siempre." % (100 * by_a[3] / total))
    A("")
    A("| | Ronda 1 (recuerdo libre) | Ronda 2 (reconocimiento) |")
    A("|---|:--:|:--:|")
    A("| Kappa de Fleiss | **%s** | **%s** |" % (num(k1), num(k2)))
    A("| Unanimidad (3/3 o 0/3) | %d de %d (%.1f %%) | %d de %d (%.1f %%) |" % (
        sum(1 for r in records if r["r1_detecciones"] == 3), total,
        100 * sum(1 for r in records if r["r1_detecciones"] == 3) / total,
        by_a[0] + by_a[3], total, 100 * (by_a[0] + by_a[3]) / total))
    A("| Interpretación al uso | acuerdo *leve* | acuerdo *sustancial* |")
    A("")
    A("La kappa pasa de %s a %s. Son la misma muestra y el mismo criterio; lo único" % (num(k1), num(k2)))
    A("que cambia es que en la ronda 2 nadie puede olvidarse de un término.\n")
    A("---\n")
    A("## 2. Tabla cruzada\n")
    A("Filas: cuántos revisores **detectaron** el término en la ronda 1. Columnas:")
    A("cuántos evaluadores lo **aprobaron** en la ronda 2.\n")
    A("| | r2 = 0/3 | r2 = 1/3 | r2 = 2/3 | r2 = 3/3 | total |")
    A("|---|---:|---:|---:|---:|---:|")
    for d in (3, 2, 1):
        cells = [xtab[(d, a)] for a in range(4)]
        A("| **r1 = %d/3** | %d | %d | %d | **%d** | %d |" % (d, *cells, sum(cells)))
    A("| **total** | %d | %d | %d | **%d** | %d |" % (
        *[sum(xtab[(d, a)] for d in (1, 2, 3)) for a in range(4)], total))
    A("")
    A("La fila que importa es la última. De los %d términos que en la ronda 1 vio **un" % sum(
        xtab[(1, a)] for a in range(4)))
    A("solo revisor** —lo que llamamos «consenso bajo»—, **%d (%.1f %%) los aprueban los" % (
        xtab[(1, 3)], 100 * xtab[(1, 3)] / sum(xtab[(1, a)] for a in range(4))))
    A("tres evaluadores de la ronda 2**. No eran candidatos débiles: eran candidatos que")
    A("a dos revisores se les pasaron.\n")
    A("### Pero la jerarquía no es ruido puro\n")
    A("La tasa de rechazo unánime sí crece de forma monótona al bajar el nivel de la")
    A("ronda 1:\n")
    A("| Nivel en la ronda 1 | Términos | Rechazo unánime en r2 | Aprobación unánime en r2 |")
    A("|---|---:|---:|---:|")
    for d in (3, 2, 1):
        tot_d = sum(xtab[(d, a)] for a in range(4))
        A("| %d/3 | %d | %d (%.1f %%) | %d (%.1f %%) |" % (
            d, tot_d, xtab[(d, 0)], 100 * xtab[(d, 0)] / tot_d,
            xtab[(d, 3)], 100 * xtab[(d, 3)] / tot_d))
    A("")
    A("El contraste de independencia da χ² = %s con %d grados de libertad," % (num(chi2, 2), dof))
    A("**p = %s**: la asociación es real y no atribuible al azar. Pero la V de Cramér" % num(
        __import__("math").exp(-chi2 / 2), 5))
    A("es **%s** y la correlación de Spearman entre los dos niveles es **%s**: el efecto" % (
        num(cramer), num(rho)))
    A("existe pero es pequeño.\n")
    A("La lectura correcta es por tanto de grado, no de tipo: el nivel de la ronda 1")
    A("**ordena bien el riesgo** —un término de consenso bajo tiene ~6 veces más")
    A("probabilidad de ser rechazado que uno de consenso máximo— pero **en términos")
    A("absolutos casi todo se aprueba**, así que usarlo como criterio de corte habría")
    A("descartado mucho material bueno.\n")
    A("Una salvedad técnica sobre la correlación: la ronda 2 concentra el %.1f %% de los" % (
        100 * by_a[3] / total))
    A("términos en un único nivel, así que apenas tiene varianza. Cualquier correlación")
    A("contra ella está atenuada por efecto techo, y el %s no debe leerse como «las dos" % num(rho))
    A("rondas no tienen nada que ver».\n")
    A("---\n")
    A("## 3. Lo más estable de todo: el pilar\n")
    A("De los %d términos con pilar asignado en ambas rondas, **%d coinciden (%.1f %%)**," % (
        len(both), len(same), 100 * len(same) / len(both)))
    A("pese a tratarse de seis personas distintas que nunca se vieron.\n")
    A("| | |")
    A("|---|---:|")
    A("| Coinciden | %d |" % len(same))
    A("| Discrepan | %d |" % (len(both) - len(same)))
    A("")
    A("Las discrepancias, por par:\n")
    A("| Ronda 1 | Ronda 2 | n |")
    A("|:--:|:--:|---:|")
    for (a, b), k in Counter((r["pilar_r1"], r["pilar_r2"]) for r in both
                             if not r["pilar_coincide"]).most_common():
        A("| %s | %s | %d |" % (a, b, k))
    A("")
    A("Este es el resultado más sólido del ejercicio completo, y respalda la decisión de")
    A("la fase A de asignar **un único pilar por término**: la adscripción E/S/G/TRANS")
    A("es reproducible entre equipos independientes. Lo que no es reproducible es la")
    A("lista de qué términos se te ocurren.\n")
    A("---\n")
    A("## 4. Qué cambia esto respecto a la ronda 1\n")
    A("### 4.1. El listado de huecos crece de 63 a %d\n" % len(gaps))
    A("Huecos = términos aprobados por los tres evaluadores de la ronda 2 que nuestro")
    A("vocabulario no captura hoy, ni como entrada literal ni por patrón.\n")
    A("| Pilar | Huecos confirmados |")
    A("|---|---:|")
    for p in PILLARS:
        A("| %s | %d |" % (PILLAR_NAME[p], sum(1 for r in gaps if r["pilar_r2"] == p)))
    A("| **Total** | **%d** |" % len(gaps))
    A("")
    A("De ellos, **%d venían marcados como «consenso bajo» en la ronda 1** y habrían" % sum(
        1 for r in gaps if r["r1_detecciones"] == 1))
    A("quedado fuera si hubiéramos usado ese nivel como filtro. Entre ellos hay material")
    A("de gobernanza y de seguridad laboral difícil de justificar como descartable:")
    A("`director independence`, `risk committee`, `competition law`,")
    A("`long-term incentive plan`, `three lines of defence`, `working hours`,")
    A("`workplace safety`, `product recall`, `supplier audit`, `personal data`,")
    A("`women in management`.\n")
    A("### 4.2. Ahora sí hay base para retirar términos\n")
    A("La ronda 1 concluía que no se podía retirar nada, y era correcto **para aquella")
    A("evidencia**: que un término no apareciera en 17 informes no dice nada en contra")
    A("suya. La ronda 2 aporta evidencia de otra clase —tres evaluadores independientes")
    A("**rechazándolo explícitamente**— y eso sí es motivo para revisar.\n")
    A("**%d términos fueron rechazados por los tres.** Los que hoy están en nuestro" % len(rejected))
    A("vocabulario o quedan cubiertos por él:\n")
    A(tbl([r for r in rejected if r["estado_vocabulario"] != "HUECO"],
          cols=("termino", "r1_detecciones", "estado_vocabulario")))
    A("")
    A("Y estos son los que **están en el vocabulario y no llegan a 2 aprobaciones**, es")
    A("decir, la lista corta de candidatos a retirar o a convertir en colocación:\n")
    A(tbl(drop, cols=("termino", "r1_detecciones", "r2_aprobaciones", "estado_vocabulario")))
    A("")
    A("El patrón es coherente entre los tres evaluadores y fácil de nombrar: **prosa de")
    A("negocio genérica** (`risk management`, `customer satisfaction`, `product quality`),")
    A("**homógrafos contables** (`internal control over financial reporting`,")
    A("`key audit matter`, `employee benefit`, `risk appetite`) y **retórica sin poder de")
    A("medida** (`sustainable value creation`, `sustainable growth`, `accountability`).")
    A("`scenario analysis` merece mención propia: se rechaza porque en la muestra")
    A("aparece como escenarios de retribución variable del consejero delegado, que es")
    A("exactamente la duda que un revisor de la ronda 1 había anotado por su cuenta.\n")
    A("### 4.3. Los %d términos repartidos\n" % len(split))
    A("Ni aprobados ni rechazados por unanimidad. Es la zona que de verdad requiere")
    A("criterio humano:\n")
    A(tbl(sorted(split, key=lambda r: (r["r2_aprobaciones"], r["clave"])),
          cols=("termino", "pilar_r2", "r1_detecciones", "r2_aprobaciones",
                "estado_vocabulario")))
    A("")
    A("---\n")
    A("## 5. Huecos confirmados, por pilar\n")
    A("Los %d términos aprobados 3/3 en la ronda 2 que el vocabulario no captura." % len(gaps))
    A("`Ronda 1` indica cuántos revisores lo habían detectado en el primer ejercicio.\n")
    for p in PILLARS:
        sub = sorted((r for r in gaps if r["pilar_r2"] == p), key=lambda r: r["clave"])
        if not sub:
            continue
        A("### %s — %d\n" % (PILLAR_NAME[p], len(sub)))
        A(tbl(sub, cols=("termino", "sectorial_r2", "r1_detecciones")))
        A("")
    A("---\n")
    A("## 6. Límites de este ejercicio\n")
    A("- **El universo de la ronda 1 es endógeno.** Lo definen sus propias detecciones,")
    A("  así que en ella no existe un término con 0 detecciones mientras que en la ronda")
    A("  2 sí. Eso sesga su kappa **al alza**, con lo que la brecha real entre %s y %s" % (num(k1), num(k2)))
    A("  es si acaso mayor que la medida. La comparación es indicativa, no un contraste")
    A("  formal entre dos tareas equivalentes.")
    A("- **La ronda 2 no aporta términos nuevos por construcción.** Solo puede validar o")
    A("  rechazar lo que la ronda 1 encontró. Si las tres primeras personas pasaron por")
    A("  alto una familia entera de vocabulario, este ejercicio no la recupera.")
    A("- **Sigue siendo la misma muestra** de 17 informes sobre 343. Todo lo dicho aquí")
    A("  se refiere a cómo se comportan estos términos en esta muestra.")
    if dups:
        A("- **%d par(es) de variantes ortográficas** se contabilizan por separado" % len(dups))
        A("  (%s), así que de los %d términos hay %d conceptos" % (
            ", ".join("`%s`/`%s`" % (a.split()[-1], b.split()[-1]) for a, b in dups),
            total, total - len(dups)))
        A("  distintos. Se declara en vez de corregirse para no alterar las cifras ya")
        A("  publicadas de la ronda 1.")
    A("")
    A("---\n")
    A("## 7. Qué hacer\n")
    A("1. **Dejar de usar el nivel de consenso de la ronda 1 como filtro.** Es un")
    A("   indicador de riesgo débil, no un criterio de inclusión: descartaría %d" % sum(
        1 for r in gaps if r["r1_detecciones"] == 1))
    A("   términos buenos para evitar %d malos." % xtab[(1, 0)])
    A("2. **Incorporar los %d huecos confirmados**, priorizando los no sectoriales." % len(gaps))
    A("3. **Revisar los %d términos del vocabulario que no llegan a 2 aprobaciones**," % len(drop))
    A("   decidiendo caso a caso entre retirarlos y convertirlos en colocación.")
    A("4. **Resolver a mano los %d repartidos.**" % len(split))
    A("5. **Dar por cerrada la asignación de pilares**: con un %.1f %% de coincidencia" % (
        100 * len(same) / len(both)))
    A("   entre equipos independientes, no es donde está el riesgo del vocabulario.\n")
    A("---\n")
    A("Datos completos en `metadata/ESG_deep_consensus_v2.csv`. Juicios sin agregar en")
    A("`paper/_research/ronda2_{1,2,3}.csv`; lista entregada a los evaluadores en")
    A("`paper/_research/terminos_ronda2.txt`. Ronda 1 en `informes/ESG_deep_consensus.md`.")

    # Decimal espanol tambien en los porcentajes construidos con %.1f.
    text = re.sub(r"(\d)\.(\d+ %)", r"\1,\2", "\n".join(L) + "\n")

    out = BASE_DIR / "informes" / "ESG_deep_consensus_v2.md"
    out.write_text(text, encoding="utf-8")
    print(f"Escrito: {out}")


if __name__ == "__main__":
    main()
