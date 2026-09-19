"""
build_vocab_v3_report.py
------------------------
Informe legible del vocabulario propuesto v3, por categorias de confianza.

    python scripts/build_vocab_v3_report.py

Lee las salidas de build_vocab_v3.py y de las auditorias, y escribe
informes/ESG_vocabulario_v3.md.

Las recomendaciones sobre los terminos dudosos siguen una regla fija:
  * si la mayoria de sus fallos son de NAVEGACION (indices, tablas de
    correspondencia, glosarios, encabezados), se deja: el problema es de la
    extraccion, no del termino;
  * si fallan por SIGNIFICADO (homografos, nombres propios, prosa de negocio),
    la recomendacion es manual y va razonada en MANUAL.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
MET = BASE / "metadata"
RES = BASE / "paper" / "_research"

KEEP_AT = 0.80
NAV_SHARE = 0.60

# Terminos que fallan por significado: recomendacion razonada.
MANUAL = {
    "water": ("dejar", "roza el umbral (78 % con 40 muestras); los fallos son homografos dispersos (bombas de agua, amortizacion de terrenos) sin patron comun"),
    "electricity": ("dejar", "75 % con 20 muestras; falla en eléctricas al hablar de su negocio, pero es vocabulario ambiental central"),
    "inclusion": ("dejar", "75 %; homografos dispersos (inclusion en el perimetro, IFRS 17) sin patron que merezca colocacion"),
    "independent director": ("dejar", "todos sus fallos son biografias de consejeros; el resto es divulgacion de gobierno. Se arreglaria mejor con un filtro de biografias"),
    "circular economy": ("dejar", "75 %; fallos sueltos en tendencias de mercado"),
    "internal audit": ("dejar", "estructura de gobierno; falla por el informe del auditor externo, un ruido que conviene filtrar en extraccion"),
    "recycling": ("dejar, vigilar", "67 %; homografo contable sistematico en bancos y aseguradoras (reciclaje de OCI a resultados), que infla a ese sector"),
    "due diligence": ("dejar", "75 %; fallos repartidos entre tres causas distintas"),
    "environmentally": ("dejar", "67 % con 6 muestras; falla por texto citado de metas ODS"),
    "social responsibility": ("dejar", "67 %; fallos variados, ninguno sistematico"),
    "injur\\w+": ("dejar", "67 %; mezcla glosarios, seguros y riesgo operacional sin patron dominante"),
    "works council": ("dejar", "58 %; casi todo son biografias de consejeros; es vocabulario central de relaciones laborales"),
    "local\\s+communit\\w+": ("dejar", "75 %; fallos sueltos"),
    "compliance committee": ("dejar", "todos sus fallos son cargos en otras empresas dentro de biografias; mismo arreglo que independent director"),
    "circularity": ("dejar", "50 % con 6 muestras; nombres de empresas y lineas de negocio. Poca masa"),
    "transition plan": ("dejar", "67 %; un fallo curioso (relevo del socio auditor) y una tabla"),
    "animal welfare": ("dejar", "50 %; falla por la declaracion legal francesa de que el tema no es material, que tambien es divulgacion aunque sea negativa"),
    "wind farm": ("dejar", "50 %; en electricas aparece en notas contables y listas de filiales, pero son sus activos renovables. Sectorial de hecho"),
    "flood": ("dejar", "50 %; listas de riesgos genericas, pero el riesgo fisico climatico es contenido E legitimo"),
    "recyclable": ("descartar", "sus fallos son el homografo contable 'not recyclable to profit or loss'; el sentido ESG ya lo cubre recyclable packaging"),
    "human capital": ("dejar", "67 %; marco de capitales del modelo de negocio, cercano a divulgacion"),
    "absenteeism": ("dejar", "50 % con 6 muestras; es un KPI laboral estandar"),
    "hydroelectric": ("dejar", "50 %; tres fallos de tres tipos distintos"),
    "energy storage": ("dejar", "50 %; descripcion de producto, pero ya esta marcado como sectorial y pesa poco"),
    "indigenous": ("dejar", "67 %; fallos sueltos (nombre de fundacion, meta ODS)"),
    "social protection": ("dejar", "67 %; fallos sueltos"),
    "responsible investment": ("dejar", "33 %, pero casi todo son glosarios y el nombre de los PRI, que es ESG"),
    "ergonomic\\w*": ("dejar", "67 %; glosario y un software"),
    "land use": ("dejar", "67 %; urbanismo de planta y derechos de uso del suelo"),
    "energy mix": ("dejar", "67 %; tendencias de mercado del sector energetico"),
    "lobbying": ("dejar", "50 %; seguimiento legislativo, cercano a divulgacion de gobierno"),
    "clean energy": ("dejar", "50 %; nombres propios (Clean Energy Fuels) y contexto regulatorio"),
    "apprentice": ("descartar", "33 %; casi siempre en notas metodologicas que excluyen a los aprendices del perimetro de plantilla"),
    "food safety": ("dejar", "50 %; nombres de instituciones; sectorial de alimentacion"),
    "social performance": ("dejar", "67 %; boilerplate de alcance del informe"),
    "tax compliance": ("descartar", "50 %; honorarios de servicios fiscales del auditor y la ley FATCA; tax transparency cubre el sentido ESG"),
    "proxy advisor": ("dejar", "67 %; fallos sueltos"),
    "training and development": ("dejar", "67 %; un caso de I+D de componentes"),
    "green energy": ("dejar", "50 %; nombres de empresas y tendencias de mercado"),
    "geothermal": ("dejar", "50 %; deterioro de activos y glosario"),
    "ftse4good": ("dejar", "67 %; aparece en la lista de indices bursatiles de la accion, que es pertenecer a un indice ESG"),
    "executive compensation": ("dejar", "50 %; titulos de informes y marco regulatorio"),
    "compliance violation": ("dejar", "50 %; aparece en provisiones por litigios, que tambien es informacion de gobierno"),
    "affordability": ("descartar", "sus fallos son politica comercial de precios; poca masa"),
    "persons with disabilities": ("dejar", "50 %; texto citado de metas ODS y tablas ESRS"),
    "alternative fuel": ("dejar", "67 %; nombre de un reglamento"),
    "wind turbine": ("descartar", "33 %; demanda de semiconductores, contratos de construccion: prosa de negocio"),
    "wind energy": ("descartar", "33 %; nombres de filiales y unidades de negocio"),
    "solar farm": ("dejar", "67 %; contabilidad de PPA, pero son activos renovables"),
    "msci esg": ("dejar", "67 %; disclaimer legal de MSCI"),
    "tax avoidance": ("descartar", "33 %; normativa fiscal y boilerplate del verificador"),
    "social norm": ("descartar", "50 %; riesgo conductual interno; poca masa"),
    "b corp": ("dejar", "67 %; certificacion de una agencia de rating citada"),
    "low-carbon fuel": ("dejar", "67 %; nombres de normas y alianzas"),
    "supply chain footprint": ("descartar", "huella geografica o industrial, el mismo problema que footprint suelto"),
    "supply chain labour": ("descartar", "artefacto de extraccion: une dos temas de una lista"),
    "fuel consumption": ("dejar", "67 %; un aviso legal repetido y un glosario; es un KPI ambiental estandar"),
    "compliance officer": ("dejar", "67 %; organigramas con nombres, que son estructura de gobierno"),
    "collective bargaining": ("dejar", "67 %; un encabezado y una definicion; es un KPI laboral estandar"),
    "materiality analysis": ("dejar", "67 %; referencia a la web y el informe del verificador; sustituye a materiality, que era mucho peor (42 %)"),
}

AVISOS = {
    "sustainab\\w+": ("dejar y declararlo", "el discurso vago de sostenibilidad ('sustainable growth') es en parte lo que un indice de greenwashing quiere medir; neutralizar esos usos solo sube un punto"),
    "sustainable": ("dejar y declararlo", "mismo caso que sustainab\\w+"),
    "governance": ("dejar y mejorar la extraccion", "14 de sus 17 fallos son encabezados y referencias cruzadas que el filtro de indices no captura; cuando describe el consejo acierta"),
    "audit committee": ("dejar", "biografias y boilerplate del auditor; pesa poco comparado con los dos anteriores"),
    "hydrogen": ("dejar", "casi todos sus fallos son la misma plantilla legal de la taxonomia sobre nuclear y gas"),
    "tcfd": ("dejar", "todos sus fallos son tablas de correspondencia; en prosa es valido"),
}

PILLARS = [("E", "Ambiental"), ("S", "Social"), ("G", "Gobernanza"), ("TRANS", "Transversal")]


# Los motivos y las notas de los auditores se escribieron sin tildes. Se
# corrigen al renderizar, solo fuera de los spans de codigo, con una tabla
# explicita y la terminacion -cion/-sion, para no tocar lo que no toca.
ACCENTS = {
    "auditoria": "auditoría", "auditorias": "auditorías", "biografia": "biografía",
    "biografias": "biografías", "bateria": "batería", "baterias": "baterías",
    "categoria": "categoría", "climatico": "climático", "deontologico": "deontológico",
    "economico": "económico", "electricas": "eléctricas", "energetico": "energético",
    "estandar": "estándar", "fisico": "físico", "genericas": "genéricas",
    "geografica": "geográfica", "grafico": "gráfico", "guia": "guía",
    "homografo": "homógrafo", "homografos": "homógrafos", "indice": "índice",
    "indices": "índices", "juridica": "jurídica", "maritimas": "marítimas",
    "metodologicas": "metodológicas", "metricas": "métricas", "pagina": "página",
    "paginas": "páginas", "politica": "política", "sistematico": "sistemático",
    "tambien": "también", "taxonomia": "taxonomía", "triada": "tríada",
    "valido": "válido", "codigo": "código", "organos": "órganos",
    "perimetro": "perímetro", "arreglaria": "arreglaría",
    "patron": "patrón", "opinion": "opinión", "linea": "línea", "lineas": "líneas",
    "comun": "común",
}


def accent(text: str) -> str:
    def fix_word(m):
        word = m.group(0)
        low = word.lower()
        if low in ACCENTS:
            rep = ACCENTS[low]
        elif low.endswith(("cion", "sion")):
            rep = low[:-2] + "ón"
        else:
            return word
        return rep[0].upper() + rep[1:] if word[0].isupper() else rep

    parts = re.split(r"(`[^`]*`)", text)
    return "".join(part if part.startswith("`") else re.sub(r"\b[A-Za-z]+\b", fix_word, part)
                   for part in parts)


def rd(path):
    with open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh, delimiter=";"))


def pct(x):
    return f"{float(x) * 100:.0f} %" if x not in ("", None) else "—"


def num(x):
    return f"{int(float(x)):,}".replace(",", ".") if x not in ("", None) else "—"


def code(t):
    return "`" + t.replace("|", "\\|") + "`"


def main():
    v3 = rd(MET / "ESG_terms_v3.csv")
    final = {r["termino"]: r for r in rd(MET / "ESG_terms_final.csv")}
    audit = {}
    for f in ("ESG_deep_consensus_v3.csv", "ESG_deep_consensus_v3_candidatos.csv",
              "ESG_deep_consensus_v3_colocaciones.csv"):
        audit.update({r["termino"]: r for r in rd(MET / f)})
    cand = rd(MET / "ESG_deep_consensus_v3_candidatos.csv")
    summ = json.loads((RES / "vocab_v3_resumen.json").read_text(encoding="utf-8"))

    def is_colloc(r):
        return r["origen"].startswith("v3: sustituye") or final.get(r["termino"], {}).get("colocacion") == "1"

    avisos = [r for r in v3 if r["termino"] in AVISOS]
    doubts = [r for r in v3 if r["termino"] not in AVISOS and r["precision"] and float(r["precision"]) < KEEP_AT]
    colloc = [r for r in v3 if r not in avisos and r not in doubts and is_colloc(r)]
    clear = [r for r in v3 if r not in avisos and r not in doubts and r not in colloc]

    missing = [r["termino"] for r in doubts
               if r["termino"] not in MANUAL and
               (lambda a: (int(a["fp_indice"]) / max(1, sum(int(a[k]) for k in ("fp_indice", "fp_financiero", "fp_generico", "fp_otro")))) < NAV_SHARE)(audit[r["termino"]])]
    if missing:
        raise SystemExit(f"faltan recomendaciones manuales para: {missing}")

    L = []
    A = L.append
    A("# Vocabulario ESG v3: propuesta por niveles de confianza\n")
    A("Este documento recoge el vocabulario que proponemos tras la auditoría de falsos positivos. "
      "Está ordenado de más a menos seguro: primero los términos que funcionan solos, después los que solo "
      "funcionan acompañados (colocaciones) y al final las dudas, con la recomendación y el motivo de cada una.\n")
    A("## Resumen\n")
    A("| | |")
    A("|---|---|")
    A(f"| Vocabulario de partida | {summ['entradas_antes']} entradas |")
    A(f"| Vocabulario propuesto | **{summ['entradas_despues']} entradas** |")
    A(f"| Retiradas o sustituidas | {summ['retiradas_o_sustituidas']} |")
    A(f"| Colocaciones nuevas | {summ['colocaciones_nuevas']} |")
    A(f"| Términos nuevos (huecos de la ronda 2) | {summ['candidatos_incorporados']} |")
    A(f"| Precisión ponderada por masa, vocabulario de partida | {summ['precision_antes'] * 100:.1f} % |".replace(".", ","))
    A(f"| Precisión ponderada por masa, vocabulario propuesto | **{summ['precision_tras_depurar_y_ampliar'] * 100:.1f} %** |".replace(".", ","))
    A("")
    A("| Categoría | Entradas |")
    A("|---|---:|")
    A(f"| 1. Claros | {len(clear)} |")
    A(f"| 2. Con colocación | {len(colloc)} |")
    A(f"| 3. Dudas que se mantienen | {len(avisos) + len(doubts)} |")
    A("")
    A("### Cómo leer las cifras\n")
    A("- **Precisión**: de las apariciones del término en el corpus que revisamos en contexto, qué porcentaje "
      "eran menciones ESG reales. Se revisaron entre 6 y 40 apariciones por término, más cuanto más pesa el término.")
    A("- **Disparos**: cuántas veces aparece el término en las zonas extraídas de los 343 informes, "
      "ya sin índices de contenidos. Es su peso en el índice.")
    A("- **Precisión ponderada por masa**: la precisión media del vocabulario dando a cada término el peso de sus "
      "disparos. Es la que importa: un término que falla mucho pero aparece poco apenas mueve el índice.")
    A("- Con 6 apariciones revisadas, una precisión del 50-67 % es poco fiable (el intervalo va aproximadamente del 20 % al 90 %). "
      "Por eso un término ligero con precisión baja no se retira solo por la cifra, sino cuando hay un motivo sistemático.")
    A("- `[S]` marca los términos **sectoriales**. `TRANS` son términos transversales: cuentan en el índice agregado "
      "pero en ningún subíndice de pilar.\n")

    # --- 1. claros ----------------------------------------------------------
    A("---\n")
    A("## 1. Vocabulario claro\n")
    A(f"Términos con precisión igual o superior al {KEEP_AT * 100:.0f} % que funcionan por sí solos. "
      "Ordenados por peso dentro de cada pilar.\n")
    for p, name in PILLARS:
        rows = sorted((r for r in clear if r["pilar"] == p), key=lambda r: -float(r["ocurrencias"] or 0))
        A(f"### {name} ({p}) — {len(rows)} términos\n")
        A("| Término | Precisión | Disparos | Origen |")
        A("|---|---:|---:|---|")
        for r in rows:
            origin = "nuevo (ronda 2)" if r["origen"].startswith("v3: hueco") else ""
            A(f"| {code(r['termino'])}{' `[S]`' if r['sectorial'] == '1' else ''} | {pct(r['precision'])} | {num(r['ocurrencias'])} | {origin} |")
        A("")

    # --- 2. colocaciones ------------------------------------------------------
    A("---\n")
    A("## 2. Vocabulario con colocación\n")
    A("La palabra suelta es ambigua (tiene un significado no ESG frecuente en estas memorias), así que solo se "
      "cuenta dentro de una expresión que fija el sentido. Por ejemplo, `footprint` suelto es a menudo la "
      "\"huella industrial\" de la empresa; `carbon footprint` no deja duda.\n")
    for p, name in PILLARS:
        rows = sorted((r for r in colloc if r["pilar"] == p), key=lambda r: -float(r["ocurrencias"] or 0))
        if not rows:
            continue
        A(f"### {name} ({p}) — {len(rows)} colocaciones\n")
        A("| Colocación | Precisión | Disparos | Origen |")
        A("|---|---:|---:|---|")
        for r in rows:
            if r["origen"].startswith("v3: sustituye a "):
                origin = "sustituye a " + code(r["origen"][len("v3: sustituye a "):])
            else:
                origin = "colocación previa"
            note = " (menos de 20 disparos, sin revisar)" if r["aviso"].startswith("colocacion nueva, menos") else ""
            A(f"| {code(r['termino'])}{' `[S]`' if r['sectorial'] == '1' else ''} | {pct(r['precision'])} | {num(r['ocurrencias'])} | {origin}{note} |")
        A("")

    # --- 3. dudas -------------------------------------------------------------
    A("---\n")
    A("## 3. Dudas\n")
    A("### 3.1. Términos con mucho peso y precisión baja\n")
    A("Son los que más importan: entre `sustainab\\w+` y `governance` suman uno de cada siete disparos falsos del "
      "vocabulario. Recomendamos mantenerlos todos, pero conviene decidirlo explícitamente y declararlo.\n")
    A("| Término | Pilar | Precisión | Disparos | Recomendación | Por qué |")
    A("|---|:--:|---:|---:|---|---|")
    for r in sorted(avisos, key=lambda r: -float(r["ocurrencias"])):
        rec, why = AVISOS[r["termino"]]
        A(f"| {code(r['termino'])} | {r['pilar']} | {pct(r['precision'])} | {num(r['ocurrencias'])} | **{rec}** | {why} |")
    A("")

    nav, meaning = [], []
    for r in doubts:
        a = audit[r["termino"]]
        fps = {k: int(a[k]) for k in ("fp_indice", "fp_financiero", "fp_generico", "fp_otro")}
        tot = sum(fps.values())
        (nav if r["termino"] not in MANUAL and fps["fp_indice"] / max(1, tot) >= NAV_SHARE else meaning).append(r)

    A("### 3.2. Precisión baja por fallos de navegación: dejar\n")
    A("Fallan sobre todo porque aparecen en índices, tablas de correspondencia GRI/ESRS, glosarios o encabezados. "
      "El término en sí es correcto: cuando aparece en prosa, es divulgación ESG. Quitarlos tiraría señal buena "
      "por un problema que es de la extracción, y que en parte ya corrige el filtro de índices.\n")
    A("| Término | Pilar | Precisión | Disparos | Qué falla |")
    A("|---|:--:|---:|---:|---|")
    for r in sorted(nav, key=lambda r: -float(r["ocurrencias"])):
        a = audit[r["termino"]]
        notes = "; ".join(n for n in a["notas_auditor"].split(" | ")[:3] if n) or             f"{a['fp_indice']} de {int(a['n_muestra']) - int(a['aciertos'])} fallos en indices y tablas de correspondencia"
        A(f"| {code(r['termino'])} | {r['pilar']} | {pct(r['precision'])} | {num(r['ocurrencias'])} | {notes} |")
    A("")

    A("### 3.3. Precisión baja por significado: caso a caso\n")
    A("Fallan porque el término tiene otro significado en estas memorias (contable, comercial, un nombre propio). "
      "Aquí la decisión sí depende del término.\n")
    drop = [r for r in meaning if MANUAL[r["termino"]][0] == "descartar"]
    A(f"De {len(meaning)} términos, recomendamos **descartar {len(drop)}** y dejar el resto.\n")
    A("| Término | Pilar | Precisión | Disparos | Recomendación | Por qué |")
    A("|---|:--:|---:|---:|---|---|")
    order = {"descartar": 0, "dejar, vigilar": 1, "dejar": 2}
    for r in sorted(meaning, key=lambda r: (order[MANUAL[r["termino"]][0]], -float(r["ocurrencias"]))):
        rec, why = MANUAL[r["termino"]]
        A(f"| {code(r['termino'])} | {r['pilar']} | {pct(r['precision'])} | {num(r['ocurrencias'])} | **{rec}** | {why} |")
    A("")

    A("### 3.4. Términos retirados del vocabulario de partida\n")
    A("Ya están fuera de la propuesta. Se listan para que se pueda revisar cada decisión.\n")
    A("| Término | Precisión | Disparos | Qué se hace | Por qué |")
    A("|---|---:|---:|---|---|")
    decisions = {r["termino"].removeprefix("re:"): r for r in rd(MET / "ESG_terms_v3_decisiones.csv")}
    for r in sorted(summ["retirados"], key=lambda r: -float(r["ocurrencias"] or 0)):
        d = decisions.get(r["termino"], {})
        subs = [s.removeprefix("re:") for s in d.get("sustituto", "").split("|") if s]
        action = ("sustituido por " + ", ".join(code(s) for s in subs)) if subs else \
                 ("restringido" if r["accion"] == "restringir" else "retirado")
        if r["accion"] == "restringir":
            action = "restringido a " + ", ".join(code(s) for s in subs)
        A(f"| {code(r['termino'])} | {pct(r['precision'])} | {num(r['ocurrencias'])} | {action} | {r['motivo']} |")
    A("")

    A("### 3.5. Términos nuevos que no entran\n")
    A("Huecos que la ronda 2 aprobó por unanimidad como ESG, pero que al revisarlos en contexto no alcanzan el 80 % "
      "de precisión. Quedan fuera de la propuesta; se pueden recuperar con una colocación.\n")
    A("| Término | Pilar | Precisión | Disparos | Qué falla |")
    A("|---|:--:|---:|---:|---|")
    for r in sorted((c for c in cand if c["decision_propuesta"] in ("revisar", "retirar")),
                    key=lambda r: -float(r["ocurrencias"])):
        notes = r["notas_auditor"].split(" | ")[:3]
        A(f"| {code(r['termino'])} | {r['pilar']} | {pct(r['precision'])} | {num(r['ocurrencias'])} | {'; '.join(notes)} |")
    A("")
    low = sorted((c for c in cand if c["decision_propuesta"].startswith("no entra")), key=lambda r: -float(r["ocurrencias"]))
    A(f"Otros **{len(low)}** términos nuevos no se revisaron porque aparecen menos de 50 veces o en menos de 5 "
      "empresas: no moverían el índice y suelen ser vocabulario de una sola compañía.\n")
    A(", ".join(f"{code(c['termino'])} ({num(c['ocurrencias'])})" for c in low))
    A("")

    # --- 4. pendientes ----------------------------------------------------------
    A("---\n")
    A("## 4. Decisiones pendientes\n")
    A("1. **Taxonomía europea**: `eu taxonomy`, `taxonomy eligible` y `taxonomy alignment` están hoy en **TRANS**, "
      "pero en la revisión ciega dos de cada tres revisores los pusieron en E. El reglamento clasifica actividades según "
      "seis objetivos que son todos ambientales (mitigación, adaptación, agua, economía circular, contaminación y "
      "biodiversidad), así que no es un marco transversal a los tres pilares. Proponemos pasar toda la familia a **E**.")
    A("2. **Los términos de la sección 3.3 marcados para descartar**: aplicarlos quitaría "
      f"{len(drop)} entradas más. No están aplicados en las cifras de este documento.")
    A("3. **Umbral de entrada de los términos nuevos** (50 disparos y 5 empresas). Es un criterio práctico, no estadístico.")
    A("4. **Filtros de extracción adicionales**. Dos fuentes de ruido afectan a muchos términos a la vez y se "
      "arreglarían mejor en la extracción que retirando términos: las biografías de consejeros (cargos en otras "
      "empresas) y la prosa legal fija de los informes de auditoría (\"whether due to fraud or error\").\n")
    A("---\n")
    A("Datos completos en `metadata/ESG_terms_v3.csv` y en las auditorías `metadata/ESG_deep_consensus_v3*.csv`. "
      "Decisiones razonadas término a término en `metadata/ESG_terms_v3_decisiones.csv`.")

    out = MET.parent / "informes" / "ESG_vocabulario_v3.md"
    out.write_text(accent("\n".join(L)) + "\n", encoding="utf-8")
    print(f"escrito {out}")
    print(f"claros {len(clear)} | colocaciones {len(colloc)} | avisos {len(avisos)} | "
          f"dudas navegacion {len(nav)} | dudas significado {len(meaning)} (descartar {len(drop)})")


if __name__ == "__main__":
    main()
