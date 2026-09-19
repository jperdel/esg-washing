"""
build_kwic_sample.py
--------------------
Prepara la auditoria de FALSOS POSITIVOS del vocabulario (ronda 3).

    python scripts/build_kwic_sample.py

QUE HACE

    Para cada entrada de `metadata/ESG_terms_final.csv`, busca TODAS sus
    apariciones en el corpus que el indice puntua realmente (las zonas ya
    extraidas, sin parrafos de navegacion), mide su
    masa, y extrae una muestra aleatoria de apariciones EN CONTEXTO para que
    unos evaluadores juzguen si cada una es un uso ESG genuino.

    Salidas:
      paper/_research/term_mass_sin_navegacion.csv   masa por entrada
      paper/_research/kwic/bloque_NN.txt   concordancias a evaluar, por bloques
      paper/_research/kwic/indice.csv      que termino esta en que bloque

POR QUE ESTO Y NO OTRA COSA

    Las rondas 1 y 2 midieron COBERTURA: que terminos existen y si los
    evaluadores coinciden en que son ESG *como tipo*. Ninguna midio lo
    contrario: de todas las veces que una entrada dispara sobre el corpus, que
    fraccion son aciertos. Esa es la cifra que entra en el SUS, y no la hemos
    medido nunca.

    El analisis de colocaciones existente (informes/colocaciones_analisis.md)
    tampoco es esto: clasifica por vocabulario circundante, que es un proxy, y
    el propio documento lo deja declarado como pendiente de validar.

DOS DECISIONES DE MUESTREO QUE IMPORTAN

  1. MUESTREO INDEPENDIENTE POR TERMINO, no por posicion del texto.
     Un mismo tramo ("carbon footprint") dispara `carbon` Y `footprint` si
     ambos estan en el vocabulario, porque el CountVectorizer cuenta cada
     entrada por separado sobre el texto lematizado. La auditoria tiene que
     reproducir ese modelo: cada entrada se audita por lo que ELLA dispara,
     aunque los tramos se solapen. Por eso NO se usa la alternancia ordenada
     por longitud del pipeline de extraccion, que atribuye cada tramo al
     termino mas especifico.

  2. MUESTRA UNIFORME POR TERMINO, PONDERACION POR MASA AL AGREGAR.
     Se toman K apariciones de cada entrada, repartidas entre documentos
     distintos para que un solo informe no domine la estimacion. Eso da
     precision por termino (que es lo que se pidio). La precision GLOBAL del
     vocabulario se obtiene despues reponderando por masa:

         P_global = sum_t (ocurrencias_t * p_t) / sum_t ocurrencias_t

     Es un muestreo estratificado con pesos conocidos. Estimar la precision
     global con una muestra uniforme SIN reponderar seria un error: daria el
     mismo peso a un termino de 15.000 apariciones que a uno de 12.
"""

from __future__ import annotations

import bisect
import csv
import hashlib
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
METADATA_DIR = BASE_DIR / "metadata"
RESEARCH_DIR = BASE_DIR / "paper" / "_research"
# --candidatos: en vez del vocabulario vigente, audita los huecos que la ronda 2
# aprobo por unanimidad (metadata/ESG_deep_consensus_v2.csv). La ronda 2 solo
# confirmo que son ESG como tipo; aqui se mide si disparan bien antes de darlos
# de alta.
CANDIDATES = "--candidatos" in sys.argv
# --colocaciones: audita las colocaciones nuevas de la propuesta v3
# (metadata/ESG_terms_v3.csv), que sustituyen a terminos retirados.
COLLOCATIONS = "--colocaciones" in sys.argv
KWIC_DIR = RESEARCH_DIR / ("kwic_colocaciones" if COLLOCATIONS
                           else "kwic_candidatos" if CANDIDATES else "kwic")
MIN_OCC_CANDIDATE = 50        # disparos en el corpus
MIN_COMPANIES_CANDIDATE = 5   # de 49 empresas
MIN_OCC_COLLOCATION = 20      # por debajo no mueve el indice: se lista con su masa
MASS_FILE = RESEARCH_DIR / ("term_mass_colocaciones.csv" if COLLOCATIONS
                            else "term_mass_candidatos.csv" if CANDIDATES
                            else "term_mass_sin_navegacion.csv")

sys.path.insert(0, str(BASE_DIR / "src"))
from lexical_document_filter import _term_to_pattern  # noqa: E402

SEED = 20260905
CONTEXT_CHARS = 260   # a cada lado de la aparicion
RESERVOIR = 90        # candidatos retenidos por termino antes de submuestrear
LINES_PER_BLOCK = 400

# Reparto de la muestra por tramos de masa. La masa esta muy concentrada (64
# entradas acumulan el 78,5 % de los 1,37 M de disparos), asi que una muestra
# uniforme gastaria casi todo el esfuerzo en terminos que no mueven el indice.
# Se da mas a las pesadas SIN dejar sin estimacion a las ligeras, porque el
# encargo pide precision por termino y no solo la global.
#
#   posicion en el ranking de masa  ->  apariciones a juzgar
K_TIERS = [(10, 40), (30, 20), (100, 12)]
K_TAIL = 6

SEPARATOR = "\n\x00\n"
PER_DOC = 4           # apariciones retenidas al azar por documento


def _company_key(name: str) -> str:
    """El CSV trae los acentos corruptos ('M�nich Re'); se compara sin ellos."""
    return re.sub(r"[^a-z0-9]", "", str(name).encode("ascii", "ignore").decode().lower())


def load_corpus() -> pd.DataFrame:
    """Las zonas extraidas, con la MISMA deduplicacion que main.py, y SIN los
    parrafos de navegacion (indices, tablas de correspondencia).

    Se audita el texto que se va a puntuar. El filtro de navegacion ya esta en
    el extractor (`LexicalDocumentFilter._is_navigation`), pero el corpus aun no
    se ha re-extraido; aqui se aplica sobre los parrafos de las zonas actuales.
    Es una aproximacion: al re-extraer, quitar un parrafo tambien puede cambiar
    que parrafos vecinos entran como contexto.
    """
    df = pd.read_csv(BASE_DIR / "data/clean/processed_texts.csv", sep=";",
                     encoding="utf-8")
    df.columns = ["Documento", "Pais", "Compania", "Anio", "TipoDocumento",
                  "clean_text", "raw_text"]
    seen, keep = set(), []
    for i, row in df.iterrows():
        digest = hashlib.md5(str(row["clean_text"]).encode("utf-8")).hexdigest()
        if digest not in seen:
            seen.add(digest)
            keep.append(i)
    df = df.loc[keep].reset_index(drop=True)

    from lexical_document_filter import LexicalDocumentFilter
    by_key = {(_company_key(p.parent.name), p.name): p
              for p in (BASE_DIR / "data" / "chunks_lexical").rglob("*.json")}
    texts, removed, total = [], 0, 0
    for _, row in df.iterrows():
        path = by_key[(_company_key(row["Compania"]), row["Documento"])]
        kept = []
        for zone in json.loads(path.read_text(encoding="utf-8"))["zones"]:
            for para in zone["text"].split("\n\n"):
                para = re.sub(r"\s+", " ", para).strip()
                words = len(para.split())
                total += words
                if LexicalDocumentFilter._is_navigation(para):
                    removed += words
                    continue
                kept.append(para)
        texts.append(" ".join(kept))
    df["text"] = texts
    print(f"corpus: {len(df)} documentos, {df['Compania'].nunique()} empresas, "
          f"{df['Pais'].nunique()} paises; navegacion retirada: "
          f"{removed / total * 100:.2f} % de las palabras")
    return df


def build_matchers() -> list[tuple[dict, re.Pattern]]:
    """Una regex por entrada. Los 'patron' se usan tal cual; los literales
    pasan por el MISMO constructor que usa el pipeline de extraccion, para que
    la auditoria mida lo que el pipeline realmente dispara."""
    if COLLOCATIONS:
        rows = [{"termino": r["termino"], "tipo": r["tipo"], "pilar": r["pilar"],
                 "sectorial": r["sectorial"]}
                for r in csv.DictReader(open(METADATA_DIR / "ESG_terms_v3.csv",
                                             encoding="utf-8-sig"), delimiter=";")
                if r["aviso"] == "colocacion nueva sin auditar"]
    elif CANDIDATES:
        rows = [{"termino": r["termino"], "tipo": "termino", "pilar": r["pilar_r2"],
                 "sectorial": r["sectorial_r2"]}
                for r in csv.DictReader(open(METADATA_DIR / "ESG_deep_consensus_v2.csv",
                                             encoding="utf-8-sig"), delimiter=";")
                if r["r2_aprobaciones"] == "3" and r["estado_vocabulario"] == "HUECO"]
    else:
        rows = list(csv.DictReader(
            open(METADATA_DIR / "ESG_terms_final.csv", encoding="utf-8-sig"),
            delimiter=";"))
    out = []
    for row in rows:
        term, kind = row["termino"].strip(), row["tipo"].strip()
        pattern = term if kind == "patron" else r"\b" + _term_to_pattern(term) + r"\b"
        try:
            out.append((row, re.compile(pattern, re.IGNORECASE)))
        except re.error as exc:
            print(f"  AVISO: regex invalida, entrada omitida: {term} ({exc})")
    print(f"vocabulario: {len(out)} entradas compiladas")
    return out


def scan(df: pd.DataFrame, matchers) -> tuple[list[dict], dict]:
    """Una sola pasada por entrada sobre el corpus concatenado.

    Se concatena para no pagar 361 x 343 llamadas a finditer; los limites de
    documento se recuperan por biseccion sobre los offsets.
    """
    texts = [str(t) for t in df["text"]]
    offsets, pos = [], 0
    for text in texts:
        offsets.append(pos)
        pos += len(text) + len(SEPARATOR)
    corpus = SEPARATOR.join(texts)
    print(f"corpus concatenado: {len(corpus) / 1e6:.1f} MB")

    rng = random.Random(SEED)
    mass, samples = [], {}
    start = time.time()
    for n, (row, regex) in enumerate(matchers, 1):
        by_doc = defaultdict(list)
        seen_in_doc = defaultdict(int)
        total = 0
        for m in regex.finditer(corpus):
            total += 1
            doc = bisect.bisect_right(offsets, m.start()) - 1
            # Reservorio aleatorio por documento (algoritmo R). La version
            # anterior guardaba las CUATRO PRIMERAS apariciones de cada informe,
            # que caen casi siempre en el indice inicial: la mediana de posicion
            # de la muestra quedaba en el 0,1 % del documento y la precision de
            # los terminos pesados salia hundida por un artefacto de muestreo.
            seen_in_doc[doc] += 1
            if len(by_doc[doc]) < PER_DOC:
                by_doc[doc].append((m.start(), m.end()))
            else:
                j = rng.randrange(seen_in_doc[doc])
                if j < PER_DOC:
                    by_doc[doc][j] = (m.start(), m.end())
        docs = sorted(by_doc)
        mass.append({
            "termino": row["termino"], "tipo": row["tipo"], "pilar": row["pilar"],
            "sectorial": row["sectorial"], "ocurrencias": total,
            "documentos": len(docs),
            "empresas": df.loc[docs, "Compania"].nunique() if docs else 0,
        })
        # candidatos: como mucho uno por documento en la primera vuelta, para
        # repartir; luego se completa si hacen falta mas.
        for d in docs:
            rng.shuffle(by_doc[d])
        pool = [(d, by_doc[d][0]) for d in docs]
        rng.shuffle(pool)
        extra = [(d, s) for d in docs for s in by_doc[d][1:]]
        rng.shuffle(extra)
        samples[row["termino"]] = (pool + extra)[:RESERVOIR]
        if n % 40 == 0:
            print(f"  {n}/{len(matchers)} entradas  ({time.time() - start:.0f}s)")
    print(f"escaneo completo en {time.time() - start:.0f}s")
    return mass, samples


def kwic_line(corpus_text: str, span: tuple[int, int],
              doc_start: int, doc_end: int) -> str:
    """Aparicion con contexto, en una sola linea, con el disparo entre << >>.

    El contexto se recorta a los limites del documento. Sin ese recorte, una
    aparicion cerca del principio o del final de un informe arrastraba texto
    del informe contiguo (75 de 3.180 concordancias en la primera version).
    """
    a, b = span
    left = corpus_text[max(doc_start, a - CONTEXT_CHARS):a]
    right = corpus_text[b:min(doc_end, b + CONTEXT_CHARS)]
    hit = corpus_text[a:b]
    # recortar por limite de palabra para no cortar a mitad
    if " " in left:
        left = left[left.index(" ") + 1:]
    if " " in right:
        right = right[:right.rindex(" ")]
    text = f"{left}<<{hit}>>{right}"
    return re.sub(r"\s+", " ", text).strip()


def k_for_rank(rank: int) -> int:
    """Cuantas apariciones se juzgan de la entrada que ocupa esa posicion."""
    if CANDIDATES or COLLOCATIONS:
        return K_TAIL
    for limit, k in K_TIERS:
        if rank <= limit:
            return k
    return K_TAIL


def main() -> None:
    df = load_corpus()
    KWIC_DIR.mkdir(parents=True, exist_ok=True)
    cache = KWIC_DIR / "_reservoir.json"

    # El escaneo cuesta ~18 min sobre 200 MB de corpus. Se cachea el reservorio
    # para poder recomponer la muestra con otro reparto sin volver a escanear.
    if cache.exists() and "--rescan" not in sys.argv:
        import json
        blob = json.loads(cache.read_text(encoding="utf-8"))
        mass = blob["mass"]
        samples = {t: [(d, (a, b)) for d, a, b in v]
                   for t, v in blob["samples"].items()}
        print(f"reservorio cacheado: {len(mass)} entradas "
              f"(usa --rescan para recalcular)")
    else:
        matchers = build_matchers()
        mass, samples = scan(df, matchers)
        mass.sort(key=lambda d: -d["ocurrencias"])
        import json
        cache.write_text(json.dumps({
            "mass": mass,
            "samples": {t: [[d, a, b] for d, (a, b) in v]
                        for t, v in samples.items()},
        }), encoding="utf-8")
        print(f"reservorio guardado en {cache}")

    with open(MASS_FILE, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(mass[0]), delimiter=";")
        writer.writeheader()
        writer.writerows(mass)

    total = sum(d["ocurrencias"] for d in mass)
    dead = [d["termino"] for d in mass if d["ocurrencias"] == 0]
    print(f"\ndisparos totales: {total:,}")
    print(f"entradas que NUNCA disparan: {len(dead)}")
    if dead:
        print("   " + ", ".join(dead))
    for k in (10, 25, 50, 100):
        share = sum(d["ocurrencias"] for d in mass[:k]) / total
        print(f"  top {k:3d} entradas = {share * 100:5.1f} % de la masa")

    # --- muestra a evaluar ------------------------------------------------
    texts = [str(t) for t in df["text"]]
    offsets, pos = [], 0
    for text in texts:
        offsets.append(pos)
        pos += len(text) + len(SEPARATOR)
    corpus = SEPARATOR.join(texts)

    for old in KWIC_DIR.glob("bloque_*.txt"):
        old.unlink()

    index, blocks, current, n_lines = [], [], [], 0
    for rank, entry in enumerate(mass, 1):
        term = entry["termino"]
        if entry["ocurrencias"] == 0:
            continue
        # Un candidato solo se audita si tiene masa y reparto minimos: por debajo
        # no mueve el indice y suele ser vocabulario de una sola empresa.
        if COLLOCATIONS and entry["ocurrencias"] < MIN_OCC_COLLOCATION:
            continue
        if CANDIDATES and (entry["ocurrencias"] < MIN_OCC_CANDIDATE
                           or entry["empresas"] < MIN_COMPANIES_CANDIDATE):
            continue
        chosen = samples[term][:k_for_rank(rank)]
        lines = []
        for doc, span in chosen:
            lines.append((df.loc[doc, "Compania"], df.loc[doc, "Anio"],
                          kwic_line(corpus, span, offsets[doc],
                                    offsets[doc] + len(texts[doc]))))
        current.append((entry, lines))
        n_lines += len(lines)
        # Los bloques se cierran por numero de CONCORDANCIAS, no de terminos:
        # las entradas pesadas traen 40 lineas y las ligeras 6, asi que cerrar
        # por terminos dejaba el primer bloque seis veces mas grande que el
        # ultimo.
        if sum(len(x[1]) for x in current) >= LINES_PER_BLOCK:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)

    for i, block in enumerate(blocks, 1):
        n_block = sum(len(x[1]) for x in block)
        out = [
            f"# BLOQUE {i} de {len(blocks)} - auditoria de falsos positivos",
            f"# {len(block)} terminos, {n_block} concordancias.",
            "#",
            "# Cada aparicion viene con su contexto real en el corpus. El tramo que",
            "# ha disparado va entre <<...>>. Juzga si ESE uso concreto es una",
            "# mencion ESG genuina, NO si el termino es ESG en abstracto.",
            "#",
            "# Para cada linea, veredicto y categoria:",
            "#   1 ok        mencion ESG real: divulgacion, objetivo, dato, politica",
            "#   0 indice    indice de contenidos, encabezado, pie, nombre de seccion,",
            "#               referencia cruzada, entrada de tabla de navegacion",
            "#   0 financiero homografo contable, financiero o juridico",
            "#   0 generico  prosa de negocio sin contenido ESG",
            "#   0 otro      falso positivo que no encaja arriba",
            "#",
        ]
        for entry, lines in block:
            out.append("")
            out.append("=" * 78)
            out.append(f"TERMINO: {entry['termino']}")
            out.append(f"  pilar={entry['pilar']}  sectorial={entry['sectorial']}  "
                       f"tipo={entry['tipo']}")
            out.append(f"  masa: {entry['ocurrencias']} apariciones en "
                       f"{entry['documentos']} informes de {entry['empresas']} empresas")
            out.append("=" * 78)
            for j, (comp, year, text) in enumerate(lines, 1):
                out.append(f"[{j}] ({comp} {year}) {text}")
            index.append({"bloque": i, "termino": entry["termino"],
                          "n_muestras": len(lines),
                          "ocurrencias": entry["ocurrencias"]})
        (KWIC_DIR / f"bloque_{i:02d}.txt").write_text("\n".join(out) + "\n",
                                                      encoding="utf-8")

    with open(KWIC_DIR / "indice.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["bloque", "termino", "n_muestras",
                                                "ocurrencias"], delimiter=";")
        writer.writeheader()
        writer.writerows(index)

    print(f"\n{len(blocks)} bloques escritos en {KWIC_DIR}")
    print(f"{len(index)} entradas auditables, {n_lines} concordancias a juzgar")
    covered = sum(d["ocurrencias"] for d in mass if d["ocurrencias"] > 0)
    print(f"la muestra cubre entradas que suman el "
          f"{covered / total * 100:.1f} % de la masa del corpus")


if __name__ == "__main__":
    main()
