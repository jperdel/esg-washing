"""
unback.py
---------
UNBACK: compromisos ESG no respaldados (issue #7). Cuarto componente del
ESGSI extendido; sustituye a HEDGE.

Unidad: la frase, dentro de cada parrafo del texto extraido (bloques separados
por linea en blanco en las zonas del JSON de extraccion). Una frase es un
COMPROMISO ESG si
    * nombra a la empresa (we, our, the Group, the Company, the Bank, the firm);
    * contiene una senal de compromiso o aspiracion: commit to, our
      commitment, aim to, our target/goal/ambition/objective, pledge,
      plan/intend/seek/strive to, on track to, net zero, carbon/climate
      neutral, o will + un verbo de accion;
    * contiene al menos una entrada del vocabulario ESG (la misma expresion
      regular que usa la extraccion);
    * no es condicional (if, could, might, may que no es el mes), ni una
      remision interna, ni un aviso sobre forward-looking statements, ni un
      contexto financiero (retribucion, auditoria, apetito de riesgo,
      dividendos...).

Respaldo de un compromiso, en la frase o en la anterior o siguiente del mismo
parrafo (los elementos de un objetivo verificable de ESRS 2 MDR-T par. 80 y
ESRS E1-4 par. 34):
    Q  cantidad: los patrones de QUANT, dinero, halve/double;
    D  plazo: by/until/before/no later than/through + un ano posterior al del
       informe, o "<ano> target/goal";
    B  ano base: compared to/vs/from/relative to + un ano no posterior al del
       informe, baseline, base year.

Cada compromiso aporta (3 - Q - D - B) / 3. UNBACK de un informe es la media
de sus compromisos, suavizada hacia la media del corpus con PRIOR_N
pseudo-compromisos, de modo que un informe con pocos compromisos no toma un
valor extremo y uno sin compromisos toma la media del corpus:

    UNBACK_d = (S_d + PRIOR_N * m) / (n_d + PRIOR_N),

con S_d la suma de los compromisos del informe, n_d su numero y m la media de
todos los compromisos del corpus.
"""

from __future__ import annotations

import re

import numpy as np

import config
import lexical_document_filter as ldf
from esgsi_analyzer import MAY_MONTH_RE, _is_magnitude

PRIOR_N = 5

I = re.IGNORECASE
SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(])")
COMPANY_RE = re.compile(r"\b(?:we|our|us|the group|the company|the bank|the firm)\b", I)
ACTION = (r"reduce|cut|lower|decrease|achieve|reach|become|invest|phase\s+out|eliminate|increase|double|"
          r"halve|source|switch|transition|deliver|ensure|implement|launch|expand|align|electrify|install|"
          r"cover|train|hire|recruit|replace|convert|stop|end|remove|offset|compensate|neutrali[sz]e")
COMMIT_RE = re.compile(
    r"\b(?:commit(?:s|ted)?\s+to|commitments?\s+to\b|(?:our|its|the\s+group'?s?)\s+(?:[\w-]+\s+){0,2}?commitments?\b|"
    r"aim(?:s|ing)?\s+(?:to|for)\b|"
    r"(?:our|its|the|this|these|new|an?)\s+(?:[\w-]+\s+){0,3}?(?:aims?|goals?|targets?|ambitions?|objectives?)\b|"
    r"target(?:s|ed|ing)?\s+(?:to|of|is|a|an)\b|"
    r"pledge[sd]?\b|"
    r"(?:plans?|planned|intends?|intended|seeks?|strives?|endeavou?rs?)\s+to\b|"
    r"on\s+track\s+to\b|"
    r"net[- ]zero\b|(?:carbon|climate)[- ]neutral(?:ity)?\b|"
    r"will\s+(?:continue\s+to\s+|also\s+|further\s+)?(?:" + ACTION + r")\w*\b)", I)
EXCLUDE_RE = re.compile(
    r"forward[- ]looking\s+statements?|\bif\b|\bcould\b|\bmight\b|"
    r"will\s+be\s+(?:discussed|presented|described|detailed|explained|published|reported|disclosed|submitted)|"
    r"\bsee\s+(?:section|page|chapter|note)|"
    r"remuneration|\bbonus|variable\s+(?:pay|compensation)|target\s+achievement|"
    r"assurance\s+(?:engagement|conclusion)|risk\s+appetite|market\s+share|cash\s+flows?|"
    r"terminal\s+value|dividends?\b|earnings\s+per\s+share|\bebit", I)
MAY_RE = re.compile(r"\bmay\b", I)

QPAT = {k: re.compile(v, I) for k, v in config.QUANT_PATTERNS.items()}
MONEY_RE = re.compile(r"(?:€|\$|£|\beur\b|\busd\b)\s?\d|\d[\d,.]*\s?(?:million|billion|bn|mn)\b", I)
HALVE_RE = re.compile(r"\b(?:halve|halving|by\s+half|double|doubling|triple|tripling)\b", I)
YEAR = r"((?:19|20)\d{2})"
DEADLINE_RE = re.compile(r"\b(?:by|until|till|before|no\s+later\s+than|through|towards?)\s+"
                         r"(?:the\s+end\s+of\s+|end[- ]|year[- ]end\s+)?(?:fy\s?|financial\s+year\s+)?" + YEAR + r"\b", I)
YEAR_TARGET_RE = re.compile(r"\b" + YEAR + r"\s+(?:targets?|goals?|ambitions?|objectives?|horizon|roadmap)\b", I)
BASE_RE = re.compile(r"(?:compared\s+(?:to|with)|\bvs\.?|\bversus|relative\s+to|\bfrom|\bagainst|"
                     r"\bbelow|\bover)\s+(?:the\s+)?(?:fy\s?|financial\s+year\s+)?" + YEAR + r"\b", I)
BASEWORD_RE = re.compile(r"\bbase(?:line)?[- ]?years?\b|\bbaseline\b", I)


def sentences(par: str) -> list[str]:
    par = re.sub(r"\s+", " ", par).strip()
    return [s.strip() for s in SENT_RE.split(par) if s.strip()]


def is_commitment(s: str) -> bool:
    if not COMPANY_RE.search(s) or not COMMIT_RE.search(s) or not ldf._ESG_KW_RE.search(s):
        return False
    t = MAY_MONTH_RE.sub(" ", s)
    return not (EXCLUDE_RE.search(t) or MAY_RE.search(t))


def has_q(w: str) -> bool:
    for k, p in QPAT.items():
        for m in p.finditer(w):
            if k != "large_numbers" or _is_magnitude(w, m):
                return True
    return bool(MONEY_RE.search(w) or HALVE_RE.search(w))


def has_d(w: str, year: int) -> bool:
    return any(int(m.group(1)) > year for r in (DEADLINE_RE, YEAR_TARGET_RE) for m in r.finditer(w))


def has_b(w: str, year: int) -> bool:
    return bool(BASEWORD_RE.search(w)) or any(int(m.group(1)) <= year for m in BASE_RE.finditer(w))


def commitments(par: str, year: int):
    """(frase, ventana, Q, D, B) de cada compromiso ESG de un parrafo."""
    ss = sentences(par)
    for i, s in enumerate(ss):
        if is_commitment(s):
            w = " ".join(ss[max(0, i - 1): i + 2])
            yield s, w, has_q(w), has_d(w, year), has_b(w, year)


def paragraph_counts(par: str, year: int) -> tuple[int, float]:
    """Numero de compromisos de un parrafo y suma de (3 - Q - D - B) / 3."""
    n, s = 0, 0.0
    for _, _, q, d, b in commitments(par, year):
        n += 1
        s += (3 - q - d - b) / 3
    return n, s


def document_counts(zone_texts: list[str], year: int) -> tuple[int, float]:
    """Compromisos y suma de un informe, sobre los parrafos (bloques separados
    por linea en blanco) de sus zonas extraidas."""
    n, s = 0, 0.0
    for z in zone_texts:
        for par in z.split("\n\n"):
            pn, ps = paragraph_counts(par, year)
            n += pn
            s += ps
    return n, s


def scores(n: np.ndarray, s: np.ndarray, prior_n: float = PRIOR_N) -> np.ndarray:
    """UNBACK por informe: media suavizada hacia la media del corpus."""
    n = np.asarray(n, dtype=float)
    s = np.asarray(s, dtype=float)
    m = s.sum() / max(n.sum(), 1)
    return (s + prior_n * m) / (n + prior_n)
