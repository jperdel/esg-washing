"""
ctrv_common.py
--------------
Utilidades compartidas por los scripts ctrv_*.py del issue #2 (panel gratuito
de controversias ESG para H2).

- Rutas: todo lo descargado y todo lo que contiene nombres de empresa vive en
  data/issue2_controversies/ (no versionado). En results/issue2_controversies/
  solo se escriben salidas con identificadores anonimos <Industria>-NN.
- Descarga educada con cache en disco (pausa fija entre peticiones, agente
  identificado, sin credenciales; si la fuente devuelve un reto anti-bot se
  registra como no accesible y no se intenta sortear).
- Emparejamiento de nombres de empresa con expresiones regulares definidas en
  el fichero interno firm_terms_internal.csv (firm_id;gdelt;regex).

Esquema comun de eventos (interno, un fichero por fuente):
    data/issue2_controversies/events_<fuente>_internal.csv
    firm_id;date;year;source;category;severity;title;url
category en {E, S, G, GW} (GW = alegacion de greenwashing / comunicacion
enganosa, que tambien cuenta como E). severity es especifica de cada fuente
(documentada en el README).
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "issue2_controversies"
RESULTS = ROOT / "results" / "issue2_controversies"
UA = "Mozilla/5.0 (compatible; esg-research-academic/1.0; non-commercial; low-rate)"
MAIN_CLONE = ROOT
# results/ y data/ no estan en el worktree: se leen (solo lectura) del clon principal
for cand in [ROOT, *ROOT.parents]:
    if (cand / "results" / "analysis_v3" / "doc_level_internal.csv").exists():
        MAIN_CLONE = cand
        break

EVENT_COLS = ["firm_id", "date", "year", "source", "category", "severity", "title", "url"]


def load_terms() -> pd.DataFrame:
    t = pd.read_csv(DATA / "firm_terms_internal.csv", sep=";", dtype=str)
    t["rx"] = [re.compile(r, re.I) for r in t["regex"]]
    return t


def match_firms(text: str, terms: pd.DataFrame) -> list[str]:
    return [fid for fid, rx in zip(terms["firm_id"], terms["rx"]) if rx.search(text or "")]


class Fetcher:
    """GET con cache en disco y pausa minima entre peticiones."""

    def __init__(self, cache_dir: Path, pause: float = 2.0):
        self.cache = cache_dir
        self.cache.mkdir(parents=True, exist_ok=True)
        self.pause = pause
        self._last = 0.0

    def get(self, url: str, binary: bool = False, refresh: bool = False):
        key = hashlib.sha1(url.encode()).hexdigest()
        f = self.cache / key
        if f.exists() and not refresh:
            b = f.read_bytes()
            return b if binary else b.decode("utf-8", "replace")
        dt = time.time() - self._last
        if dt < self.pause:
            time.sleep(self.pause - dt)
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                b = r.read()
        except urllib.error.URLError as e:
            # almacen de certificados de este Python desactualizado: se usa el
            # curl del sistema (mismo agente, misma URL; no sortea nada)
            if "CERTIFICATE_VERIFY_FAILED" not in str(e):
                raise
            p = subprocess.run(["curl", "-sfL", "-A", UA, url], capture_output=True, timeout=120)
            if p.returncode != 0:
                raise urllib.error.URLError(f"curl {p.returncode} en {url}") from e
            b = p.stdout
        finally:
            self._last = time.time()
        txt = b.decode("utf-8", "replace")
        if "challenges.cloudflare.com" in txt[:5000] or "<title>Just a moment" in txt[:2000]:
            raise PermissionError(f"reto anti-bot en {url}: fuente no accesible de forma automatizada")
        f.write_bytes(b)
        return b if binary else txt


def html_text(html: str) -> str:
    html = re.sub(r"(?is)<script.*?</script>|<style.*?</style>", " ", html)
    x = re.sub(r"<[^>]+>", "\n", html)
    x = re.sub(r"&nbsp;|&#160;", " ", x)
    x = re.sub(r"&amp;", "&", x)
    x = re.sub(r"[ \t\r\f\v]+", " ", x)
    return re.sub(r"\n\s*\n+", "\n", x)


def save_events(df: pd.DataFrame, source: str) -> Path:
    df = df.reindex(columns=EVENT_COLS)
    out = DATA / f"events_{source}_internal.csv"
    df.to_csv(out, sep=";", index=False)
    return out


def firm_panel() -> pd.DataFrame:
    """Panel de scores (343 informes) con firm_id anonimo, del clon principal."""
    d = pd.read_csv(MAIN_CLONE / "results" / "analysis_v3" / "doc_level_internal.csv", sep=";")
    d = d.rename(columns={"firm_label": "firm_id"})
    return d
