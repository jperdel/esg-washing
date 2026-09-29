"""
robustness_v3.py
----------------
Cifras del apartado de robustez (paper, seccion 10) que no salen directamente
de stability_v3.py / stability_v3_e2e.py: la inferencia de la pendiente
temporal para cada perturbacion con nombre del vocabulario v3. Lee salidas ya
generadas y el texto procesado de la ejecucion adoptada:

    results/stability_v3/{summary.json, quitar_uno.csv, e2e_cache.pkl}
    results/metrics_v3_sec/results.csv, data/clean_v3_sec/processed_texts.csv

y escribe results/analysis_v3/robustness_v3.json.

    PYTHONIOENCODING=utf-8 python scripts/robustness_v3.py [--n-boot 9999] [--skip-e2e]

Solo perturbaciones del vocabulario v3 (directrices, 2) y solo medidas de
score (directrices, 5), para el ESGSI y el ESGSI_ext, contra la referencia del
mismo indice: Spearman (rho), Pearson (r), solapamiento de los deciles
extremos (top10, bot10: fraccion de los 34 informes del 10 % superior o
inferior de la referencia que siguen en el suyo) y pendiente temporal (MCO y
efectos fijos de empresa, con error agrupado por empresa y p del wild cluster
bootstrap, con las funciones de analysis_v3.py).

Bloques
  b  Texto congelado. Primero se reproduce el ESGSI adoptado desde el texto
     (dif. max < 1e-3, o se aborta). Variantes: sin 'board director' y los
     demas grandes movedores de quitar_uno.csv (menor rho), sin sectoriales,
     sin cada pilar, sin los terminos con aviso.
  c  Suelo estructural de la correlacion de Pearson del ESGSI para esas
     supresiones: con t = sd(incremento)/sd(base) el minimo alcanzable de
     corr(base, base + incremento) es sqrt(1 - t^2) (componente). Suelo del
     indice: la misma identidad con z(SEN) comun,
     corr(zS - zA, zS - zB) = (1 - a - b + c) / sqrt((2 - 2a)(2 - 2b)), con
     a, b observados y c en su suelo. Se da la posicion de lo observado en su
     ventana, (obs - suelo) / (1 - suelo).
  d  Re-extraccion simulada (modelo de stability_v3_e2e.py): las mismas
     medidas para cada supresion con nombre, en los dos modos.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import time
from pathlib import Path

os.environ["ESG_RUN_TAG"] = "v3"
os.environ["ESG_INCLUDE_SECTORAL"] = "1"

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR / "scripts"))

import analysis_v3 as A  # noqa: E402
from stability_v3 import Comparer  # noqa: E402

STAB = BASE_DIR / "results" / "stability_v3"
OUT = BASE_DIR / "results" / "analysis_v3" / "robustness_v3.json"
TOP_MOVERS = 4          # los cuatro terminos de menor rho en quitar_uno.csv (y siempre board director)
INDICES = ("ESGSI", "ESGSI_ext")


# ---------------------------------------------------------------------------
# utilidades
# ---------------------------------------------------------------------------

def floors(dens_full: np.ndarray, dens_base: np.ndarray, zsen: np.ndarray) -> dict:
    inc = dens_full - dens_base
    t = inc.std() / dens_base.std()
    comp_floor = float(np.sqrt(max(1 - t ** 2, 0.0)))
    comp_obs = float(np.corrcoef(dens_full, dens_base)[0, 1])
    a = float(np.corrcoef(zsen, dens_full)[0, 1])
    b = float(np.corrcoef(zsen, dens_base)[0, 1])

    def idx_corr(c):
        return (1 - a - b + c) / np.sqrt((2 - 2 * a) * (2 - 2 * b))

    idx_obs = float(np.corrcoef(zsen - A.z(dens_full), zsen - A.z(dens_base))[0, 1])
    idx_floor = float(idx_corr(comp_floor))
    return {"t": float(t),
            "componente": {"suelo": comp_floor, "observado": comp_obs,
                           "posicion_en_ventana": (comp_obs - comp_floor) / (1 - comp_floor)
                           if comp_floor < 1 else None},
            "indice": {"suelo": idx_floor, "observado": idx_obs,
                       "identidad_reproduce_observado": float(idx_corr(comp_obs)),
                       "posicion_en_ventana": (idx_obs - idx_floor) / (1 - idx_floor)
                       if idx_floor < 1 else None}}


def trend(d: pd.DataFrame, idx: np.ndarray, n_boot: int) -> dict:
    tb = A.trend_block(d.assign(_idx=idx), "_idx", n_boot, A.SEED)
    return {"pendiente_mco": tb["mco_convencional"]["pendiente"],
            "p_mco_convencional": tb["mco_convencional"]["p"],
            "pendiente_ef": tb["ef_cluster"]["pendiente"], "se_ef_cluster": tb["ef_cluster"]["se"],
            "p_ef_cluster": tb["ef_cluster"]["p"], "ci95_ef": tb["ef_cluster"]["ci95"],
            "p_wild": tb["ef_wild_bootstrap"]["p_wild"], "reps_wild": n_boot,
            "p_mco_cluster": tb["mco_cluster"]["p"]}


def measures(d: pd.DataFrame, idx: np.ndarray, cmp: Comparer, n_boot: int) -> dict:
    c = cmp(idx)
    return {"rho": c["rho"], "r": c["r"], "top10": c["top10"], "bot10": c["bot10"],
            **trend(d, idx, n_boot)}


# ---------------------------------------------------------------------------
# b, c  texto congelado
# ---------------------------------------------------------------------------

def block_bc(n_boot: int) -> dict:
    res_path = A.RESULTS_DIR / "metrics_v3_sec" / "results.csv"
    df = A.load_results(res_path, A.METADATA_DIR / "empresas_supersector.csv")
    corpus, dedup = A.load_corpus(A.DATA_DIR / "clean_v3_sec" / "processed_texts.csv")
    df = df.merge(corpus, on=A.KEY, how="left", validate="one_to_one")
    if df["clean_text"].isna().any():
        raise ValueError("filas de results sin clean_text")
    vocab, pillar, sectoral, _, _, _ = A.load_vocab_and_pillars(A.METADATA_DIR / "ESG_terms_v3.csv")
    texts = df["clean_text"].astype(str).tolist()
    C, cols = A.count_matrix(texts, vocab)
    col = np.array(cols)
    tokens = np.array([max(len(t.split()), 1) for t in texts], dtype=float)
    zsen = A.z(df["SEN_Score"])
    # parte del ESGSI_ext que no depende del vocabulario (ver stability_v3.py)
    ext_extra = (df["ESGSI_ext"] - df["ESGSI"]).to_numpy(float)
    dens_full = C.sum(axis=1) / tokens * 100
    ref = zsen - A.z(dens_full)

    gap = float(np.abs(ref - df["ESGSI"].to_numpy()).max())
    if gap >= 1e-3:
        raise ValueError(f"no se reproduce el ESGSI adoptado: dif. max {gap:.3g}")
    refs = {"ESGSI": ref, "ESGSI_ext": ref + ext_extra}
    cmp = {ix: Comparer(refs[ix], df["year"].to_numpy(), df["firm"].to_numpy()) for ix in INDICES}
    out = {"documentos": len(df), "deduplicacion": dedup[0], "decil_k": cmp["ESGSI"].k,
           "reproduccion_ESGSI": {"dif_abs_max": gap},
           "referencia": {ix: trend(df, refs[ix], n_boot) for ix in INDICES}}
    print(f"referencia reproducida (dif. max {gap:.2g}); pendiente EF "
          f"{out['referencia']['ESGSI']['pendiente_ef']:.4f}", flush=True)

    lo = pd.read_csv(STAB / "quitar_uno.csv", sep=";").sort_values("ESGSI_rho")
    mass = C.sum(axis=0)
    P = np.array([pillar.get(t, "?") for t in cols])
    is_sec = np.array([t in sectoral for t in cols])
    stab = json.loads((STAB / "summary.json").read_text(encoding="utf-8"))
    aviso = set(stab["terminos_con_aviso"])

    movers = list(dict.fromkeys(["board director"] + lo["termino"].head(TOP_MOVERS).tolist()))
    variants = {f"sin '{t}'": col != t for t in movers}
    variants["sin sectoriales"] = ~is_sec
    variants["sin los terminos con aviso"] = ~np.isin(col, list(aviso))
    for p in ("E", "S", "G", "TRANS"):
        variants[f"sin pilar {p}"] = P != p

    out["variantes"] = {}
    for name, keep in variants.items():
        dens = C[:, keep].sum(axis=1) / tokens * 100
        e = zsen - A.z(dens)
        idx = {"ESGSI": e, "ESGSI_ext": e + ext_extra}
        r = {"entradas_quitadas": int((~keep).sum()),
             "cuota_masa_quitada": float(1 - mass[keep].sum() / mass.sum()),
             **{ix: measures(df, idx[ix], cmp[ix], n_boot) for ix in INDICES},
             "suelo_estructural_ESGSI": floors(dens_full, dens, zsen)}
        out["variantes"][name] = r
        x = r["ESGSI"]
        print(f"{name:32s} rho {x['rho']:.4f} top {x['top10']:.3f} bot {x['bot10']:.3f} "
              f"pte EF {x['pendiente_ef']:+.4f} p_cl {x['p_ef_cluster']:.2g} p_wild {x['p_wild']:.3g}",
              flush=True)
    return out


# ---------------------------------------------------------------------------
# d  re-extraccion simulada
# ---------------------------------------------------------------------------

def block_d(n_boot: int) -> dict:
    import stability_v3_e2e as E  # usa el mismo modelo que las pruebas publicadas

    with open(E.CACHE, "rb") as fh:
        c = pickle.load(fh)
    if "quant_hits" not in c:
        raise ValueError("la cache e2e no tiene QUANT/HEDGE por parrafo: ejecuta stability_v3_e2e.py")
    m = E.Model(c)
    res = A.norm_keys(c["results"].copy())
    d = pd.DataFrame({"firm": res["Compañía"].map(A.nfc), "year": res["Año"].astype(int)})
    if d["firm"].nunique() != 49:
        raise ValueError("el modelo e2e no tiene 49 empresas")

    v3 = list(csv.DictReader(open(A.METADATA_DIR / "ESG_terms_v3.csv", encoding="utf-8-sig"), delimiter=";"))
    meta = {(r["termino"].strip() if r["tipo"] == "patron" else r["termino"].strip().lower()): r for r in v3}
    pillar = np.array([meta[e]["pilar"] for e in c["entries"]])
    sector = np.array([meta[e]["sectorial"] == "1" for e in c["entries"]])
    aviso = np.array([bool(meta[e]["aviso"]) and not meta[e]["aviso"].startswith("colocacion")
                      for e in c["entries"]])
    full = np.ones(len(c["entries"]), dtype=bool)
    variants = {"v3 completo": full, "sin sectoriales": ~sector, "sin los terminos con aviso": ~aviso}
    for p in ("E", "S", "G", "TRANS"):
        variants[f"sin pilar {p}"] = pillar != p
    variants["sin 'board director'"] = c["entry_col"] != c["sus_cols"].index("board director")
    out = {}
    for name, keep in variants.items():
        out[name] = {"entradas_quitadas": int((~keep).sum())}
        for mo, rex in E.MODES:
            idx = m.indices(keep, reextract=rex)
            out[name][mo] = {ix: measures(d, idx[ix], m.cmp[mo][ix], n_boot) for ix in INDICES}
        x = out[name]["re-extraida"]["ESGSI"]
        print(f"e2e {name:28s} re-extr. rho {x['rho']:.4f} top {x['top10']:.3f} bot {x['bot10']:.3f} "
              f"pte EF {x['pendiente_ef']:+.4f} p_cl {x['p_ef_cluster']:.2g} p_wild {x['p_wild']:.3g}",
              flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boot", type=int, default=A.N_BOOT)
    ap.add_argument("--skip-e2e", action="store_true", help="no cargar la cache de la re-extraccion")
    a = ap.parse_args()
    t0 = time.time()
    out = {"generado": time.strftime("%Y-%m-%d %H:%M:%S"), "n_boot": a.n_boot, "semilla": A.SEED,
           "medidas": "rho Spearman, r Pearson, top10/bot10 solapamiento de deciles extremos (k = 34), "
                      "pendiente MCO y de efectos fijos de empresa con SE agrupado y p wild"}
    out["b_c_texto_congelado"] = block_bc(a.n_boot)
    if not a.skip_e2e:
        out["d_re_extraccion"] = block_d(a.n_boot)
    out["segundos"] = round(time.time() - t0)
    OUT.write_text(json.dumps(A.jsonable(out), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"listo: {OUT} ({out['segundos']}s)")


if __name__ == "__main__":
    main()
