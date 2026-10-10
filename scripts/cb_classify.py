"""
cb_classify.py
--------------
Paso 2: aplica los clasificadores abiertos de ClimateBERT a los pasajes de
cb_segment.py, en CPU, con predicciones en cache por HASH DEL TEXTO del pasaje
(SHA-1, columna h de passages.parquet): un pasaje cuyo texto no cambia entre
extracciones del corpus no se vuelve a clasificar, aunque cambie su posicion.

Orden (Bingler et al., 2022, 2024): el detector de texto climatico se aplica a
todos los pasajes; compromiso, especificidad y sentimiento solo a los pasajes
climaticos (prob. 'yes' del detector > 0.5). Se guardan las probabilidades de
cada clase; la clase predicha es la de mayor probabilidad (argmax), como en el
pipeline de text-classification de los modelos.

    .venv-cb/Scripts/python.exe scripts/cb_classify.py [--models detector,commitment,...]
                                                       [--threads 8]
                                                       [--seed-from <cache del piloto>]

--seed-from importa las predicciones de una cache con el formato del piloto
(issue #2: passages.parquet con doc_key, pid, text y pred_<modelo>/part_*.parquet
indexados por doc_key, pid), convirtiendolas a la clave por hash.

Salidas (cache, no versionada):
    data/climatebert/pred_<modelo>.parquet   h, p_<clase>...  (una fila por texto distinto)
    data/climatebert/pred_<modelo>_new/part_XXXX.parquet  bloques nuevos (reanudable)
    data/climatebert/runtime.json            pasajes, reutilizados, inferidos y tiempo
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cb_common import CACHE_DIR, MODELS, setup_hf_env, text_hash

setup_hf_env()
import torch  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

SHARD = 4000
BATCH = 16  # el mas rapido en CPU (i7-9700K, 8 hilos): ~54 pasajes/s
CLIMATE_CUT = 0.5
ORDER = ["detector", "commitment", "specificity", "sentiment"]


def load_model(name: str):
    repo, rev = MODELS[name]
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    mod = AutoModelForSequenceClassification.from_pretrained(repo, revision=rev)
    mod.eval()
    labels = [mod.config.id2label[i] for i in range(mod.config.num_labels)]
    return tok, mod, labels


@torch.inference_mode()
def predict(texts: list[str], tok, mod) -> np.ndarray:
    enc_len = [len(t) for t in texts]
    order = np.argsort(enc_len, kind="stable")
    out = np.zeros((len(texts), mod.config.num_labels), dtype=np.float32)
    for i in range(0, len(texts), BATCH):
        idx = order[i:i + BATCH]
        enc = tok([texts[j] for j in idx], padding=True, truncation=True, max_length=512,
                  return_tensors="pt")
        logits = mod(**enc).logits
        out[idx] = torch.softmax(logits, dim=-1).numpy()
    return out


def cache_file(name: str) -> Path:
    return CACHE_DIR / f"pred_{name}.parquet"


def load_cache(name: str) -> pd.DataFrame:
    """Predicciones por hash: la cache consolidada mas los bloques nuevos."""
    parts = []
    if cache_file(name).exists():
        parts.append(pd.read_parquet(cache_file(name)))
    parts += [pd.read_parquet(f) for f in sorted((CACHE_DIR / f"pred_{name}_new").glob("part_*.parquet"))]
    if not parts:
        return pd.DataFrame(columns=["h"])
    return pd.concat(parts, ignore_index=True).drop_duplicates("h", keep="first")


def seed_from(old: Path):
    """Convierte la cache del piloto (clave doc_key, pid) a la clave por hash."""
    P = pd.read_parquet(old / "passages.parquet", columns=["doc_key", "pid", "text"])
    P["h"] = P["text"].map(text_hash)
    for name in ORDER:
        fs = sorted((old / f"pred_{name}").glob("part_*.parquet"))
        if not fs:
            continue
        pr = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
        pr = pr.merge(P[["doc_key", "pid", "h"]], on=["doc_key", "pid"], how="inner",
                      validate="one_to_one")
        pr = pr.drop(columns=["doc_key", "pid"]).drop_duplicates("h")
        cur = load_cache(name)
        allp = pd.concat([cur, pr], ignore_index=True).drop_duplicates("h", keep="first")
        allp.to_parquet(cache_file(name), index=False)
        print(f"{name}: {len(pr)} textos importados de la cache del piloto; cache {len(allp)}")


def run(name: str, P: pd.DataFrame, rt: dict):
    """Clasifica los textos de P (columnas h, text) que no estan en la cache."""
    cache = load_cache(name)
    todo = P.drop_duplicates("h")
    todo = todo[~todo["h"].isin(set(cache["h"]))].reset_index(drop=True)
    n_total, n_new = len(P), len(todo)
    print(f"{name}: {n_total} pasajes, {P['h'].nunique()} textos distintos, "
          f"{n_new} sin prediccion en cache", flush=True)
    t0 = time.time()
    if n_new:
        d = CACHE_DIR / f"pred_{name}_new"
        d.mkdir(parents=True, exist_ok=True)
        tok, mod, labels = load_model(name)
        start = len(list(d.glob("part_*.parquet")))
        for s in range(-(-n_new // SHARD)):
            sub = todo.iloc[s * SHARD:(s + 1) * SHARD]
            t1 = time.time()
            pr = predict(sub["text"].tolist(), tok, mod)
            res = sub[["h"]].copy()
            for j, lab in enumerate(labels):
                res[f"p_{lab}"] = pr[:, j]
            res.to_parquet(d / f"part_{start + s:04d}.parquet", index=False)
            el = time.time() - t1
            print(f"{name} bloque {s + 1}: {len(sub)} pasajes en {el:.0f}s ({len(sub) / el:.1f}/s)",
                  flush=True)
        # Consolidar.
        load_cache(name).to_parquet(cache_file(name), index=False)
        for f in d.glob("part_*.parquet"):
            f.unlink()
        d.rmdir()
    el = time.time() - t0
    rt[name] = {"pasajes": int(n_total), "textos_distintos": int(P["h"].nunique()),
                "reutilizados_de_cache": int(P["h"].nunique() - n_new),
                "inferidos_en_esta_ejecucion": int(n_new),
                "segundos_esta_ejecucion": round(el, 1),
                "pasajes_por_segundo": round(n_new / el, 1) if n_new and el else None,
                "revision": MODELS[name][1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(ORDER))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed-from", default=None)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    if a.seed_from:
        seed_from(Path(a.seed_from))
    P = pd.read_parquet(CACHE_DIR / "passages.parquet", columns=["doc_key", "pid", "h", "text"])
    rtf = CACHE_DIR / "runtime.json"
    rt = json.loads(rtf.read_text()) if rtf.exists() else {}
    for name in a.models.split(","):
        if name == "detector":
            sub = P
        else:
            det = load_cache("detector")
            clim = set(det.loc[det["p_yes"] > CLIMATE_CUT, "h"])
            sub = P[P["h"].isin(clim)]
        run(name, sub, rt)
        rtf.write_text(json.dumps(rt, indent=2))


if __name__ == "__main__":
    main()
