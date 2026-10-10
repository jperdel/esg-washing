"""
esgm_classify.py
----------------
Paso 2 del issue #5: aplica los modelos de esgm_common.MODELS en CPU, con
predicciones en cache por hash del texto (como cb_classify.py).

Conjuntos:
    tone              frases de la muestra comun (FinBERT-tone se entreno con
                      frases de informes de analistas)
    esg4, esg9, env,  pasajes de la muestra comun
    soc, gov
    action            pasajes de la muestra comun que EnvironmentalBERT-
                      environmental marca como ambientales (p > 0.5): la ficha
                      del modelo recomienda filtrar antes por tema ambiental
    netzero           TODOS los pasajes climaticos del #4 (detector de
                      ClimateBERT, p > 0.5): la ficha recomienda ese filtro
                      para parrafos; los objetivos son raros y la muestra
                      comun daria cuotas por informe muy ruidosas
    --set recall      pasajes del experimento de recall (esgm_recall_extract.py)
                      con env, soc, gov y esg4

    .venv-cb/Scripts/python.exe scripts/esgm_classify.py --models tone,esg4,env,soc,gov,action,netzero,esg9
    .venv-cb/Scripts/python.exe scripts/esgm_classify.py --set recall --models env,soc,gov,esg4
    .venv-cb/Scripts/python.exe scripts/esgm_classify.py --selftest

Salidas (cache): data/esg_models/pred_<modelo>.parquet; runtime.json.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from esgm_common import CACHE, CB_CACHE, CUT, MODELS, label_col, load_cache, load_model, predict, run_model

EXAMPLES = {  # ejemplos de las fichas de los modelos (comprobacion de etiquetas)
    "tone": ["there is a shortage of capital, and we need extra financing",
             "growth is strong and we have plenty of liquidity", "there are doubts about our finances",
             "profits are flat"],
    "esg4": ["Rhonda has been volunteering for several years for a variety of charitable community programs."],
    "esg9": ["For 2002, our total net emissions were approximately 60 million metric tons of CO2 equivalents "
             "for all businesses and operations we have financial interests in, based on its equity share in "
             "those businesses and operations."],
    "env": ["Scope 1 emissions are reported here on a like-for-like basis against the 2013 baseline and "
            "exclude emissions from additional vehicles used during repairs."],
    "soc": ["We follow rigorous supplier checks to prevent slavery and ensure workers' rights."],
    "gov": ["An ethical code has been issued to all Group employees."],
    "action": ["We are actively working to reduce our CO2 emissions by planting trees in 25 countries."],
    "netzero": ["We aim to reach net zero emissions across our value chain by 2040.",
                "We target a 50% reduction in absolute scope 1 and 2 emissions by 2030 against 2019.",
                "The Board met eleven times during the year."],
}


def selftest():
    out = {}
    for name, ex in EXAMPLES.items():
        tok, mod, labels = load_model(name)
        pr = predict(ex, tok, mod)
        out[name] = [{"texto": t[:60], **{label_col(l): round(float(p), 3) for l, p in zip(labels, row)}}
                     for t, row in zip(ex, pr)]
        print(name, json.dumps(out[name], ensure_ascii=False))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="tone,esg4,env,soc,gov,action,netzero,esg9")
    ap.add_argument("--set", default="sample", choices=["sample", "recall"])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    CACHE.mkdir(parents=True, exist_ok=True)
    rtf = CACHE / "runtime.json"

    def save(key, val):  # relee antes de escribir: puede haber otro proceso en marcha
        rt = json.loads(rtf.read_text()) if rtf.exists() else {}
        rt[key] = val
        rtf.write_text(json.dumps(rt, indent=2, ensure_ascii=False))

    if a.selftest:
        save("selftest", selftest())
        return
    if a.set == "recall":
        R = pd.read_json(CACHE / "recall_passages.jsonl.gz", lines=True, dtype={"h": str, "text": str})[["h", "text"]]
        for name in a.models.split(","):
            save(f"recall_{name}", run_model(name, R, tag="|recall"))
        return
    S = pd.read_parquet(CACHE / "sample.parquet", columns=["doc_key", "h", "text"])
    for name in a.models.split(","):
        assert name in MODELS, name
        if name == "tone":
            F = pd.read_parquet(CACHE / "sentences.parquet", columns=["hs", "text"]).rename(columns={"hs": "h"})
            T = F
        elif name == "action":
            env = load_cache("env")
            T = S[S["h"].isin(set(env.loc[env["p_environmental"] > CUT, "h"]))]
        elif name == "netzero":
            P = pd.read_parquet(CB_CACHE / "passages.parquet", columns=["h", "text"])
            det = pd.read_parquet(CB_CACHE / "pred_detector.parquet")
            T = P[P["h"].isin(set(det.loc[det["p_yes"] > CUT, "h"]))]
        else:
            T = S
        save(name, run_model(name, T[["h", "text"]]))


if __name__ == "__main__":
    main()
