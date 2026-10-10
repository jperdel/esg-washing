"""
esgm_analysis.py
----------------
Paso 3 del issue #5: agrega las predicciones de FinBERT-tone, FinBERT-ESG,
ESGBERT y ClimateBERT-NetZero por informe y las compara con los componentes
del paper (solo scores continuos; ni umbrales ni etiquetas sobre informes).

Experimentos
    1  Tono en todo el texto: tono neto de FinBERT (positivas - negativas, por
       frase) frente a Z(SEN); agregado, intra y entre empresas; nivel de pasaje
       (polaridad L&M por pasaje frente al tono FinBERT, dentro de cada
       informe); por pilar (pilar del pasaje segun FinBERT-ESG o segun nuestro
       vocabulario); ruptura de 2024 con EF de empresa, con y sin los pasajes
       dominados por la lista A de palabras negativas de temas ESRS (#4).
    2  Pilares: cuotas E/S/G de FinBERT-ESG y ESGBERT frente a nuestras cuotas
       y densidades por pilar; tendencias.
    3  Recall de la extraccion (esgm_recall_extract.py): cuota del texto que los
       detectores clasifican como ESG que queda fuera de la extraccion.
    4  Sustancia concreta: cuota de pasajes de accion (EnvironmentalBERT-action,
       en pasajes ambientales) y de objetivos (NetZero: net zero o reduccion,
       en pasajes climaticos) frente a Z(SUS) y Z(QUANT); tendencias.

Inferencia: correlaciones con IC 95 % por bootstrap de empresas (B = 2000,
cb_analysis.boot_corr); saltos de 2024 y pendientes con EF de empresa y error
agrupado por empresa (cb_analysis.break_block, analysis_v3.trend_block).

    .venv-cb/Scripts/python.exe scripts/esgm_analysis.py

Salidas versionables (solo firm_label anonimo) en results/esg_models/.
"""

from __future__ import annotations

import json
import re
import time

import numpy as np
import pandas as pd
from scipy import stats

from esgm_common import AV3, CACHE, CB_CACHE, CUT, MODELS, OUT, load_scores
import cb_analysis as cba  # noqa: E402  (AN, ESRS_A, boot_corr, break_block; fija semilla y B)
from cb_analysis import AN, ESRS_A, boot_corr, break_block, sen_word_counts, zpop  # noqa: E402

av = cba.av
B_CI = cba.B
SEED = 20261013
MIN_PASS = 10  # pasajes minimos de un pilar en un informe para medir su tono

PILLARS = ["E", "S", "G"]


# ---------------------------------------------------------------------------
# Vocabulario por pilar sobre texto crudo (aproximacion: el SUS del paper
# cuenta lemas; aqui se buscan las formas naturales de los terminos, con plural
# regular y guion/espacio intercambiables, como en LexicalDocumentFilter)
# ---------------------------------------------------------------------------

def vocab_regex():
    M = pd.read_csv(AV3 / "term_lemma_pillar_map.csv", sep=";")
    M = M[M["en_vocabulario"]]
    norm = {re.sub(r"[\s\-]+", " ", t.strip().lower()): p for t, p in zip(M["termino"], M["pilar"])}
    terms = sorted(norm, key=lambda t: (-len(t.split()), -len(t)))
    pat = "|".join(r"[\s\-]+".join(re.escape(x) for x in t.split()) + "s?" for t in terms)
    return re.compile(r"\b(?:" + pat + r")\b", re.IGNORECASE), norm


VOC_RE, VOC_PILLAR = vocab_regex()


def vocab_hits(text: str) -> dict:
    out = {"E": 0, "S": 0, "G": 0, "TRANS": 0}
    for m in VOC_RE.findall(text):
        t = re.sub(r"[\s\-]+", " ", m.lower())
        p = VOC_PILLAR.get(t) or VOC_PILLAR.get(t[:-1])
        if p:
            out[p] += 1
    return out


# ---------------------------------------------------------------------------
# Pasajes de la muestra comun
# ---------------------------------------------------------------------------

def pred(name: str) -> pd.DataFrame:
    return pd.read_parquet(CACHE / f"pred_{name}.parquet")


def argmax_label(df: pd.DataFrame, cols: list[str], names: list[str]) -> np.ndarray:
    return np.array(names)[df[cols].to_numpy().argmax(axis=1)]


def passage_frame(scores: pd.DataFrame) -> pd.DataFrame:
    S = pd.read_parquet(CACHE / "sample.parquet")
    e4 = pred("esg4")
    e4["f_lab"] = argmax_label(e4, ["p_none", "p_environmental", "p_social", "p_governance"],
                               ["None", "E", "S", "G"])
    S = S.merge(e4.rename(columns={"p_none": "f_none", "p_environmental": "f_E", "p_social": "f_S",
                                   "p_governance": "f_G"}), on="h", how="left", validate="many_to_one")
    e9 = pred("esg9")
    c9 = [c for c in e9.columns if c != "h"]
    e9["f9_lab"] = argmax_label(e9, c9, [c[2:] for c in c9])
    S = S.merge(e9[["h", "f9_lab"]], on="h", how="left", validate="many_to_one")
    for name, col, new in [("env", "p_environmental", "b_E"), ("soc", "p_social", "b_S"),
                           ("gov", "p_governance", "b_G")]:
        S = S.merge(pred(name)[["h", col]].rename(columns={col: new}), on="h", how="left",
                    validate="many_to_one")
    S = S.merge(pred("action")[["h", "p_action"]], on="h", how="left", validate="many_to_one")
    det = pd.read_parquet(CB_CACHE / "pred_detector.parquet")[["h", "p_yes"]].rename(columns={"p_yes": "p_clim"})
    S = S.merge(det, on="h", how="left", validate="many_to_one")
    assert S[["f_E", "f9_lab", "b_E", "b_S", "b_G", "p_clim"]].notna().all().all()
    assert S.loc[S["b_E"] > CUT, "p_action"].notna().all()
    # Tono FinBERT por frase -> pasaje.
    F = pd.read_parquet(CACHE / "sentences.parquet", columns=["h", "sid", "hs"])
    tn = pred("tone")
    F = F.merge(tn, left_on="hs", right_on="h", how="left", suffixes=("", "_t"), validate="many_to_one")
    assert F["p_positive"].notna().all()
    lab = F[["p_neutral", "p_positive", "p_negative"]].to_numpy().argmax(axis=1)
    F["pos"], F["neg"] = (lab == 1).astype(int), (lab == 2).astype(int)
    F["pnet"] = F["p_positive"] - F["p_negative"]
    T = F.groupby("h").agg(n_sent=("sid", "size"), t_pos=("pos", "sum"), t_neg=("neg", "sum"),
                           t_pnet=("pnet", "sum")).reset_index()
    S = S.merge(T, on="h", how="left", validate="many_to_one")
    S["ptone"] = (S["t_pos"] - S["t_neg"]) / S["n_sent"]   # tono del pasaje (frases)
    S["ptone_prob"] = S["t_pnet"] / S["n_sent"]
    # Diccionario L&M por pasaje (misma logica que ESGSIAnalyzer.sen_counts).
    lm = []
    for t in S["text"]:
        pos, neg, ntok = sen_word_counts(t)
        nA = sum(v for w, v in neg.items() if w in ESRS_A)
        vh = vocab_hits(t)
        lm.append((sum(pos.values()), sum(neg.values()), nA, ntok, AN.quant_hits(t) / max(len(t.split()), 1),
                   vh["E"], vh["S"], vh["G"], vh["TRANS"]))
    L = pd.DataFrame(lm, columns=["lm_pos", "lm_neg", "lm_negA", "ntok", "quant", "v_E", "v_S", "v_G", "v_T"],
                     index=S.index)
    S = pd.concat([S, L], axis=1)
    S["lm_pol"] = np.where(S["lm_pos"] + S["lm_neg"] > 0,
                           (S["lm_pos"] - S["lm_neg"]) / np.maximum(S["lm_pos"] + S["lm_neg"], 1), 0.0)
    S["A_any"] = S["lm_negA"] > 0
    S["A_dom"] = (S["lm_negA"] > 0) & (S["lm_negA"] >= 0.5 * S["lm_neg"])
    # Pilar por vocabulario: el de mas coincidencias (E incluye TRANS); empate o 0 -> sin pilar.
    V = np.column_stack([S["v_E"] + S["v_T"], S["v_S"], S["v_G"]])
    mx = V.max(axis=1)
    uniq = (V == mx[:, None]).sum(axis=1) == 1
    S["v_lab"] = np.where((mx > 0) & uniq, np.array(PILLARS)[V.argmax(axis=1)], "none")
    S["clim"] = S["p_clim"] > CUT
    S = S.merge(scores[["doc_key", "firm_label", "year"]], on="doc_key", how="left", validate="many_to_one")
    return S


def tone_of(g: pd.DataFrame) -> float:
    n = g["n_sent"].sum()
    return (g["t_pos"].sum() - g["t_neg"].sum()) / n if n else np.nan


def sen_of(g: pd.DataFrame, drop_A_words: bool = False) -> float:
    p = g["lm_pos"].sum()
    n = g["lm_neg"].sum() - (g["lm_negA"].sum() if drop_A_words else 0)
    return (p - n) / (p + n) if p + n else 0.0


def ratio_se(num: np.ndarray, den: np.ndarray, N: int) -> float:
    """Error tipico (linealizado, con correccion por poblacion finita) de
    sum(num)/sum(den) en un muestreo aleatorio simple de n de N pasajes."""
    n = len(num)
    if n < 2 or den.sum() == 0:
        return np.nan
    R = num.sum() / den.sum()
    d = num - R * den
    f = n / N
    return float(np.sqrt((1 - f) * d.var(ddof=1) / n) / den.mean())


def doc_frame(S: pd.DataFrame, scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, se_rows = [], []
    for k, g in S.groupby("doc_key"):
        n = len(g)
        N = int(g["N_report"].iloc[0])
        r = {"doc_key": k, "n_sample": n, "N_passages": N, "fin_tone": tone_of(g),
             "fin_tone_prob": g["t_pnet"].sum() / g["n_sent"].sum(),
             "fin_pos": g["t_pos"].sum() / g["n_sent"].sum(), "fin_neg": g["t_neg"].sum() / g["n_sent"].sum(),
             "SEN_s": sen_of(g), "SEN_s_exAw": sen_of(g, drop_A_words=True),
             "A_dom_share": g["A_dom"].mean(), "A_any_share": g["A_any"].mean()}
        for nm, col in [("exAdom", "A_dom"), ("exAany", "A_any")]:
            gg = g[~g[col]]
            r[f"fin_tone_{nm}"] = tone_of(gg)
            r[f"SEN_s_{nm}"] = sen_of(gg)
        # Tono por pilar (FinBERT-ESG y vocabulario).
        for src, col in [("f", "f_lab"), ("v", "v_lab")]:
            for p in PILLARS + (["None"] if src == "f" else ["none"]):
                gg = g[g[col] == p]
                key = f"{src}{p}"
                r[f"n_{key}"] = len(gg)
                r[f"fin_tone_{key}"] = tone_of(gg) if len(gg) >= MIN_PASS else np.nan
                r[f"SEN_{key}"] = sen_of(gg) if len(gg) >= MIN_PASS else np.nan
        # Cuotas de tema.
        for p in PILLARS:
            r[f"fesg_{p}"] = (g["f_lab"] == p).mean()           # sobre todos los pasajes
            r[f"besg_{p}"] = (g[f"b_{p}"] > CUT).mean()          # ESGBERT (binario por pilar)
            r[f"voc_{p}"] = (g["v_lab"] == p).mean()
        r["fesg_None"] = (g["f_lab"] == "None").mean()
        r["besg_any"] = (g[["b_E", "b_S", "b_G"]].max(axis=1) > CUT).mean()
        fs = sum(r[f"fesg_{p}"] for p in PILLARS)
        bs = sum(r[f"besg_{p}"] for p in PILLARS)
        for p in PILLARS:  # cuotas dentro del texto ESG (comparables a las del paper)
            r[f"fesgn_{p}"] = r[f"fesg_{p}"] / fs if fs else np.nan
            r[f"besgn_{p}"] = r[f"besg_{p}"] / bs if bs else np.nan
        for lab in ["climate_change", "natural_capital", "pollution_waste", "human_capital", "product_liability",
                    "community_relations", "corporate_governance", "business_ethics_values", "non_esg"]:
            r[f"f9_{lab}"] = (g["f9_lab"] == lab).mean()
        env = g[g["b_E"] > CUT]
        r["n_env"] = len(env)
        r["action_env"] = (env["p_action"] > CUT).mean() if len(env) else np.nan
        r["action_all"] = ((g["b_E"] > CUT) & (g["p_action"] > CUT)).mean()
        rows.append(r)
        # Error de muestreo por informe.
        ones = np.ones(n)
        se_rows.append({"doc_key": k, "n": n, "N": N,
                        "se_fin_tone": ratio_se((g["t_pos"] - g["t_neg"]).to_numpy(float),
                                                g["n_sent"].to_numpy(float), N),
                        "se_fesgn_G": ratio_se((g["f_lab"] == "G").to_numpy(float),
                                               g["f_lab"].isin(PILLARS).to_numpy(float), N),
                        "se_besg_G": ratio_se((g["b_G"] > CUT).to_numpy(float), ones, N),
                        "se_besg_E": ratio_se((g["b_E"] > CUT).to_numpy(float), ones, N),
                        "se_action_env": ratio_se(((g["b_E"] > CUT) & (g["p_action"] > CUT)).to_numpy(float),
                                                  (g["b_E"] > CUT).to_numpy(float), N),
                        "se_SEN_s": ratio_se((g["lm_pos"] - g["lm_neg"]).to_numpy(float),
                                             (g["lm_pos"] + g["lm_neg"]).to_numpy(float), N)})
    D = pd.DataFrame(rows)
    keep = ["doc_key", "firm_label", "year", "country", "industry", "SEN_Score", "SUS_density", "QUANT_Score",
            "HEDGE_Score", "Z_SEN", "Z_SUS", "Z_QUANT", "Z_HEDGE", "ESGSI", "ESGSI_ext"]
    D = scores[keep].merge(D, on="doc_key", how="left", validate="one_to_one")
    assert len(D) == 343 and D["fin_tone"].notna().all()
    return D, pd.DataFrame(se_rows)


def netzero_frame() -> pd.DataFrame:
    """NetZero sobre TODOS los pasajes climaticos del #4."""
    P = pd.read_parquet(CB_CACHE / "passages.parquet", columns=["doc_key", "h"])
    det = pd.read_parquet(CB_CACHE / "pred_detector.parquet")[["h", "p_yes"]]
    P = P.merge(det, on="h", how="left", validate="many_to_one")
    P["clim"] = P["p_yes"] > CUT
    nz = pred("netzero")
    P = P.merge(nz, on="h", how="left", validate="many_to_one")
    C = P[P["clim"]]
    assert C["p_none"].notna().all()
    lab = C[["p_none", "p_reduction", "p_net_zero"]].to_numpy().argmax(axis=1)
    C = C.assign(red=lab == 1, net=lab == 2)
    g = C.groupby("doc_key")
    out = pd.DataFrame({"n_clim_all": g.size(), "nz_red": g["red"].mean(), "nz_net": g["net"].mean()})
    out["nz_clim"] = out["nz_red"] + out["nz_net"]
    out["nz_all"] = (out["nz_clim"] * out["n_clim_all"]) / P.groupby("doc_key").size()
    out["nz_prob"] = g.apply(lambda x: (1 - x["p_none"]).mean(), include_groups=False)
    return out.reset_index()


def sen_by_length(scores: pd.DataFrame) -> pd.DataFrame:
    """SEN por informe en (i) los pasajes de >= 30 palabras (toda la poblacion,
    no la muestra) y (ii) los fragmentos cortos (< 30 palabras: rotulos de
    tablas, listas, titulos), con la segmentacion del #4. Sirve para ver donde
    esta la caida de 2024 de Z(SEN)."""
    from cb_common import MIN_WORDS, iter_docs, paragraphs
    rows = []
    for k, zones in iter_docs(scores):
        acc = {"long": [0, 0, 0, 0], "short": [0, 0, 0, 0]}
        for zt in zones:
            for par in paragraphs(zt):
                part = "long" if len(par.split()) >= MIN_WORDS else "short"
                pos, neg, ntok = sen_word_counts(par)
                a = acc[part]
                a[0] += sum(pos.values())
                a[1] += sum(neg.values())
                a[2] += sum(v for w, v in neg.items() if w in ESRS_A)
                a[3] += ntok
        r = {"doc_key": k}
        for part, (p, n, nA, t) in acc.items():
            r[f"SEN_{part}"] = (p - n) / (p + n) if p + n else 0.0
            r[f"SEN_{part}_exAw"] = (p - n + nA) / (p + n - nA) if p + n - nA else 0.0
            r[f"negA_per_1k_{part}"] = nA / max(t, 1) * 1e3
            r[f"tokens_{part}"] = t
        rows.append(r)
    R = pd.DataFrame(rows)
    R["short_token_share"] = R["tokens_short"] / (R["tokens_short"] + R["tokens_long"])
    return R.drop(columns=["tokens_short", "tokens_long"])


def our_pillars() -> pd.DataFrame:
    Pd = pd.read_csv(AV3 / "pillar_by_doc.csv", sep=";").rename(columns={"firm_id": "firm_label"})
    for p in ["E", "S", "G", "TRANS"]:
        Pd[f"sh_{p}"] = Pd[f"m_{p}"] / Pd["m_total"]
    Pd["sh_ET"] = (Pd["m_E"] + Pd["m_TRANS"]) / Pd["m_total"]
    Pd["dens_ET"] = Pd["dens_E"] + Pd["dens_TRANS"]
    return Pd[["firm_label", "year", "sh_E", "sh_S", "sh_G", "sh_TRANS", "sh_ET", "dens_E", "dens_S", "dens_G",
               "dens_TRANS", "dens_ET", "m_total"]]


# ---------------------------------------------------------------------------
# Correlaciones
# ---------------------------------------------------------------------------

def corr_block(D: pd.DataFrame, xs: list[str], ys: list[str], block: str) -> pd.DataFrame:
    return boot_corr(D, xs, ys).assign(block=block)


def corr_pairs(D: pd.DataFrame, pairs: list[tuple[str, str]], block: str) -> pd.DataFrame:
    """Pares con NaN distintos: un bootstrap por par."""
    return pd.concat([boot_corr(D, [a], [b]) for a, b in pairs], ignore_index=True).assign(block=block)


def fmt_corr(C: pd.DataFrame) -> pd.DataFrame:
    t = C[((C["method"] == "spearman") & (C["view"] == "pooled")) |
          ((C["method"] == "pearson") & (C["view"].isin(["within", "between", "twoway"])))].copy()
    t["txt"] = t.apply(lambda r: f"{r.r:+.2f} [{r.ci_lo:+.2f},{r.ci_hi:+.2f}]", axis=1)
    return t.pivot_table(index=["block", "cb", "score"], columns="view", values="txt",
                         aggfunc="first")[["pooled", "within", "between", "twoway"]]


# ---------------------------------------------------------------------------
# Nivel de pasaje
# ---------------------------------------------------------------------------

def within_rho(G: pd.DataFrame, a: str, b: str, min_n: int = 10) -> pd.Series:
    def f(g):
        if len(g) < min_n or g[a].nunique() < 2 or g[b].nunique() < 2:
            return np.nan
        return stats.spearmanr(g[a], g[b])[0]
    return G.groupby("doc_key").apply(f, include_groups=False).dropna()


def summ_rho(rr: pd.Series, G: pd.DataFrame, a: str, b: str) -> dict:
    return {"informes": int(len(rr)), "mediana_rho": float(rr.median()), "p25": float(rr.quantile(.25)),
            "p75": float(rr.quantile(.75)), "cuota_positiva": float((rr > 0).mean()), "pasajes": int(len(G)),
            "rho_todos_los_pasajes": float(stats.spearmanr(G[a], G[b])[0])}


def passage_level(S: pd.DataFrame) -> dict:
    res = {}
    for a, b, sub, nm in [
        ("lm_pol", "ptone", S, "lm_pol~finbert_tone"),
        ("lm_pol", "ptone_prob", S, "lm_pol~finbert_tone_prob"),
        ("lm_pol", "ptone", S[(S["lm_pos"] + S["lm_neg"]) > 0], "lm_pol~finbert_tone|con_tono_LM"),
        ("lm_pol", "quant", S, "control:lm_pol~quant"),
        ("ptone", "quant", S, "control:finbert_tone~quant"),
        ("lm_pol", "b_G", S, "control:lm_pol~p_gobernanza"),
    ]:
        res[nm] = summ_rho(within_rho(sub, a, b), sub, a, b)
    for src, col in [("finbert_esg", "f_lab"), ("vocabulario", "v_lab")]:
        for p in PILLARS + ["None", "none"]:
            sub = S[S[col] == p]
            if len(sub) < 1000:
                continue
            res[f"lm_pol~finbert_tone|pilar_{src}_{p}"] = summ_rho(within_rho(sub, "lm_pol", "ptone"), sub,
                                                                    "lm_pol", "ptone")
    sub = S[~S["clim"]]
    res["lm_pol~finbert_tone|no_climaticos"] = summ_rho(within_rho(sub, "lm_pol", "ptone"), sub, "lm_pol", "ptone")
    rr = within_rho(S, "lm_pol", "ptone").rename("rho").reset_index()
    rr = rr.merge(S[["doc_key", "year"]].drop_duplicates(), on="doc_key")
    res["lm_pol~finbert_tone_por_ano"] = rr.groupby("year")["rho"].median().to_dict()
    # Clase de tono FinBERT de la frase frente a las palabras L&M del pasaje: media del tono L&M por clase
    res["tono_LM_medio_por_signo_finbert"] = {
        "pasajes_finbert_neg": float(S.loc[S["ptone"] < 0, "lm_pol"].mean()),
        "pasajes_finbert_0": float(S.loc[S["ptone"] == 0, "lm_pol"].mean()),
        "pasajes_finbert_pos": float(S.loc[S["ptone"] > 0, "lm_pol"].mean())}
    return res


# ---------------------------------------------------------------------------
# Experimento 3: recall de la extraccion
# ---------------------------------------------------------------------------

def recall_block(scores: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    f = CACHE / "recall_passages.jsonl.gz"
    if not f.exists():
        return {"aviso": "sin datos de recall"}, pd.DataFrame()
    RP = pd.read_json(f, lines=True, dtype={"h": str, "text": str, "doc_key": str})
    Dd = pd.read_csv(CACHE / "recall_docs.csv", sep=";")
    for name, col, new in [("env", "p_environmental", "b_E"), ("soc", "p_social", "b_S"),
                           ("gov", "p_governance", "b_G")]:
        RP = RP.merge(pred(name)[["h", col]].rename(columns={col: new}), on="h", how="left", validate="many_to_one")
    e4 = pred("esg4")
    e4["f_lab"] = argmax_label(e4, ["p_none", "p_environmental", "p_social", "p_governance"], ["None", "E", "S", "G"])
    RP = RP.merge(e4[["h", "f_lab"]], on="h", how="left", validate="many_to_one")
    assert RP[["b_E", "b_S", "b_G", "f_lab"]].notna().all().all()
    flags = {"besg_any": (RP[["b_E", "b_S", "b_G"]].max(axis=1) > CUT)}
    for p in PILLARS:
        flags[f"besg_{p}"] = RP[f"b_{p}"] > CUT
        flags[f"fesg_{p}"] = RP["f_lab"] == p
    flags["fesg_any"] = RP["f_lab"].isin(PILLARS)
    F = pd.DataFrame({k: v.astype(float) for k, v in flags.items()})
    keys = list(flags)
    RP = pd.concat([RP, F], axis=1)
    Dd = Dd.merge(scores[["doc_key"]], on="doc_key")
    docs = Dd["doc_key"].tolist()
    firm_of = dict(zip(Dd["doc_key"], Dd["firm_label"]))
    year_of = dict(zip(Dd["doc_key"], Dd["year"]))
    W = {(r.doc_key, part): getattr(r, f"words_{part}_passages") for r in Dd.itertuples() for part in ("in", "out")}
    arr = {}
    for (k, part), g in RP.groupby(["doc_key", "part"]):
        arr[(k, part)] = (g["n_words"].to_numpy(float), g[keys].to_numpy(float))

    def estimate(sel_docs, resample_rng=None):
        """ESG estimado (palabras) dentro y fuera de la extraccion, por clase."""
        tot = {"in": np.zeros(len(keys)), "out": np.zeros(len(keys))}
        for k in sel_docs:
            for part in ("in", "out"):
                if (k, part) not in arr:
                    continue
                w, X = arr[(k, part)]
                if resample_rng is not None:
                    i = resample_rng.integers(0, len(w), len(w))
                    w, X = w[i], X[i]
                rate = (w[:, None] * X).sum(axis=0) / w.sum()
                tot[part] += W[(k, part)] * rate
        return tot["out"] / (tot["out"] + tot["in"]), tot

    est, tot = estimate(docs)
    firms = sorted(set(firm_of.values()))
    by_firm = {f_: [k for k in docs if firm_of[k] == f_] for f_ in firms}
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(B_CI):
        pick = rng.choice(firms, size=len(firms), replace=True)
        sel = [k for f_ in pick for k in by_firm[f_]]
        draws.append(estimate(sel, rng)[0])
    A = np.array(draws)
    lo, hi = np.nanpercentile(A, 2.5, axis=0), np.nanpercentile(A, 97.5, axis=0)
    # Tasas de clasificacion por parte (ponderadas por palabras, agregadas).
    rates = {}
    for part in ("in", "out"):
        sub = RP[RP["part"] == part]
        rates[part] = {k: float((sub["n_words"] * sub[k]).sum() / sub["n_words"].sum()) for k in keys}
    by_year = []
    for y in sorted(set(year_of.values())):
        ds = [k for k in docs if year_of[k] == y]
        e, _ = estimate(ds)
        by_year.append({"year": y, "informes": len(ds), **{f"fuera_{k}": float(v) for k, v in zip(keys, e)}})
    words_in, words_out = Dd["words_in"].sum(), Dd["words_out"].sum()
    out = {
        "informes": int(len(Dd)), "pasajes_clasificados": int(len(RP)),
        "pasajes_clasificados_por_parte": RP["part"].value_counts().to_dict(),
        "zonas_reproducidas_identicas_a_los_json": int(Dd["zones_match_json"].sum()),
        "cuota_palabras_de_parrafos_extraidas": float(words_in / (words_in + words_out)),
        "cuota_palabras_en_pasajes_ge30": {
            "extraidas": float(Dd["words_in_passages"].sum() / words_in),
            "no_extraidas": float(Dd["words_out_passages"].sum() / words_out)},
        "densidad_vocabulario_pasajes_no_extraidos": {
            "media": float(RP.loc[RP["part"] == "out", "kw_density"].mean()),
            "cuota_cero": float((RP.loc[RP["part"] == "out", "kw_density"] == 0).mean())},
        "tasa_clasificada_ponderada_por_palabras": rates,
        "cuota_del_texto_ESG_fuera_de_la_extraccion": {
            k: {"estimacion": float(v), "ci95": [float(a), float(b)]} for k, v, a, b in zip(keys, est, lo, hi)},
        "palabras_ESG_estimadas": {part: {k: float(v) for k, v in zip(keys, tot[part])} for part in tot},
    }
    return out, pd.DataFrame(by_year)


# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    scores = load_scores()
    S = passage_frame(scores)
    D, SE = doc_frame(S, scores)
    D = D.merge(netzero_frame(), on="doc_key", how="left", validate="one_to_one")
    D = D.merge(sen_by_length(scores), on="doc_key", how="left", validate="one_to_one")
    D = D.merge(our_pillars(), on=["firm_label", "year"], how="left", validate="one_to_one")
    assert D["sh_G"].notna().all()
    cb = pd.read_csv(BASE_CB := (OUT.parent / "climatebert" / "doc_level_cb.csv"), sep=";")
    D = D.merge(cb[["firm_label", "year", "sent_net", "spec_share", "clim_share", "commit_share", "CTI"]],
                on=["firm_label", "year"], how="left", validate="one_to_one")
    D["firm"] = D["firm_label"]
    D.drop(columns=["doc_key", "firm"]).to_csv(OUT / "doc_level_esgm.csv", sep=";", index=False)

    Sm: dict = {"modelos": {k: {"repo": v["repo"], "revision": v["rev"], "unidad": v["unit"]}
                            for k, v in MODELS.items()}}
    rtf = CACHE / "runtime.json"
    if rtf.exists():
        Sm["runtime"] = json.loads(rtf.read_text())
    Sm["muestra"] = json.loads((OUT / "sample.json").read_text(encoding="utf-8"))
    SE = SE.merge(D[["doc_key", "fin_tone"]], on="doc_key")
    Sm["error_de_muestreo_por_informe"] = {
        c: {"mediana": float(SE[c].median()), "p90": float(SE[c].quantile(.9)),
            "fiabilidad_entre_informes": float(1 - np.nanmean(SE[c] ** 2) /
                                               D[c.replace("se_", "")].var(ddof=1))}
        for c in ["se_fin_tone", "se_fesgn_G", "se_besg_G", "se_besg_E", "se_action_env", "se_SEN_s"]}
    Sm["cobertura"] = {
        "pasajes_muestra": int(len(S)), "frases": int(S["n_sent"].sum()),
        "cuota_pasajes_finbert_esg": S["f_lab"].value_counts(normalize=True).to_dict(),
        "cuota_pasajes_esgbert": {p: float((S[f"b_{p}"] > CUT).mean()) for p in PILLARS},
        "cuota_pasajes_esgbert_alguno": float((S[["b_E", "b_S", "b_G"]].max(axis=1) > CUT).mean()),
        "cuota_pasajes_pilar_vocabulario": S["v_lab"].value_counts(normalize=True).to_dict(),
        "finbert9": S["f9_lab"].value_counts(normalize=True).to_dict(),
        "acuerdo_pilar_finbert_esgbert": {p: float(((S["f_lab"] == p) == (S[f"b_{p}"] > CUT)).mean())
                                          for p in PILLARS},
        "acuerdo_pilar_finbert_vocabulario": float((S.loc[S["v_lab"] != "none", "f_lab"] ==
                                                    S.loc[S["v_lab"] != "none", "v_lab"]).mean()),
        "pasajes_ambientales_esgbert_muestra": int((S["b_E"] > CUT).sum()),
        "pasajes_climaticos_netzero_todos": int(D["n_clim_all"].sum()),
        "frases_finbert": {"positivas": float(S["t_pos"].sum() / S["n_sent"].sum()),
                           "negativas": float(S["t_neg"].sum() / S["n_sent"].sum())},
        "pasajes_con_lista_A": {"alguna": float(S["A_any"].mean()), "dominados": float(S["A_dom"].mean())},
        "informes_con_tono_por_pilar_finbert": {p: int(D[f"fin_tone_f{p}"].notna().sum()) for p in PILLARS},
        "informes_con_tono_por_pilar_vocabulario": {p: int(D[f"fin_tone_v{p}"].notna().sum()) for p in PILLARS},
    }
    D["firm"] = D["firm_label"]

    # --- Correlaciones
    C = []
    C.append(corr_block(D, ["fin_tone", "fin_tone_prob", "fin_pos", "fin_neg"],
                        ["Z_SEN", "SEN_s", "Z_SUS", "Z_QUANT", "Z_HEDGE", "ESGSI", "ESGSI_ext"], "1_tono"))
    C.append(corr_pairs(D, [("fin_tone", "sent_net"), ("SEN_s", "Z_SEN"), ("fin_tone_exAdom", "SEN_s_exAdom"),
                            ("fin_tone", "SEN_s_exAw")], "1_tono_extra"))
    pairs = []
    for src in ["f", "v"]:
        for p in PILLARS:
            pairs.append((f"fin_tone_{src}{p}", f"SEN_{src}{p}"))
    pairs.append(("fin_tone_fNone", "SEN_fNone"))
    C.append(corr_pairs(D, pairs, "1_tono_por_pilar"))
    C.append(corr_block(D, ["fesgn_E", "fesgn_S", "fesgn_G", "besgn_E", "besgn_S", "besgn_G"],
                        ["sh_ET", "sh_E", "sh_S", "sh_G", "sh_TRANS"], "2_pilares_cuotas"))
    C.append(corr_block(D, ["fesg_E", "fesg_S", "fesg_G", "besg_E", "besg_S", "besg_G", "besg_any"],
                        ["dens_ET", "dens_S", "dens_G", "SUS_density"], "2_pilares_densidades"))
    C.append(corr_block(D, ["voc_E", "voc_S", "voc_G"], ["sh_ET", "sh_S", "sh_G"], "2_control_vocabulario_crudo"))
    C.append(corr_pairs(D, [(a, b) for a in ["action_env", "action_all", "nz_clim", "nz_all", "nz_net", "nz_red",
                                              "nz_prob"]
                            for b in ["Z_SUS", "Z_QUANT", "ESGSI", "ESGSI_ext", "spec_share", "clim_share"]],
                        "4_sustancia"))
    C = pd.concat(C, ignore_index=True)
    C.to_csv(OUT / "correlations.csv", sep=";", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print(fmt_corr(C).to_string())

    # --- Ruptura de 2024
    brk = ["fin_tone", "fin_tone_prob", "fin_pos", "fin_neg", "fin_tone_exAdom", "fin_tone_exAany",
           "Z_SEN", "SEN_s", "SEN_s_exAw", "SEN_s_exAdom", "SEN_s_exAany", "A_dom_share", "A_any_share",
           "SEN_long", "SEN_long_exAw", "SEN_short", "SEN_short_exAw", "negA_per_1k_long", "negA_per_1k_short",
           "short_token_share",
           "fin_tone_fE", "fin_tone_fS", "fin_tone_fG", "fin_tone_fNone", "SEN_fE", "SEN_fS", "SEN_fG",
           "fin_tone_vE", "fin_tone_vS", "fin_tone_vG",
           "fesgn_G", "besgn_G", "sh_G", "fesgn_E", "sh_ET", "action_env", "action_all", "nz_clim", "nz_all",
           "Z_SUS", "Z_QUANT"]
    Sm["ruptura_2024"] = {c: break_block(D, c, av.N_BOOT) for c in brk}
    for c, v in Sm["ruptura_2024"].items():
        e, f_ = v["salto_ef"], v["salto_sobre_tendencia"]
        print(f"{c:16s} salto={e['b']:+.3f} [{e['ci95'][0]:+.3f},{e['ci95'][1]:+.3f}] p={e['p']:.3g} "
              f"pw={e['p_wild']:.3g} | s/tend={f_['b']:+.3f} p={f_['p']:.3g} | "
              f"pend.pre={v['pendiente_2018_2023_z_pre']['pendiente']:+.3f} p={v['pendiente_2018_2023_z_pre']['p']:.3g}")
    # Amplitud por empresa
    Sm["ruptura_2024_por_empresa"] = {}
    for c in ["fin_tone", "fin_tone_exAdom", "SEN_s", "SEN_s_exAdom"]:
        w = D.pivot_table(index="firm", columns="year", values=c)
        dif = (w[2024] - w.loc[:, 2018:2023].mean(axis=1)).dropna()
        Sm["ruptura_2024_por_empresa"][c] = {"empresas": int(len(dif)), "cuota_que_baja": float((dif < 0).mean())}

    # --- Tendencias (EF de empresa, cluster por empresa)
    tr = ["fesg_E", "fesg_S", "fesg_G", "fesgn_E", "fesgn_S", "fesgn_G", "besgn_E", "besgn_S", "besgn_G", "besg_E", "besg_S", "besg_G",
          "fesg_None", "sh_ET", "sh_E", "sh_S", "sh_G", "sh_TRANS", "voc_G", "action_env", "action_all",
          "nz_clim", "nz_all", "nz_net", "nz_red", "fin_tone", "f9_corporate_governance",
          "f9_business_ethics_values"]
    Sm["pendientes"] = {}
    for c in tr:
        d = D.dropna(subset=[c])
        tb = av.trend_block(d, c, av.N_BOOT, av.SEED)
        tb["ef_2018_2023"] = av.fe_slope(d[d["year"] <= 2023], c)
        tb["media_2018"] = float(d.loc[d["year"] == 2018, c].mean())
        tb["media_2024"] = float(d.loc[d["year"] == 2024, c].mean())
        Sm["pendientes"][c] = tb
        e = tb["ef_cluster"]
        print(f"{c:28s} 2018={tb['media_2018']:.3f} 2024={tb['media_2024']:.3f} pend={e['pendiente']:+.4f} "
              f"[{e['ci95'][0]:+.4f},{e['ci95'][1]:+.4f}] p={e['p']:.3g} pw={tb['ef_wild_bootstrap']['p_wild']:.3g}")
    ym = D.groupby("year")[[c for c in dict.fromkeys(tr + brk) if c in D.columns and not c.startswith("Z_")]
                           + ["Z_SEN", "Z_SUS", "Z_QUANT"]].mean()
    ym.round(4).to_csv(OUT / "yearly_means.csv", sep=";")
    yc = []
    for c in ["fin_tone", "fin_tone_exAdom", "SEN_s", "SEN_s_exAdom", "Z_SEN", "fesgn_G", "besgn_G", "sh_G",
              "action_env", "nz_clim"]:
        d = D.dropna(subset=[c]).copy()
        d[c] = zpop(d[c])
        yc.append(av.yearly_ci(d, c))
    pd.concat(yc, ignore_index=True).round(4).to_csv(OUT / "yearly_ci_z.csv", sep=";", index=False)

    # --- Nivel de pasaje
    Sm["pasaje"] = passage_level(S)
    print(json.dumps(av.jsonable(Sm["pasaje"]), indent=1, ensure_ascii=False))

    # --- Recall
    Sm["recall"], RY = recall_block(scores)
    if len(RY):
        RY.round(4).to_csv(OUT / "recall_by_year.csv", sep=";", index=False)
    print(json.dumps(av.jsonable(Sm["recall"]), indent=1, ensure_ascii=False))
    print(RY.round(3).to_string())

    Sm["segundos"] = round(time.time() - t0, 1)
    with open(OUT / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(av.jsonable(Sm), fh, indent=2, ensure_ascii=False)
    print(json.dumps(av.jsonable({k: Sm[k] for k in ["cobertura", "error_de_muestreo_por_informe"]}), indent=1,
                     ensure_ascii=False))
    print(ym.round(3).T.to_string())
    print(f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
