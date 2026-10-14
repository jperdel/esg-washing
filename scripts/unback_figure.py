"""
unback_figure.py
----------------
Figura del piloto UNBACK (issue #7), con la misma estructura que la Figura 1
del paper pero con UNBACK en lugar de HEDGE. No toca el paper.

    (a) ESGSI y ESGSI extendido con UNBACK (pesos 0,5 y 0,5)
    (b) Z(SUS) y Z(SEN)
    (c) Z(QUANT) y Z(UNBACK); Z(UNBACK_DB), sin cantidad, en trazo fino

Medias anuales con IC 95 % agrupado por empresa (analysis_v3.yearly_ci).
Lee results/unback/doc_level_unback.csv (unback_pilot.py) y escribe
results/unback/fig_temporal_unback.pdf y .png.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_v3 as A  # noqa: E402
import figures_v3 as F  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results" / "unback"


def main():
    D = pd.read_csv(OUT / "doc_level_unback.csv", sep=";").rename(columns={"firm_label": "firm"})
    D["ESGSI_ext_unback"] = D["Z_SEN"] - D["Z_SUS"] - 0.5 * D["Z_QUANT"] + 0.5 * D["Z_UNBACK"]
    cols = ["ESGSI", "ESGSI_ext_unback", "Z_SUS", "Z_SEN", "Z_QUANT", "Z_UNBACK", "Z_UNBACK_DB"]
    ci = pd.concat([A.yearly_ci(D, c) for c in cols])

    F.setup_style()
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(F.FULL, 2.6), sharey=True, layout="constrained")
    years = sorted(ci.year.unique())
    style = {"ESGSI": F.SCORE_STYLE["ESGSI"],
             "ESGSI_ext_unback": {**F.SCORE_STYLE["ESGSI_ext"], "label": "Extended ESGSI (UNBACK)"}}
    for var, st in style.items():
        e = ci[ci.variable == var]
        a.fill_between(e.year, e.ci_lo, e.ci_hi, color=st["color"], alpha=st["band"], lw=0)
        a.plot(e.year, e["mean"], color=st["color"], ls=st["ls"], marker=st["marker"], ms=4.5,
               mec="white", mew=1, label=st["label"])
    a.set_title("(a) Scores")
    a.set_ylabel("Mean and 95% CI (z units; 0 = corpus mean)")
    a.legend(loc="upper right", handlelength=2.2)
    for ax, comps in ((b, (("Z_SUS", F.C1, "o", r"$Z$(SUS$^{\rm dens}$)"), ("Z_SEN", F.C2, "s", r"$Z$(SEN)"))),
                      (c, (("Z_QUANT", F.C3, "^", r"$Z$(QUANT)"), ("Z_UNBACK", F.C4, "v", r"$Z$(UNBACK)")))):
        for var, col, mk, lab in comps:
            s = ci[ci.variable == var]
            ax.fill_between(s.year, s.ci_lo, s.ci_hi, color=col, alpha=0.13, lw=0)
            ax.plot(s.year, s["mean"], color=col, marker=mk, ms=4.5, mec="white", mew=1, label=lab)
    s = ci[ci.variable == "Z_UNBACK_DB"]
    c.plot(s.year, s["mean"], color=F.C4, ls=(0, (2, 1.5)), lw=0.9, label=r"$Z$(UNBACK), no quantity")
    for ax in (a, b, c):
        ax.axhline(0, color=F.AXIS, lw=0.8)
        ax.set_xticks(years)
        ax.set_xticklabels([str(y) if (y - years[0]) % 2 == 0 else "" for y in years])
        ax.set_xlim(years[0] - 0.4, years[-1] + 0.4)
    b.legend(loc="upper left", handlelength=1.6)
    c.legend(loc="upper left", handlelength=1.6)
    b.set_title("(b) Components of the ESGSI")
    c.set_title("(c) Added in the Extended ESGSI")
    lim = np.ceil(max(abs(ci.ci_lo.min()), abs(ci.ci_hi.max())) * 4 + 0.5) / 4
    a.set_ylim(-lim, lim)
    fig.savefig(OUT / "fig_temporal_unback.pdf")
    fig.savefig(OUT / "fig_temporal_unback.png", dpi=200)
    plt.close(fig)
    print(ci[ci.variable.isin(["ESGSI_ext_unback", "Z_UNBACK", "Z_UNBACK_DB", "Z_QUANT"])]
          .pivot(index="variable", columns="year", values="mean").round(3))


if __name__ == "__main__":
    main()
