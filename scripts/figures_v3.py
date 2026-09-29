"""
figures_v3.py
-------------
Figuras vectoriales (PDF) del paper a partir de las salidas de
scripts/analysis_v3.py y de las pruebas de estabilidad. No recalcula nada del
indice: solo lee CSV/JSON ya escritos.

    python scripts/figures_v3.py
    python scripts/figures_v3.py --analysis <dir> --stability-dir <dir> --out <dir>

Salidas en paper/figures/:
    fig_temporal.pdf            media anual del ESGSI y del ESGSI extendido (IC 95 % agrupado)
                                y componentes Z(SUS), Z(SEN)
    fig_industry_year.pdf       mapas de calor industria x ano: (a) ESGSI, (b) ESGSI extendido
    fig_firms.pdf               empresas ANONIMAS por industria: media 2018-2024 y rango anual,
                                (a) ESGSI, (b) ESGSI extendido
    fig_pillars.pdf             cuota de menciones por pilar, por industria y por ano
    fig_countries.pdf           trayectorias por pais (multiples pequenos)
    fig_stability.pdf           Spearman, solapamiento de deciles extremos y pendiente frente al borrado de vocabulario
    fig_sus_specs.pdf           las tres especificaciones del SUS y la amplitud
    fig_ext_weights.pdf         rejilla de pesos del ESGSI extendido: rho con (0.5, 0.5) y pendiente EF

Sin etiquetas ni umbral (el indice es un score) y sin nombres de empresa.

Estilo (adaptado a impresion): fondo blanco, trazos finos, rejilla en linea
fina continua, a lo sumo cuatro tonos categoricos (orden fijo, validado para
daltonismo), identidad nunca solo por color (etiquetas directas, marcadores
distintos o valores impresos), fuente con serifa como el cuerpo del paper.

Convenciones comunes a todas las figuras:
  * tamano final exacto (ancho 6.3 in, sin recorte 'tight'), de modo que los
    tamanos de fuente del codigo son los impresos (>= 7 pt);
  * el ESGSI es siempre tinta oscura (INK_SERIES); Z(SUS) azul, Z(SEN) naranja;
  * las industrias ICB aparecen siempre en el mismo orden (ESGSI medio de los
    informes, de mayor a menor), el del mapa de calor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgb
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

BASE_DIR = Path(__file__).resolve().parent.parent

# --- Paleta (orden categorico fijo; validado con el validador de dataviz) ---
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"   # azul, naranja, aqua, amarillo
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
INK_SERIES = "#222220"          # serie del ESGSI (enfasis en tinta; se lee igual en gris)
GRID, AXIS = "#e1e0d9", "#c3c2b7"
GRAY_LINE = "#c9c8c2"
NEUTRAL = "#f0efec"
DIVERGING = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#6da7ec", NEUTRAL, "#ee8a88", "#c23a39"])
PILLAR_COL = {"E": C1, "S": C2, "G": C3, "TRANS": C4}
PILLAR_NAME = {"E": "Environmental", "S": "Social", "G": "Governance", "TRANS": "Transversal"}
COUNTRY_CODE = {"Germany": "DE", "France": "FR", "Netherlands": "NL", "Italy": "IT",
                "Spain": "ES", "Finland": "FI", "Belgium": "BE"}
MINUS = "−"
# ESGSI: tinta, linea continua, circulo. ESGSI_ext: gris oscuro, discontinua,
# rombo (identidad por trazo y marcador, no solo por tono; se lee en gris).
SCORE_STYLE = {
    "ESGSI": {"color": INK_SERIES, "ls": "-", "marker": "o", "band": 0.10, "label": "ESGSI"},
    "ESGSI_ext": {"color": INK2, "ls": (0, (4, 2)), "marker": "D", "band": 0.08, "label": "Extended ESGSI"},
}
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])
ORANGES = LinearSegmentedColormap.from_list("oranges", ["#fde3d4", "#f4a57f", "#eb6834", "#b84a1c", "#7d3010"])

FULL, HALF = 6.3, 3.1
FS_SMALL = 7.0      # minimo impreso


def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
        "axes.axisbelow": True, "axes.titlelocation": "left", "axes.titlecolor": INK,
        "lines.linewidth": 1.4, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.unicode_minus": True,
        "figure.constrained_layout.h_pad": 0.03, "figure.constrained_layout.w_pad": 0.03,
    })


def rd(p: Path) -> pd.DataFrame:
    return pd.read_csv(p, sep=";", encoding="utf-8")


def ink_on(color) -> str:
    r, g, b = to_rgb(color)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if lum < 0.5 else INK


def fmt(v: float, nd: int = 2, sign: bool = False) -> str:
    s = f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"
    return s.replace("-", MINUS)


def industry_order(d: pd.DataFrame) -> list[str]:
    """Orden comun de industrias: ESGSI medio de los informes, de mayor a menor."""
    return d.groupby("industry")["ESGSI"].mean().sort_values(ascending=False).index.tolist()


def save(fig, out: Path, name: str):
    # sin bbox 'tight': el PDF conserva el ancho pedido y las fuentes su tamano impreso
    fig.savefig(out / name)
    plt.close(fig)
    print(f"  {name}")


# ---------------------------------------------------------------------------

def fig_temporal(A: Path, out: Path):
    ci = rd(A / "yearly_ci.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 2.6), sharey=True, layout="constrained")
    years = sorted(ci.year.unique())
    for var, st in SCORE_STYLE.items():
        e = ci[ci.variable == var]
        a.fill_between(e.year, e.ci_lo, e.ci_hi, color=st["color"], alpha=st["band"], lw=0)
        a.plot(e.year, e["mean"], color=st["color"], ls=st["ls"], marker=st["marker"], ms=4.5,
               mec="white", mew=1, label=st["label"])
    a.axhline(0, color=AXIS, lw=0.8)
    a.set_title("(a) Yearly mean and 95% CI (clustered by firm)")
    a.set_ylabel("Mean (z units; 0 = corpus mean)")
    a.legend(loc="upper right", handlelength=2.2)
    for var, col, mk, lab in (("Z_SUS", C1, "o", r"$Z$(SUS$^{\rm dens}$)"), ("Z_SEN", C2, "s", r"$Z$(SEN)")):
        s = ci[ci.variable == var]
        b.fill_between(s.year, s.ci_lo, s.ci_hi, color=col, alpha=0.13, lw=0)
        b.plot(s.year, s["mean"], color=col, marker=mk, ms=4.5, mec="white", mew=1, label=lab)
    b.axhline(0, color=AXIS, lw=0.8)
    b.set_title(f"(b) Components: ESGSI = $Z$(SEN) {MINUS} $Z$(SUS)")
    b.legend(loc="upper left", handlelength=1.6)
    lo = ci[ci.variable.isin(["ESGSI", "ESGSI_ext", "Z_SUS", "Z_SEN"])]
    lim = np.ceil(max(abs(lo.ci_lo.min()), abs(lo.ci_hi.max())) * 4 + 0.5) / 4
    for ax in (a, b):
        ax.set_xticks(years)
        ax.set_xlim(years[0] - 0.4, years[-1] + 0.4)
    a.set_ylim(-lim, lim)
    save(fig, out, "fig_temporal.pdf")


def _heat(ax, m: pd.DataFrame, norm, cmap, fmt_nd=2):
    im = ax.imshow(m.to_numpy(), cmap=cmap, norm=norm, aspect="auto")
    ax.grid(False)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            val = m.iat[i, j]
            ax.text(j, i, fmt(val, fmt_nd), ha="center", va="center", fontsize=7.5,
                    color=ink_on(cmap(norm(val))))
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, m.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, m.shape[0]), minor=True)
    ax.grid(which="minor", color="white", lw=1.5)
    ax.tick_params(which="minor", length=0)
    return im


def fig_industry_year(A: Path, out: Path):
    d = rd(A / "doc_level.csv")
    nf = d.groupby("industry")["firm_id"].nunique()
    order = industry_order(d)
    ms = {v: d.pivot_table(index="industry", columns="year", values=v, aggfunc="mean").loc[order]
          for v in SCORE_STYLE}
    vmax = max(np.nanmax(np.abs(m.to_numpy())) for m in ms.values())
    v = np.ceil(vmax * 2) / 2      # escala simetrica comun, redondeada
    norm = TwoSlopeNorm(0, -v, v)
    fig, axes = plt.subplots(2, 1, figsize=(FULL, 5.4), layout="constrained", sharex=True)
    for ax, (var, m), tag in zip(axes, ms.items(), "ab"):
        im = _heat(ax, m, norm, DIVERGING)
        ax.set_xticks(range(m.shape[1]), m.columns.astype(str))
        ax.set_yticks(range(m.shape[0]), [f"{i} ({nf[i]})" for i in m.index])
        ax.xaxis.tick_top()
        ax.tick_params(labeltop=(tag == "a"), labelbottom=False)
        ax.set_title(f"({tag}) {SCORE_STYLE[var]['label']}", pad=14 if tag == "a" else 4)
        ax.set_ylabel("ICB industry (number of firms)")
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.015, aspect=40)
    cb.set_label("Mean of the industry's reports (z units; 0 = corpus mean)", color=INK2, fontsize=7.5)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=FS_SMALL, length=2)
    save(fig, out, "fig_industry_year.pdf")


def fig_firms(A: Path, out: Path):
    """Empresas anonimas: media 2018-2024 (punto) y rango de sus valores anuales
    (linea), agrupadas por industria; sin nombres."""
    fm = rd(A / "firm_level.csv")
    d = rd(A / "doc_level.csv")
    order = industry_order(d)
    fm["ind_pos"] = fm["industry"].map({k: i for i, k in enumerate(order)})
    fm = fm.sort_values(["ind_pos", "ESGSI"], ascending=[True, False]).reset_index(drop=True)
    # posiciones con un hueco entre industrias
    y, pos, prev = [], 0.0, None
    for ind in fm["industry"]:
        if prev is not None and ind != prev:
            pos += 0.9
        y.append(pos)
        pos += 1
        prev = ind
    y = -np.array(y)
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 6.2), sharey=True, layout="constrained")
    for ax, (var, st), tag in zip(axes, SCORE_STYLE.items(), "ab"):
        ax.hlines(y, fm[f"min_{var}"], fm[f"max_{var}"], color=GRAY_LINE, lw=1.3, zorder=1)
        ax.scatter(fm[var], y, s=20, color=st["color"], marker=st["marker"], edgecolor="white",
                   linewidth=0.9, zorder=3)
        ax.axvline(0, color=MUTED, lw=0.7, zorder=0)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        lo, hi = np.floor(fm[f"min_{var}"].min()), np.ceil(fm[f"max_{var}"].max())
        ax.set_xlim(lo - 0.2, hi + 0.2)
        step = 2 if hi - lo > 8 else 1
        ax.set_xticks(np.arange(lo + (lo % step), hi + 1, step))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda t, _: fmt(t, 0)))
        ax.set_title(f"({tag}) {st['label']}")
        ax.set_xlabel("z units (0 = corpus mean)")
    # etiquetas de industria centradas en su bloque; separadores finos
    centers = [y[fm.industry == ind].mean() for ind in order]
    nfirm = fm.groupby("industry").size()
    axes[0].set_yticks(centers, [f"{ind} ({nfirm[ind]})" for ind in order])
    for ind in order[1:]:
        yy = y[fm.industry == ind].max() + 0.95
        for ax in axes:
            ax.axhline(yy, color=GRID, lw=0.6, zorder=0)
    axes[0].set_ylim(y.min() - 0.8, y.max() + 0.8)
    fig.supxlabel("Each row is one firm (anonymous), ordered within its industry by mean ESGSI. "
                  "Dot: mean over 2018–2024; line: range of its yearly values.",
                  fontsize=FS_SMALL, color=INK2)
    save(fig, out, "fig_firms.pdf")


def fig_pillars(A: Path, out: Path):
    d = rd(A / "doc_level.csv")
    ind = rd(A / "pillar_by_industry.csv").set_index("industry").loc[industry_order(d)].reset_index()
    yr = rd(A / "pillar_by_year.csv")
    ps = ["E", "S", "G", "TRANS"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 3.0), layout="constrained",
                               gridspec_kw={"width_ratios": [1.3, 1]})

    def stack(ax, df, labels):
        left = np.zeros(len(df))
        yy = np.arange(len(df))
        for p in ps:
            w = 100 * df[f"share_{p}"].to_numpy()
            ax.barh(yy, w, left=left, height=0.7, color=PILLAR_COL[p], edgecolor="white",
                    linewidth=1.0, label=PILLAR_NAME[p])
            for yi, (l, wi) in enumerate(zip(left, w)):
                if wi >= 8:
                    ax.text(l + wi / 2, yi, f"{wi:.0f}", ha="center", va="center", fontsize=FS_SMALL,
                            color=ink_on(PILLAR_COL[p]))
            left += w
        ax.set_yticks(yy, labels)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.invert_yaxis()
        ax.set_xlabel("Share of ESG mentions (%)")

    stack(a, ind, ind["industry"].tolist())
    a.set_title("(a) By ICB industry (ordered by mean ESGSI)")
    stack(b, yr, yr["year"].astype(str).tolist())
    b.set_title("(b) By year, all firms")
    h, l = a.get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=4, handlelength=1.2, handleheight=0.9)
    save(fig, out, "fig_pillars.pdf")


def fig_countries(A: Path, out: Path):
    d = rd(A / "doc_level.csv")
    sl = rd(A / "group_slopes.csv")
    sl = sl[(sl.dimension == "country") & (sl.variable == "ESGSI")].set_index("group")
    summ = json.loads((A / "summary.json").read_text(encoding="utf-8"))
    sl.loc["All", "slope"] = summ["c_temporal"]["tendencias"]["ESGSI"]["ef_cluster"]["pendiente"]
    order = d.groupby("country")["firm_id"].nunique().sort_values(ascending=False).index.tolist()
    fig, axes = plt.subplots(2, 4, figsize=(FULL, 3.6), sharex=True, sharey=True, layout="constrained")
    years = sorted(d.year.unique())
    panels = order + ["All"]
    for ax, c in zip(axes.flat, panels):
        sub = d if c == "All" else d[d.country == c]
        for f, g in sub.groupby("firm_id"):
            ax.plot(g.year, g.ESGSI, color=GRAY_LINE, lw=0.6, zorder=1)
        mm = sub.groupby("year")["ESGSI"].mean()
        ax.plot(mm.index, mm.values, color=INK_SERIES, lw=1.6, marker="o", ms=3.2, mec="white",
                mew=0.8, zorder=3)
        ax.axhline(0, color=MUTED, lw=0.6, zorder=0)
        nf = sub.firm_id.nunique()
        name = "All countries" if c == "All" else c
        ax.set_title(f"{name} ({nf} firm{'s' if nf > 1 else ''})")
        key = c if c in sl.index else None
        if key is not None:
            r = sl.loc[key]
            ax.text(0.97, 0.97, f"slope {fmt(r.slope, 2, sign=True)}/yr",
                    transform=ax.transAxes, fontsize=FS_SMALL, color=INK2, ha="right", va="top",
                    bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.9))
        ax.set_xticks([years[0], years[3], years[-1]])
        ax.set_yticks([-4, -2, 0, 2, 4])
    for ax in axes[:, 0]:
        ax.set_ylabel("ESGSI")
    fig.supxlabel("Grey lines: individual firms. Dark line: mean of the reports in that year. "
                  "Slope: within-firm (firm fixed effects) trend.", fontsize=FS_SMALL, color=INK2)
    save(fig, out, "fig_countries.pdf")


def fig_stability(S: Path, out: Path):
    """Estabilidad del ESGSI frente a borrados del vocabulario v3 (medidas de score)."""
    rb = rd(S / "e2e_borrado_aleatorio.csv")
    summ = json.loads((S / "e2e_summary.json").read_text(encoding="utf-8"))
    n_ent = summ["entradas"]
    k_dec = summ["decil_k"]
    ref_slope = summ["referencia"]["re-extraida"]["ESGSI"]["pendiente_ef"]
    # solapamiento de deciles extremos: media del 10 % superior y del inferior, por sorteo
    for mo in ("fija", "re-extraida"):
        rb[f"{mo}_ESGSI_dec"] = 100 * (rb[f"{mo}_ESGSI_top10"] + rb[f"{mo}_ESGSI_bot10"]) / 2
    # fija = contexto (gris); re-extraida = prueba de referencia (azul)
    modes = (("fija", MUTED, "s", "Fixed extracted text"),
             ("re-extraida", C1, "o", "Simulated re-extraction"))
    panels = (("rho", 1.0), ("dec", 100.0), ("pendiente_ef", ref_slope))
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.6), layout="constrained")
    x = np.r_[0, 100 * np.array(sorted(rb.fraccion.unique()))]
    for mode, col, mk, lab in modes:
        g = rb.groupby("fraccion")
        for ax, (var, ref) in zip(axes, panels):
            q = g[f"{mode}_ESGSI_{var}"].quantile([0.05, 0.5, 0.95]).unstack()
            lo = np.r_[ref, q[0.05].to_numpy()]
            mid = np.r_[ref, q[0.5].to_numpy()]
            hi = np.r_[ref, q[0.95].to_numpy()]
            ax.fill_between(x, lo, hi, color=col, alpha=0.13, lw=0)
            ax.plot(x, mid, color=col, marker=mk, ms=3.8, mec="white", mew=0.8, label=lab)
    # supresiones con nombre (re-extraccion), en la fraccion de entradas quitadas;
    # etiqueta corta junto al punto, desplazamiento (puntos) fijado a mano.
    names = {"sin sectoriales": "Sect", "sin los terminos con aviso": "Warn",
             "sin pilar E": "E", "sin pilar S": "S", "sin pilar G": "G",
             "sin pilar TRANS": "T", "sin 'board director'": "BD"}
    # etiquetas del grupo que se amontona cerca de 0-8 %: en hueco libre, con guia
    # (coordenadas de datos por panel, fijadas a mano); el resto, junto al punto.
    PLACED = {("BD", 0): (3.0, 0.962), ("Warn", 0): (3.0, 0.953), ("T", 0): (9.5, 0.944),
              ("BD", 1): (2.0, 85.0), ("Warn", 1): (2.0, 81.0), ("T", 1): (9.0, 77.0),
              ("BD", 2): (2.0, -0.214), ("Warn", 2): (0.5, -0.150), ("Sect", 2): (11.0, -0.145)}
    for k, lab in names.items():
        v = summ["variantes"].get(k)
        if not v:
            continue
        xv = 100 * v["entradas_quitadas"] / n_ent
        e = v["re-extraida"]["ESGSI"]
        vals = (e["rho"], 100 * (e["top10"] + e["bot10"]) / 2, e["pendiente_ef"])
        for i, (ax, yv) in enumerate(zip(axes, vals)):
            ax.scatter([xv], [yv], marker="D", s=16, color=INK, edgecolor="white", linewidth=0.8, zorder=5)
            if (lab, i) in PLACED:
                ax.annotate(lab, (xv, yv), xytext=PLACED[(lab, i)], textcoords="data", fontsize=FS_SMALL,
                            color=INK, ha="left", va="center", zorder=6,
                            bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.85),
                            arrowprops=dict(arrowstyle="-", lw=0.5, color=INK2, shrinkA=1, shrinkB=2.5))
            else:
                ax.annotate(lab, (xv, yv), xytext=(4, 2), textcoords="offset points", fontsize=FS_SMALL,
                            color=INK, ha="left", va="bottom")
    for ax in axes:
        ax.set_xlabel(f"Entries deleted (% of {n_ent})")
        ax.set_xlim(-2, 52)
        ax.set_xticks([0, 10, 20, 30, 40, 50])
    a, b, c = axes
    a.set_title("(a) Spearman ρ with full vocabulary")
    a.set_ylabel("Spearman ρ")
    a.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.02))
    a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda t, _: fmt(t, 2)))
    b.set_title("(b) Extreme-decile overlap")
    b.set_ylabel("Top/bottom-10% reports kept (%)")
    b.set_ylim(None, 102)
    c.set_title("(c) Within-firm trend slope")
    c.set_ylabel("Slope per year (firm FE)")
    c.axhline(0, color=MUTED, lw=0.6)
    c.axhline(ref_slope, color=AXIS, lw=0.8, zorder=0)
    c.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda t, _: fmt(t, 2)))
    h = [Line2D([], [], color=cc, marker=m, ms=3.8, mec="white", label=l) for _, cc, m, l in modes]
    h.append(Line2D([], [], ls="", marker="D", ms=4, color=INK, mec="white",
                    label="Named deletion (re-extraction)"))
    leg = fig.legend(handles=h, loc="outside lower center", ncol=3, handlelength=1.6,
                     title=f"Lines: median of 1,000 random draws; bands: 5th–95th percentile; "
                           f"overlap over the {k_dec} reports of each extreme decile.\n"
                           "E, S, G, T: pillar; Sect: sectoral entries; Warn: warned terms; "
                           "BD: board of directors.",
                     title_fontsize=FS_SMALL)
    leg.get_title().set_color(INK2)
    save(fig, out, "fig_stability.pdf")


def fig_sus_specs(A: Path, out: Path):
    d = rd(A / "doc_level.csv")
    corr = rd(A / "sus_corr_pearson.csv").set_index("var")
    dens = r"SUS$^{\rm dens}$ (mentions per 100 words)"
    l2 = r"SUS$^{\rm tfidf\text{-}L2}$ (L2-normalised TF-IDF)"
    pairs = ((("SUS_density", dens), ("SUS_tfidf_length", r"SUS$^{\rm tfidf\text{-}\ell}$ (length-scaled TF-IDF)")),
             (("SUS_density", dens), ("SUS_lagasio", l2)),
             (("Breadth", "BREADTH (effective distinct terms)"), ("SUS_lagasio", l2)))
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.3), layout="constrained")
    for ax, ((xc, xl), (yc, yl)), tag in zip(axes, pairs, "abc"):
        ax.scatter(d[xc], d[yc], s=6, color=C1, alpha=0.5, edgecolor="none", rasterized=False)
        r = corr.loc[xc, yc]
        ax.text(0.04, 0.96, f"({tag})  Pearson $r$ = {r:.2f}", transform=ax.transAxes,
                fontsize=7.5, color=INK, va="top")
        ax.set_xlabel(xl, fontsize=7.5)
        ax.set_ylabel(yl, fontsize=7.5)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    axes[0].text(0.96, 0.04, f"n = {len(d)} reports", transform=axes[0].transAxes, fontsize=FS_SMALL,
                 color=INK2, ha="right", va="bottom")
    save(fig, out, "fig_sus_specs.pdf")


def fig_ext_weights(A: Path, out: Path):
    """Rejilla de pesos del ESGSI extendido: Spearman con el adoptado (0,5; 0,5)
    y pendiente con EF de empresa, para w_q (columnas) y w_h (filas)."""
    g = rd(A / "ext_weights_grid.csv")
    wq = sorted(g.w_q.unique())
    wh = sorted(g.w_h.unique(), reverse=True)          # w_h crece hacia arriba
    rho = g.pivot(index="w_h", columns="w_q", values="rho_default").loc[wh, wq]
    slope = g.pivot(index="w_h", columns="w_q", values="pendiente_ef").loc[wh, wq]
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 2.9), layout="constrained")
    from matplotlib.colors import Normalize
    ims = []
    for ax, m, cmap, norm, nd, title in (
            (a, rho, BLUES, Normalize(np.floor(rho.to_numpy().min() * 20) / 20, 1.0), 3,
             "(a) Spearman $\\rho$ with the adopted weights (0.5, 0.5)"),
            (b, -slope, ORANGES, Normalize(0, np.ceil((-slope).to_numpy().max() * 20) / 20), 3,
             "(b) Within-firm trend: slope (per year)")):
        shown = m if ax is a else slope
        im = ax.imshow(m.to_numpy(), cmap=cmap, norm=norm, aspect="auto")
        ax.grid(False)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                txt = fmt(shown.iat[i, j], nd)
                if wq[j] == 0.0 and wh[i] == 0.0:
                    txt += "\n(= ESGSI)"        # (0, 0) es el ESGSI
                ax.text(j, i, txt, ha="center", va="center", fontsize=7.5 if "\n" not in txt else 7.0,
                        color=ink_on(cmap(norm(m.iat[i, j]))), linespacing=1.1)
        ax.set_xticks(range(len(wq)), [f"{w:g}" for w in wq])
        ax.set_yticks(range(len(wh)), [f"{w:g}" for w in wh])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(wq)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(wh)), minor=True)
        ax.grid(which="minor", color="white", lw=1.5)
        ax.tick_params(which="minor", length=0)
        # celda adoptada (0.5, 0.5): borde de tinta; (0, 0) es el ESGSI
        ja, ia = wq.index(0.5), wh.index(0.5)
        ax.add_patch(plt.Rectangle((ja - 0.5, ia - 0.5), 1, 1, fill=False, ec=INK, lw=1.2))
        ax.set_xlabel(r"$w_q$ (weight on $Z$(QUANT), subtracted)")
        ax.set_ylabel(r"$w_h$ (weight on $Z$(HEDGE), added)")
        ax.set_title(title)
    nneg, nsig = int((g.pendiente_ef < 0).sum()), int((g.p_ef < 0.05).sum())
    fig.supxlabel(f"Framed cell: adopted weights. Slopes: firm fixed effects, standard errors clustered by firm "
                  f"({nneg}/{len(g)} negative; {nsig}/{len(g)} with p < 0.05).", fontsize=FS_SMALL, color=INK2)
    save(fig, out, "fig_ext_weights.pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis", type=Path, default=BASE_DIR / "results" / "analysis_v3")
    ap.add_argument("--stability-dir", type=Path, default=BASE_DIR / "results" / "stability_v3")
    ap.add_argument("--out", type=Path, default=BASE_DIR / "paper" / "figures")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    setup_style()
    fig_temporal(a.analysis, a.out)
    fig_industry_year(a.analysis, a.out)
    fig_firms(a.analysis, a.out)
    fig_pillars(a.analysis, a.out)
    fig_countries(a.analysis, a.out)
    # fig_stability es de la tarea A2 (cambian sus entradas): si aun no casan, no para el resto.
    try:
        fig_stability(a.stability_dir, a.out)
    except Exception as e:  # noqa: BLE001
        print(f"  fig_stability.pdf NO regenerada ({type(e).__name__}: {e})")
    fig_sus_specs(a.analysis, a.out)
    fig_ext_weights(a.analysis, a.out)


if __name__ == "__main__":
    main()
