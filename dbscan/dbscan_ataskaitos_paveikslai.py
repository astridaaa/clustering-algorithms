from dbscan_common import (FIGURES_DIR, MIN_SAMPLES, NOISE_COLOUR, OTHER_COLOUR, OUTPUT_DIR,
                           PALETTE, SPACE_NAMES, background_map, k_distances, knee_index,
                           load, load_chosen)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

import evaluation as ev
from common import load_meta
from config import CLASS_COLUMN, CLASSES, FEATURES, INPUT_FILE
from evaluation import NOISE

REPORT_DIR = FIGURES_DIR / "ataskaitai"
FINAL_DIR = OUTPUT_DIR / "galutiniai"

CHOSEN = ["originali", "pca3", "tsne2"]            # ataskaitoje pasirinktos erdvės
SCANNED = ["originali", "pca2", "pca3", "tsne2", "tsne3", "tsne5"]  # PCA 5D = pradinė erdvė
VARIANT_NAMES = {"kelio": "kelio metodas", "k5": "5 klasteriai"}

MIN_SAMPLES_GRID = [5, 10, 20]
MIN_SAMPLES_COLOURS = {5: "#4c72b0", 10: "#c44e52", 20: "#55a868"}
EPS_FACTORS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
KNEE_STYLE = {"color": "#6b6b6b", "linestyle": "--", "linewidth": 1.1}
K5_STYLE = {"color": "#1f1f1f", "linestyle": "-.", "linewidth": 1.1}
ZOOM_FROM = 80  # 5 pav. rodomi paskutiniai 20 % taškų: visi keliai yra ties 86–98 %
MIN_SHARED = 0.5  # 14 pav. spalva suporuojama, kai bendri bent 50 % mažesniojo klasterio
HEADROOM = 0.12  # 14 pav. tuščia juosta viršuje legendai
SMALL = 100  # 14 pav. mažesni klasteriai paryškinami ir sunumeruojami
TOP_CLUSTERS = 6  # 19 pav. t-SNE 2D: šeši didžiausi klasteriai, kiti sujungiami
CLASS_COLOURS = {"N": "#4c72b0", "L": "#dd8452", "R": "#55a868", "V": "#c44e52", "A": "#8172b3"}
UNITS = {"RR_sant_vid": "", "RR_pre": "ms", "RR_sant_post": "", "QRS": "ms", "QTc": "ms"}
SPACE_COLOURS = dict(zip(CHOSEN, PALETTE))  # kaip hierarchinio stabilumo paveiksle

EPS = r"$\mathit{Eps}$"
EPS_K = r"$\mathit{Eps}_k$"
MINPTS = r"$\mathit{MinPts}$"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5,
                     "axes.labelsize": 9, "legend.fontsize": 8.5,
                     "xtick.labelsize": 8, "ytick.labelsize": 8})


def comma(value, digits=3):
    return f"{value:.{digits}f}".replace(".", ",")


COMMA_TICKS = FuncFormatter(lambda value, _: f"{value:g}".replace(".", ","))


def style(ax, grid_axis="both"):
    ax.grid(alpha=0.3, axis=grid_axis)
    ax.spines[["top", "right"]].set_visible(False)


def save(fig, name):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / name
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Išsaugota: {path}")


# 5 pav. ---------------------------------------------------------------------------------
def plot_k_distances(chosen):
    """k-atstumų kreivių pabaiga visose erdvėse (be PCA 5D, kuri sutampa su pradine)."""
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 6.4))
    for ax, space in zip(axes.flat, SCANNED):
        _, X = load(space)
        for min_samples in MIN_SAMPLES_GRID:
            distances = k_distances(X, min_samples)
            share = np.linspace(0, 100, len(distances))
            keep = share >= ZOOM_FROM
            knee = knee_index(distances)
            ax.plot(share[keep], distances[keep], color=MIN_SAMPLES_COLOURS[min_samples],
                    linewidth=1.6)
            ax.plot(share[knee], distances[knee], "o", markersize=5, markerfacecolor="white",
                    markeredgecolor="#1f1f1f", markeredgewidth=1.1, zorder=3)
            if min_samples == MIN_SAMPLES:
                top = np.quantile(distances[keep], 0.95) * 1.5
        eps = chosen[(space, "kelio")]
        ax.axhline(eps, **KNEE_STYLE)
        ax.set_xlim(ZOOM_FROM, 100)
        ax.set_ylim(0, top)
        ax.yaxis.set_major_formatter(COMMA_TICKS)
        ax.set_title(f"{SPACE_NAMES[space]}, {EPS} = {comma(eps)}")
        style(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("$k$-atstumas")
    for ax in axes[-1]:
        ax.set_xlabel("Taškai, surikiuoti pagal $k$-atstumą, %")
    handles = [Line2D([], [], color=MIN_SAMPLES_COLOURS[m], linewidth=1.6) for m in MIN_SAMPLES_GRID]
    names = [f"{MINPTS} = {m}" for m in MIN_SAMPLES_GRID]
    handles += [Line2D([], [], marker="o", linestyle="", markerfacecolor="white",
                       markeredgecolor="#1f1f1f"), Line2D([], [], **KNEE_STYLE)]
    names += ["kelias", f"kelio metodu parinktas {EPS} ({MINPTS} = {MIN_SAMPLES})"]
    fig.legend(handles, names, loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "dbscan_k_atstumai.png")


# 6 pav. ---------------------------------------------------------------------------------
def plot_eps_scan(scan, chosen):
    """Klasterių skaičius ir triukšmo dalis pagal Eps (MinPts = 10): du blokai po 3 erdves."""
    fig = plt.figure(figsize=(9.5, 10.4), layout="constrained")
    blocks = fig.subfigures(2, 1, hspace=0.03)
    for block, (subfig, spaces, k_max) in enumerate(zip(blocks, [SCANNED[:3], SCANNED[3:]], [40, 200])):
        axes = subfig.subplots(2, 3, sharex="col", height_ratios=[1.25, 1])
        for col, space in enumerate(spaces):
            top, bottom = axes[0, col], axes[1, col]
            data = scan[scan["erdve"] == space]
            colour = MIN_SAMPLES_COLOURS[MIN_SAMPLES]
            top.plot(data["eps"], data["k"], color=colour, linewidth=1.4)
            bottom.plot(data["eps"], data["triuksmo_dalis"] * 100, color=colour, linewidth=1.4)
            top.axhline(5, color="#9a9a9a", linestyle=":", linewidth=1.1)
            for ax in (top, bottom):
                ax.axvline(chosen[(space, "kelio")], **KNEE_STYLE)
                ax.axvline(chosen[(space, "k5")], **K5_STYLE)
                style(ax)
            top.set_yscale("log")
            top.set_ylim(0.8, k_max)
            ticks = [t for t in [1, 2, 5, 10, 20, 50, 100] if t < k_max]
            top.set_yticks(ticks)
            top.set_yticklabels([str(t) for t in ticks])
            top.minorticks_off()
            bottom.set_ylim(0, 100)
            bottom.xaxis.set_major_formatter(COMMA_TICKS)
            bottom.set_xlabel(EPS, labelpad=1)
            top.set_title(SPACE_NAMES[space])
            if col == 0:
                top.set_ylabel("Klasterių skaičius")
                bottom.set_ylabel("Triukšmas, %")
    handles = [Line2D([], [], color=MIN_SAMPLES_COLOURS[MIN_SAMPLES], linewidth=1.4),
               Line2D([], [], **KNEE_STYLE), Line2D([], [], **K5_STYLE),
               Line2D([], [], color="#9a9a9a", linestyle=":", linewidth=1.1)]
    names = [f"{MINPTS} = {MIN_SAMPLES}", f"kelio metodu parinktas {EPS}",
             f"{EPS}, parinktas 5 klasteriams", "5 klasteriai"]
    fig.legend(handles, names, loc="outside lower center", ncol=4, frameon=False)
    save(fig, "dbscan_k_pagal_eps.png")


# 13 pav. --------------------------------------------------------------------------------
LINE_PANELS = [("k", "Klasterių skaičius"),
               ("triuksmo_dalis_%", "Triukšmas, %"),
               ("didziausio_dalis_%", "Didžiausio klasterio dalis, %"),
               ("ari_klase", "ARI su dūžių klasėmis")]


def plot_parameters(table):
    """Eps ir MinPts įtaka: eilutėse – matai, stulpeliuose – erdvės, linijos – MinPts."""
    fig, axes = plt.subplots(len(LINE_PANELS), len(CHOSEN), figsize=(9.5, 9.6),
                             sharex=True, sharey="row")
    for col, space in enumerate(CHOSEN):
        data = table[table["erdve"] == space]
        knee = data.loc[data["eps_daugiklis"] == 1.0, "eps"].iloc[0]
        axes[0, col].set_title(f"{SPACE_NAMES[space]}, {EPS_K} = {comma(knee)}")
        for row, (column, label) in enumerate(LINE_PANELS):
            ax = axes[row, col]
            for min_samples in MIN_SAMPLES_GRID:
                line = data[data["min_samples"] == min_samples].sort_values("eps_daugiklis")
                ax.plot(line["eps_daugiklis"], line[column], "-o", markersize=3.5, linewidth=1.5,
                        color=MIN_SAMPLES_COLOURS[min_samples])
            ax.axvline(1.0, **KNEE_STYLE)
            style(ax)
            if col == 0:
                ax.set_ylabel(label)
            if row == len(LINE_PANELS) - 1:
                ax.set_xlabel(f"{EPS} / {EPS_K}")
    axes[0, 0].set_yscale("symlog", linthresh=10)
    axes[0, 0].set_ylim(0, 250)
    axes[0, 0].set_yticks([0, 5, 10, 20, 50, 100, 200])
    axes[0, 0].set_yticklabels(["0", "5", "10", "20", "50", "100", "200"])
    axes[0, 0].minorticks_off()
    for row in (1, 2):
        axes[row, 0].set_ylim(0, 102)
    axes[-1, 0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: comma(v, 2)))
    axes[-1, 0].set_xticks(EPS_FACTORS)
    axes[-1, 0].xaxis.set_major_formatter(COMMA_TICKS)
    handles = [Line2D([], [], color=MIN_SAMPLES_COLOURS[m], marker="o", markersize=3.5,
                      linewidth=1.5) for m in MIN_SAMPLES_GRID]
    names = [f"{MINPTS} = {m}" for m in MIN_SAMPLES_GRID]
    handles.append(Line2D([], [], **KNEE_STYLE))
    names.append(f"kelio metodu parinktas {EPS} ({EPS_K})")
    fig.legend(handles, names, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    save(fig, "dbscan_parametrai.png")


# 14 pav. --------------------------------------------------------------------------------
def report_labels(labels, reference):
    """Numeriai (ir spalvos) 14 pav.: klasteris gauna atskaitos klasterio numerį, jei su juo
    suporuotas vienas su vienu (kaip 7 lentelėje) ir bent pusė mažesniojo iš jų objektų yra
    bendri. Kiti klasteriai numeruojami nuo didžiausio atskaitos numerio + 1 dydžio mažėjimo
    tvarka, todėl jų spalvos nesutampa su atskaitos klasterių spalvomis."""
    ids = labels.index.to_numpy()
    _, _, pairs, _ = ev.matched_points(ids, labels.to_numpy(), ids, reference.to_numpy())
    shared = pairs["sutampa"] >= MIN_SHARED * np.minimum(pairs["atskaitos_dydis"], pairs["dydis"])
    pairs = pairs[shared & (pairs["klasteris"] != NOISE)]
    mapping = {NOISE: NOISE, **dict(zip(pairs["klasteris"], pairs["atskaitos_klasteris"]))}
    values = labels.to_numpy()
    next_label = int(reference.max()) + 1
    for name in sorted(set(np.unique(values)) - set(mapping), key=lambda n: -(values == n).sum()):
        mapping[name] = next_label
        next_label += 1
    return labels.map(mapping)


def plot_clusters(ax, points, labels, title):
    """Kaip dbscan_common.plot_clusters: kai klasterių daugiau nei 8, spalvoti 7 pirmi
    numeriai, kiti – šviesiai pilki; triukšmas tamsiai pilkas. Maži klasteriai (< SMALL
    objektų) piešiami paskutiniai, didesniais taškais ir su numeriu šalia."""
    x, y = points["x"].to_numpy(), points["y"].to_numpy()
    labels = labels.to_numpy()
    names = [name for name in np.unique(labels) if name != NOISE]
    limit = len(PALETTE) - 1 if len(names) > len(PALETTE) else len(PALETTE)
    coloured = [name for name in names if name < limit]
    small = [name for name in coloured if (labels == name).sum() < SMALL]
    handles = {}
    for name in coloured:
        if name in small:
            continue
        mask = labels == name
        handles[name] = ax.scatter(x[mask], y[mask], s=2, color=PALETTE[name], rasterized=True,
                                   label=f"{name} ({mask.sum()})")
    other = (labels != NOISE) & (labels >= limit)
    if other.any():
        handles["kiti"] = ax.scatter(x[other], y[other], s=2, color=OTHER_COLOUR, rasterized=True,
                                     label=f"kiti {len(np.unique(labels[other]))} kl. ({other.sum()})")
    noise = labels == NOISE
    if noise.any():
        handles["triukšmas"] = ax.scatter(x[noise], y[noise], s=3, color=NOISE_COLOUR,
                                          rasterized=True, label=f"triukšmas ({noise.sum()})")
    span = x.max() - x.min()
    for name in small:
        mask = labels == name
        handles[name] = ax.scatter(x[mask], y[mask], s=9, color=PALETTE[name], edgecolors="white",
                                   linewidths=0.3, rasterized=True, zorder=3,
                                   label=f"{name} ({mask.sum()})")
        at_right = np.median(x[mask]) > x.min() + 0.9 * span  # prie dešinio krašto – numeris kairėje
        text = ax.annotate(str(name), (np.median(x[mask]), np.median(y[mask])),
                           xytext=(-5 if at_right else 4, 3), textcoords="offset points",
                           ha="right" if at_right else "left", fontsize=7.5,
                           fontweight="bold", color="#1f1f1f", zorder=4)
        text.set_path_effects([patheffects.withStroke(linewidth=2, foreground="white")])
    order = sorted((key for key in handles if isinstance(key, (int, np.integer))), key=int)
    order += [key for key in ("kiti", "triukšmas") if key in handles]
    proxies = [Line2D([], [], marker="o", linestyle="", markersize=5,
                      color=handles[key].get_facecolor()[0]) for key in order]
    ax.set_title(title, fontsize=8.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(x.min() - 0.02 * span, x.max() + 0.02 * span)
    height = y.max() - y.min()
    ax.set_ylim(y.min() - 0.02 * height, y.max() + HEADROOM * height)  # vieta legendai
    crowded = len(order) > 7  # t-SNE 2D kelio metodas: tankesnė legenda, kad neuždengtų salų
    ax.legend(proxies, [handles[key].get_label() for key in order], loc="upper left",
              fontsize=6.4 if crowded else 7, labelspacing=0.08 if crowded else 0.25,
              frameon=False, handletextpad=0.1, borderaxespad=0.15)


def plot_final(labels, results):
    """Pasirinkti suskirstymai bendrame t-SNE 2D žemėlapyje: viršuje kelio metodas,
    apačioje 5 klasteriai."""
    background = background_map().loc[labels.index]
    fig, axes = plt.subplots(2, len(CHOSEN), figsize=(8.6, 6.5))
    for row, variant in enumerate(VARIANT_NAMES):
        reference = labels[f"originali_{variant}"]
        for col, space in enumerate(CHOSEN):
            run = f"{space}_{variant}"
            info = results.loc[run]
            column = labels[run] if space == "originali" else report_labels(labels[run], reference)
            title = (f"{SPACE_NAMES[space]}, {VARIANT_NAMES[variant]}:\n{EPS} = {comma(info['eps'])}, "
                     f"{MINPTS} = {MIN_SAMPLES}, $k$ = {int(info['k'])}")
            plot_clusters(axes[row, col], background, column, title)
    fig.tight_layout(h_pad=1.2, w_pad=0.6)
    save(fig, "dbscan_klasteriai.png")


# 17 pav. --------------------------------------------------------------------------------
def plot_features(data, labels, run="originali_kelio"):
    """Požymių stačiakampės diagramos pradinės erdvės klasteriuose ir triukšme."""
    clusters = sorted(c for c in labels[run].unique() if c != NOISE) + [NOISE]
    colours = [NOISE_COLOUR if c == NOISE else PALETTE[c] for c in clusters]
    names = ["triukšmas" if c == NOISE else str(c) for c in clusters]
    fig, axes = plt.subplots(2, 3, figsize=(9, 5.6))
    for ax, feature in zip(axes.flat, FEATURES):
        values = [data.loc[labels[run] == c, feature] for c in clusters]
        parts = ax.boxplot(values, tick_labels=names, patch_artist=True, showfliers=False,
                           widths=0.55, medianprops={"color": "#0b0b0b"})
        for patch, colour in zip(parts["boxes"], colours):
            patch.set_facecolor(colour)
            patch.set_alpha(0.6)
        unit = f", {UNITS[feature]}" if UNITS[feature] else ""
        ax.set_title(f"{feature}{unit}")
        ax.yaxis.set_major_formatter(COMMA_TICKS)
        style(ax, "y")
    for ax in axes[-1]:
        ax.set_xlabel("Klasteris")
    for ax in axes.flat[len(FEATURES):]:
        ax.axis("off")
    fig.tight_layout()
    save(fig, "dbscan_pozymiai.png")


# 19 pav. --------------------------------------------------------------------------------
def composition_table(classes, labels, top=None):
    """Klasių dalys (%) kiekviename klasteryje; jei nurodyta top – tik tiek didžiausių
    klasterių, o kiti sujungiami į vieną eilutę."""
    sizes = labels.value_counts()
    clusters = [c for c in sizes.index if c != NOISE]
    keep = clusters if top is None else clusters[:top]
    group = labels.map(lambda c: c if c == NOISE or c in keep else "kiti")
    table = pd.crosstab(group, classes, normalize="index").reindex(columns=CLASSES, fill_value=0)
    counts = group.value_counts()
    order = keep + (["kiti"] if len(keep) < len(clusters) else []) + \
        ([NOISE] if NOISE in counts.index else [])
    names = []
    for c in order:
        if c == NOISE:
            names.append(f"triukšmas ({counts[c]})")
        elif c == "kiti":
            names.append(f"kiti {len(clusters) - len(keep)} kl. ({counts[c]})")
        else:
            names.append(f"{c} ({counts[c]})")
    return table.loc[order] * 100, names


def plot_composition(meta, labels, variant="kelio"):
    tables = [composition_table(meta[CLASS_COLUMN], labels[f"{space}_{variant}"],
                                TOP_CLUSTERS if space.startswith("tsne") else None)
              for space in CHOSEN]
    heights = [len(table) for table, _ in tables]
    fig, axes = plt.subplots(len(CHOSEN), 1, figsize=(8, 0.3 * sum(heights) + 1.6), sharex=True,
                             gridspec_kw={"height_ratios": heights})
    for ax, space, (table, names) in zip(axes, CHOSEN, tables):
        positions = np.arange(len(table))
        left = np.zeros(len(table))
        for name in CLASSES:
            values = table[name].to_numpy()
            ax.barh(positions, values, left=left, height=0.72, color=CLASS_COLOURS[name],
                    edgecolor="white", linewidth=0.6, label=name)
            left += values
        ax.set_yticks(positions)
        ax.set_yticklabels(names)
        ax.set_ylim(len(table) - 0.5, -0.5)
        ax.set_xlim(0, 100)
        ax.set_title(SPACE_NAMES[space], loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.25, axis="x")
        ax.set_axisbelow(True)
    axes[-1].set_xlabel("Dūžių klasių dalis klasteryje, %")
    handles, names = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    fig.legend(handles, names, title="Klasė", frameon=False, loc="upper left",
               bbox_to_anchor=(0.9, 0.97))
    save(fig, "dbscan_klases.png")


def plot_stability(pairs, by_n, variant="kelio"):
    """Stabilumas per 80 % imtis: ARI tarp imčių porų ir ARI vidurkis pagal imčių skaičių."""
    spaces = [space for space in CHOSEN if f"{space}_{variant}" in set(pairs["suskirstymas"])]
    fig, (left, right) = plt.subplots(1, 2, figsize=(8, 3.3), layout="constrained",
                                      gridspec_kw={"width_ratios": [1, 1.1]})
    values = [pairs.loc[pairs["suskirstymas"] == f"{space}_{variant}", "ari"] for space in spaces]
    boxes = left.boxplot(values, widths=0.5, patch_artist=True, showmeans=True,
                         medianprops={"color": "#1f1f1f", "linewidth": 1.4},
                         meanprops={"marker": "D", "markersize": 4.5, "markerfacecolor": "white",
                                    "markeredgecolor": "#1f1f1f"},
                         whiskerprops={"color": "#4a4a4a"}, capprops={"color": "#4a4a4a"},
                         flierprops={"marker": "o", "markersize": 3, "markerfacecolor": "none",
                                     "markeredgecolor": "#4a4a4a"})
    for patch, space in zip(boxes["boxes"], spaces):
        patch.set_facecolor(SPACE_COLOURS[space])
        patch.set_alpha(0.55)
        patch.set_edgecolor("#4a4a4a")
    n_samples = by_n.groupby("suskirstymas")["N"].max()
    left.set_xticks(range(1, len(spaces) + 1),
                    [f"{SPACE_NAMES[space]}\n({n_samples[f'{space}_{variant}']} imčių)"
                     for space in spaces])
    left.set_ylabel("ARI tarp imčių porų")
    left.set_title("ARI tarp 80 % imčių porų")

    offsets = np.linspace(-0.35, 0.35, len(spaces))
    for offset, space in zip(offsets, spaces):
        data = by_n[by_n["suskirstymas"] == f"{space}_{variant}"]
        right.errorbar(data["N"] + offset, data["vidurkis"], yerr=data["sn"], fmt="o-",
                       markersize=4.5, linewidth=1.6, capsize=3, color=SPACE_COLOURS[space],
                       label=SPACE_NAMES[space])
    right.set_xticks([5, 10, 15, 20])
    right.set_xlabel("Imčių skaičius N")
    right.set_ylabel("ARI vidurkis ± SN")
    right.set_title("ARI vidurkis pagal imčių skaičių")
    right.legend(loc="lower right", frameon=False)

    for ax in (left, right):
        ax.set_ylim(0.5, 1.02)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: comma(value, 1)))
        style(ax, "y")
    save(fig, "dbscan_stabilumas.png")


def main():
    chosen = load_chosen()
    labels = pd.read_csv(FINAL_DIR / "zymes.csv").set_index("id")
    results = pd.read_csv(FINAL_DIR / "galutiniai_rezultatai.csv").set_index("suskirstymas")
    meta = load_meta().set_index("id").loc[labels.index]
    features = pd.read_csv(INPUT_FILE).set_index("id").loc[labels.index, FEATURES]

    plot_k_distances(chosen)
    plot_eps_scan(pd.read_csv(OUTPUT_DIR / "k_pagal_eps.csv"), chosen)
    plot_parameters(pd.read_csv(OUTPUT_DIR / "parametrai.csv"))
    plot_final(labels, results)
    plot_features(features, labels)
    plot_composition(meta, labels)
    plot_stability(pd.read_csv(FINAL_DIR / "stabilumas_poros.csv"),
                   pd.read_csv(FINAL_DIR / "stabilumas_pagal_n.csv"))


if __name__ == "__main__":
    main()
