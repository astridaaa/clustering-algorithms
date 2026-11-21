from hier_common import (BEST_DIMENSION, BY_K_FILE, CHOSEN, DETAILS_FILE,
                         DIMENSIONS_FILE, FINAL_SPACES, GRID, LABELS_FILE,
                         LINKAGE_NAMES, LINKAGES, METRIC_NAMES, OUTPUT_DIR,
                         REDUCTIONS, SPACE_NAMES, SPACES, build_tree,
                         config_title, cut_height, load, save_figure)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import dendrogram, set_link_color_palette

from common import load_meta
from config import CLASSES

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED, GRID_INK = "#2b2b2b", "#6b6b6b", "#d9d9d9"
LINKAGE_COLOURS = dict(zip(LINKAGES, SERIES))
METRIC_STYLES = {"euclidean": "-", "cityblock": "--", "chebyshev": ":"}
CLASS_COLOURS = dict(zip(CLASSES, SERIES))
STABILITY_DIR = OUTPUT_DIR / "stabilumas"
SPACE_GENITIVE = {**SPACE_NAMES, "originali": "Pradinės erdvės (5D)"}
SHORT_NAMES = {**SPACE_NAMES, "originali": "Pradinė 5D"}
REDUCTION_NAMES = {"pca": "PCA", "tsne": "t-SNE"}


def is_tsne(space):
    return space.startswith("tsne")

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 10,
                     "axes.spines.top": False, "axes.spines.right": False})


def style_grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID_INK, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_dendrograms(space, details):
    ids, X = load(space)
    chosen = CHOSEN[space]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    heights_ax = axes.ravel()[-1]
    set_link_color_palette(SERIES[:5])

    for ax, method in zip(axes.ravel(), LINKAGES):
        Z = build_tree(X, method, "euclidean")
        row = details[(details["erdve"] == space) & (details["jungimas"] == method)
                      & (details["atstumas"] == "euclidean")
                      & (details["k_saltinis"] == "tarpas")].iloc[0]
        k_gap, k_sil = int(row["k_tarpas"]), int(row["k_siluetas"])
        is_chosen = (method, "euclidean") == (chosen["jungimas"], chosen["atstumas"])
        k_cut = chosen["k"] if is_chosen else k_gap
        cut = cut_height(Z, k_cut)
        dendrogram(Z, truncate_mode="lastp", p=30, ax=ax, no_labels=True,
                   color_threshold=cut, above_threshold_color=MUTED)
        ax.axhline(cut, color="#c0392b" if is_chosen else INK, linestyle="-" if is_chosen else "--",
                   linewidth=1.2, label=f"pjūvis, k = {k_cut}" + (" (pasirinkta)" if is_chosen else ""))
        ax.set_title(f"{LINKAGE_NAMES[method].capitalize()} jungimas\n"
                     f"k pagal tarpą = {k_gap}, pagal siluetą = {k_sil}; "
                     f"didž. klasteris {row['didziausio_dalis']:.1%}")
        ax.set_ylabel("Sujungimo aukštis")
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        style_grid(ax)

        n = len(Z) + 1
        ks = np.arange(2, 17)
        h = Z[n - ks, 2] / Z[-1, 2]
        heights_ax.plot(ks, h, "-o", color=LINKAGE_COLOURS[method], markersize=4,
                        linewidth=1.5, label=LINKAGE_NAMES[method])
        heights_ax.plot([k_gap], [Z[n - k_gap, 2] / Z[-1, 2]], "o", markersize=9,
                        markerfacecolor="none", markeredgecolor=LINKAGE_COLOURS[method])

    heights_ax.set_xlabel("Klasterių skaičius k (prieš sujungimą)")
    heights_ax.set_ylabel("Sujungimo aukštis / didžiausias aukštis")
    heights_ax.set_title("Paskutinių sujungimų aukščiai (Euklido)\n"
                         "apskritimas – k pagal didžiausią tarpą")
    heights_ax.set_xticks(range(2, 17, 2))
    heights_ax.legend(fontsize=8, frameon=False)
    style_grid(heights_ax, "both")

    note = (" – viršutiniai lygiai ir atstumai tarp grupių t-SNE erdvėje neinterpretuotini"
            if is_tsne(space) else "")
    fig.suptitle(f"Dendrogramos (paskutiniai 30 mazgų), {SPACE_NAMES[space]}{note}",
                 fontsize=12)
    fig.tight_layout()
    save_figure(fig, f"dendrogramos_{space}.png")


def plot_silhouette(by_k, details):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for col, space in enumerate(FINAL_SPACES):
        for method, metric in GRID:
            data = by_k[(by_k["erdve"] == space) & (by_k["jungimas"] == method)
                        & (by_k["atstumas"] == metric)]
            k_gap = int(details[(details["erdve"] == space) & (details["jungimas"] == method)
                                & (details["atstumas"] == metric)]["k_tarpas"].iloc[0])
            style = dict(color=LINKAGE_COLOURS[method], linestyle=METRIC_STYLES[metric],
                         linewidth=1.5)
            for row, column in enumerate(["siluetas", "didziausio_dalis"]):
                ax = axes[row, col]
                ax.plot(data["k"], data[column], **style)
                at = data[data["k"] == k_gap]
                ax.plot(at["k"], at[column], "o", markersize=6, color=style["color"])
        axes[0, col].set_title(SPACE_NAMES[space] + (" (siluetas dirbtinai išpūstas)"
                                                    if is_tsne(space) else ""))
        axes[0, col].set_ylabel("Vidutinis siluetas")
        axes[1, col].set_ylabel("Didžiausio klasterio dalis")
        axes[1, col].set_xlabel("Klasterių skaičius k")
        axes[1, col].set_ylim(0, 1.03)
        for ax in axes[:, col]:
            style_grid(ax, "both")
            ax.set_xticks(range(2, 11))

    handles = [Line2D([], [], color=LINKAGE_COLOURS[m], linewidth=2, label=LINKAGE_NAMES[m])
               for m in LINKAGES]
    handles += [Line2D([], [], color=INK, linestyle=s, label=METRIC_NAMES[m])
                for m, s in METRIC_STYLES.items()]
    handles.append(Line2D([], [], marker="o", color=INK, linestyle="none",
                          label="k pagal didžiausią tarpą"))
    fig.legend(handles=handles, loc="lower center", ncol=9, frameon=False, fontsize=9)
    fig.suptitle("Siluetas ir didžiausio klasterio dalis pagal k: aukštas siluetas be Ward "
                 "– tik „milžiniškas klasteris + kelios išskirtys“", fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(fig, "siluetas_pagal_k.png")


def plot_parameters(details, results):
    rows = details[details["k_saltinis"] == "tarpas"].merge(
        results[["erdve", "jungimas", "atstumas", "k", "stabilumas_ARI_vid"]],
        on=["erdve", "jungimas", "atstumas", "k"])
    order = [config_title(m, d) for m, d in GRID]
    rows["konfig"] = [config_title(m, d) for m, d in zip(rows["jungimas"], rows["atstumas"])]
    panels = [("siluetas", "Siluetas (k pagal tarpą)"),
              ("didziausio_dalis", "Didžiausio klasterio dalis"),
              ("ARI", "ARI su pradine erdve\n(tas pats jungimas ir atstumas)"),
              ("stabilumas_ARI_vid", "Stabilumas: ARI vid. 80 % imtyse")]
    fig, axes = plt.subplots(1, 4, figsize=(22, 6.5), sharey=True)
    for ax, (column, title) in zip(axes, panels):
        table = rows.pivot(index="konfig", columns="erdve", values=column).loc[order, SPACES]
        ks = rows.pivot(index="konfig", columns="erdve", values="k").loc[order, SPACES]
        ax.imshow(table.to_numpy(dtype=float), cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for (i, j), value in np.ndenumerate(table.to_numpy(dtype=float)):
            if np.isnan(value):
                text, colour = "–", MUTED
            else:
                value = 0.0 if abs(value) < 0.005 else value
                text = f"{value:.2f}" + (f"\nk={int(ks.iat[i, j])}" if column == "siluetas" else "")
                colour = "white" if value > 0.6 else INK
            ax.text(j, i, text, ha="center", va="center", fontsize=7, color=colour)
        ax.set_xticks(range(len(SPACES)), [SHORT_NAMES[s].replace(" ", "\n") for s in SPACES])
        ax.set_yticks(range(len(order)), order)
        ax.set_title(title)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("Parametrų įtaka: jungimo metodas ir atstumas visose erdvėse", fontsize=12)
    fig.text(0.5, 0.005, "Pastaba: kai didžiausio klasterio dalis ≈ 1, aukštas siluetas, ARI ir "
             "stabilumas reiškia tik tai, kad kas kartą atskiriamos tos pačios kelios išskirtys. "
             "t-SNE siluetas dirbtinai išpūstas; PCA 5D – pradinės erdvės pasukimas.",
             ha="center", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save_figure(fig, "parametrai.png")


def scatter_clusters(ax, coords, labels, outliers, title):
    for label in np.unique(labels):
        mask = labels == label
        colour = SERIES[int(label) % len(SERIES)]
        normal, outlying = mask & ~outliers, mask & outliers
        ax.scatter(coords[normal, 0], coords[normal, 1], s=3, color=colour,
                   rasterized=True, label=f"{int(label)} (n = {mask.sum()})")
        ax.scatter(coords[outlying, 0], coords[outlying, 1], s=14, facecolor=colour,
                   edgecolor=INK, linewidth=0.5, rasterized=True)
    ax.scatter([], [], s=14, facecolor="white", edgecolor=INK, linewidth=0.5,
               label="ekstremali išskirtis")
    ax.set_title(title)
    ax.legend(markerscale=2.5, fontsize=8, frameon=False, loc="best")


def plot_clusters(labels):
    meta = load_meta().set_index("id")
    ids = labels.index.to_numpy()
    outliers = meta.loc[ids, "ekstremali_isskirtis"].to_numpy(dtype=bool)
    pca_ids, pca = load("pca2")
    tsne_ids, tsne = load("tsne2")
    pca = pd.DataFrame(pca, index=pca_ids).loc[ids].to_numpy()
    tsne = pd.DataFrame(tsne, index=tsne_ids).loc[ids].to_numpy()

    def title(space, where):
        c = CHOSEN[space]
        return f"{SPACE_GENITIVE[space]} klasteriai ({config_title(c['jungimas'], c['atstumas'], c['k'])})\n{where}"

    best_pca, best_tsne = BEST_DIMENSION["pca"], BEST_DIMENSION["tsne"]
    pca_where = "PCA projekcija (PC1–PC2)" + (
        "" if best_pca == "pca2" else f"; klasterizuota {SPACE_NAMES[best_pca]}")
    tsne_where = "t-SNE 2D žemėlapis" + (
        "" if best_tsne == "tsne2" else f"; klasterizuota {SPACE_NAMES[best_tsne]}")
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    panels = [(axes[0, 0], "originali", pca, "PCA projekcija (PC1–PC2)"),
              (axes[0, 1], best_pca, pca, pca_where),
              (axes[1, 0], "originali", tsne, "t-SNE 2D žemėlapis"),
              (axes[1, 1], best_tsne, tsne, tsne_where)]
    for ax, space, coords, where in panels:
        scatter_clusters(ax, coords, labels[space].to_numpy(), outliers, title(space, where))
        ax.set_xlabel("PC1" if coords is pca else "t-SNE 1")
        ax.set_ylabel("PC2" if coords is pca else "t-SNE 2")
    fig.suptitle("Pasirinkti hierarchiniai klasteriai; ta pati spalva – suporuotas pradinės "
                 "erdvės klasteris, apvesti taškai – ekstremalios išskirtys", fontsize=12)
    fig.tight_layout()
    save_figure(fig, "klasteriai.png")


def plot_stability():
    pairs = pd.read_csv(STABILITY_DIR / "stabilumas_poros.csv")
    by_n = pd.read_csv(STABILITY_DIR / "stabilumas_pagal_n.csv")
    spaces = [s for s in SPACES if (pairs["erdve"] == s).any()]
    names = [f"{SHORT_NAMES[s]}\nk = {CHOSEN[s]['k']}" + (" ★" if s in FINAL_SPACES[1:] else "")
             for s in spaces]
    colours = {"originali": SERIES[0], "pca": SERIES[1], "tsne": SERIES[2]}
    fig, (left, right) = plt.subplots(1, 2, figsize=(15, 5), width_ratios=(1.5, 1))
    box = left.boxplot([pairs.loc[pairs["erdve"] == s, "ari"] for s in spaces],
                       tick_labels=names, patch_artist=True, widths=0.5,
                       medianprops=dict(color=INK), flierprops=dict(markersize=4))
    for patch, space in zip(box["boxes"], spaces):
        patch.set_facecolor(colours[space.rstrip("0123456789")])
        patch.set_alpha(0.55)
    left.set_ylabel("ARI tarp imčių porų (bendri objektai)")
    left.set_title("Stabilumas: ARI per 80 % imčių poras (Ward, Euklido; ★ – pasirinkta dimensija)")
    left.set_ylim(0, 1.02)
    style_grid(left)

    for space in FINAL_SPACES:
        data = by_n[by_n["erdve"] == space]
        right.errorbar(data["N"], data["vidurkis"], yerr=data["sn"], fmt="-o", capsize=3,
                       color=colours[space.rstrip("0123456789")], linewidth=2, markersize=6,
                       label=SPACE_NAMES[space])
    right.set_xticks([5, 10, 15, 20])
    right.set_xlabel("Imčių skaičius N")
    right.set_ylabel("ARI vidurkis ± SN")
    right.set_title("Imčių pakankamumas, pasirinktos erdvės (t-SNE – N ≤ 10)")
    right.set_ylim(0, 1.02)
    right.legend(frameon=False)
    style_grid(right, "both")
    fig.tight_layout()
    save_figure(fig, "stabilumas.png")


def plot_composition(labels):
    meta = load_meta().set_index("id")
    classes = meta.loc[labels.index, "klase"].to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6), sharex=True)
    for ax, space in zip(axes, FINAL_SPACES):
        table = pd.crosstab(labels[space].to_numpy(), classes)[CLASSES]
        shares = table.div(table.sum(axis=1), axis=0)
        left = np.zeros(len(shares))
        for cls in CLASSES:
            ax.barh(range(len(shares)), shares[cls], left=left, color=CLASS_COLOURS[cls],
                    edgecolor="white", linewidth=1, label=cls, height=0.7)
            left += shares[cls].to_numpy()
        ax.set_yticks(range(len(shares)),
                      [f"{int(c)} (n = {n})" for c, n in zip(shares.index, table.sum(axis=1))])
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("Dalis klasteryje")
        ax.set_title(SPACE_NAMES[space])
    axes[-1].legend(title="Klasė", bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
    fig.suptitle("Klasių sudėtis pasirinktuose klasteriuose (klasės algoritmui neperduotos)",
                 fontsize=12)
    fig.tight_layout()
    save_figure(fig, "klasiu_sudetis.png")


def plot_dimensions():
    dims = pd.read_csv(DIMENSIONS_FILE)
    original = dims[dims["erdve"] == "originali"].iloc[0]
    measures = [("siluetas_orig", "Siluetas pradinėje erdvėje", SERIES[0]),
                ("ARI", "ARI su pradine erdve", SERIES[1]),
                ("stab_ARI_vid", "Stabilumas (ARI vid. ± SN)", SERIES[2])]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, reduction in zip(axes, REDUCTIONS):
        group = dims[dims["mazinimas"] == reduction].sort_values("dimensija")
        for column, label, colour in measures:
            if column not in group or group[column].isna().all():
                continue
            yerr = group["stab_ARI_sn"] if column == "stab_ARI_vid" else None
            ax.errorbar(group["dimensija"], group[column], yerr=yerr, fmt="-o", capsize=3,
                        color=colour, linewidth=2, markersize=6, label=label)
            if column != "ARI" and not np.isnan(original.get(column, np.nan)):
                ax.axhline(original[column], color=colour, linestyle=":", linewidth=1.2)
        best = int(BEST_DIMENSION[reduction][len(reduction):])
        ax.axvspan(best - 0.25, best + 0.25, color=GRID_INK, alpha=0.6, zorder=0,
                   label="pasirinkta dimensija")
        ks = ", ".join(f"{int(r.dimensija)}D: k = {int(r.k)}" for r in group.itertuples())
        ax.set_title(f"{REDUCTION_NAMES[reduction]} (Ward, Euklido; {ks})")
        ax.set_xticks([2, 3, 5])
        ax.set_xlabel("Dimensijų skaičius")
        ax.set_ylim(0, 1.05)
        style_grid(ax, "both")
    axes[0].set_ylabel("Reikšmė")
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    fig.suptitle("Dimensijos pasirinkimas (punktyras – pradinė erdvė; savo erdvės siluetas "
                 "tarp dimensijų nepalyginamas)", fontsize=12)
    fig.tight_layout()
    save_figure(fig, "dimensijos.png")


def main():
    details = pd.read_csv(DETAILS_FILE)
    by_k = pd.read_csv(BY_K_FILE)
    results = pd.read_csv(OUTPUT_DIR / "rezultatai.csv")
    labels = pd.read_csv(LABELS_FILE).set_index("id")

    for space in FINAL_SPACES:
        plot_dendrograms(space, details)
    plot_dimensions()
    plot_silhouette(by_k, details)
    plot_parameters(details, results)
    plot_clusters(labels)
    plot_stability()
    plot_composition(labels)


if __name__ == "__main__":
    main()
