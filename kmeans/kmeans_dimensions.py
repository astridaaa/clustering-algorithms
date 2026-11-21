from kmeans_common import (ALGORITHM, CHOSEN_K, DIMENSIONS_FILE, OUTPUT_DIR,
                           RANDOM_STATE, REDUCTIONS, REFERENCE, SPACE_NAMES,
                           SPACES, align_labels, background_map, dimension_of,
                           fit_kmeans, load, plot_clusters, reduction_of,
                           save_figure, subsample_runs)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import evaluation as ev

PARAMS = {"init": "k-means++", "n_init": 10, "atstumas": "euclidean"}
TOLERANCE = 0.01
NOT_CANDIDATES = {"pca5"}

REDUCTION_NAMES = {"pca": "PCA", "tsne": "t-SNE"}
SERIES = [("silhouette_orig", "Siluetas pradinėje erdvėje", "#2a78d6"),
          ("ari", "ARI su pradine erdve", "#eb6834"),
          ("stab_ari", "Stabilumas (ARI vid. ± SN)", "#1baf7a")]


def evaluate_space(space, X_original, reference):
    ids, X = load(space)
    k = CHOSEN_K[space]
    labels = fit_kmeans(X, k)[0]

    internal = ev.internal_metrics(X, labels)
    internal_original = internal if space == REFERENCE else ev.internal_metrics(X_original, labels)
    if reference is None:
        external = {"ari": np.nan, "nmi": np.nan, "jaccard": np.nan}
        matched, share = len(labels), 1.0
        names = np.unique(labels)
        pairs = pd.DataFrame({"atskaitos_klasteris": names, "klasteris": names})
    else:
        external = ev.compare_with_reference(ids, labels, *reference)
        matched, share, pairs, _ = ev.matched_points(ids, labels, *reference)

    stability_pairs, _ = ev.stability(subsample_runs(space, k))
    ev.log_experiment(ALGORITHM, space, {"k": k, **PARAMS}, labels,
                      random_state=RANDOM_STATE, internal=internal,
                      internal_original=None if space == REFERENCE else internal_original,
                      external=None if reference is None else external,
                      stability_pairs=stability_pairs,
                      notes="dimensijų palyginimas" + ("" if reference is None
                                                       else "; išoriniai matai – su pradine erdve"))

    stab = ev.summarize(stability_pairs, ["ari", "nmi", "jaccard"])
    row = {"erdve": space, "mazinimas": reduction_of(space), "dimensija": dimension_of(space),
           "k": k, **internal,
           **{f"{name}_orig": value for name, value in internal_original.items()},
           **external, "sutampa_tasku": matched, "sutampa_dalis": share,
           **{f"stab_{name}": stab[name][0] for name in stab},
           **{f"stab_{name}_sn": stab[name][1] for name in stab},
           "klasteriu_dydziai": " / ".join(str(n) for n in np.bincount(labels))}
    return ids, labels, pairs, row


def recommend_dimension(table):
    table = table.copy()
    table["rekomenduojama"] = False
    for reduction in REDUCTIONS:
        group = table[(table["mazinimas"] == reduction) & ~table["erdve"].isin(NOT_CANDIDATES)]
        close = group[group["silhouette_orig"] >= group["silhouette_orig"].max() - TOLERANCE]
        stable = close[close["stab_ari"] >= close["stab_ari"].max() - TOLERANCE]
        table.loc[stable["dimensija"].idxmin(), "rekomenduojama"] = True
    return table


def plot_dimensions(table):
    original = table[table["erdve"] == REFERENCE].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, reduction in zip(axes, REDUCTIONS):
        group = table[table["mazinimas"] == reduction].sort_values("dimensija")
        for column, label, colour in SERIES:
            yerr = group["stab_ari_sn"] if column == "stab_ari" else None
            ax.errorbar(group["dimensija"], group[column], yerr=yerr, fmt="-o", capsize=3,
                        color=colour, linewidth=2, markersize=6, label=label)
            if column != "ari":
                ax.axhline(original[column], color=colour, linestyle=":", linewidth=1.2)
        best = int(group.loc[group["rekomenduojama"], "dimensija"].iloc[0])
        ax.axvspan(best - 0.25, best + 0.25, color="#e5e4de", alpha=0.8, zorder=0,
                   label="pasirinkta dimensija")
        ks = ", ".join(f"{int(r.dimensija)}D: k = {int(r.k)}" for r in group.itertuples())
        ax.set_title(f"{REDUCTION_NAMES[reduction]} ({ks})")
        ax.set_xticks([2, 3, 5])
        ax.set_xlabel("Dimensijų skaičius")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Reikšmė")
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    fig.suptitle("k-vidurkiai: dimensijos pasirinkimas (punktyras – pradinė erdvė)", fontsize=12)
    fig.tight_layout()
    save_figure(fig, "dimensijos.png")


def plot_clusters_by_dimension(results, background):
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    for ax, space in zip(axes.ravel(), SPACES):
        ids, aligned = results[space]
        plot_clusters(ax, ids, aligned, f"{SPACE_NAMES[space]}, k = {CHOSEN_K[space]}",
                      background)
    for ax in axes.ravel()[len(SPACES):]:
        ax.set_visible(False)
    fig.suptitle("k-vidurkių klasteriai visose erdvėse (bendras t-SNE žemėlapis; "
                 "ta pati spalva – suporuotas klasteris)", fontsize=13)
    fig.tight_layout()
    save_figure(fig, "dimensijos_klasteriai.png")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _, X_original = load(REFERENCE)
    background = background_map()

    rows, results, reference = [], {}, None
    for space in SPACES:
        print(f"{SPACE_NAMES[space]}, k = {CHOSEN_K[space]}...")
        ids, labels, pairs, row = evaluate_space(space, X_original, reference)
        if reference is None:
            reference = (ids, labels)
        results[space] = (ids, align_labels(labels, pairs, CHOSEN_K[REFERENCE]))
        rows.append(row)

    table = recommend_dimension(pd.DataFrame(rows))
    table.round(4).to_csv(DIMENSIONS_FILE, index=False)

    pd.set_option("display.width", 250)
    print("\nDimensijų palyginimas (išoriniai matai – su pradine erdve, k = "
          f"{CHOSEN_K[REFERENCE]}):")
    print(table[["erdve", "k", "silhouette", "silhouette_orig", "ari", "nmi", "jaccard",
                 "sutampa_dalis", "stab_ari", "stab_ari_sn", "klasteriu_dydziai",
                 "rekomenduojama"]].round(3).to_string(index=False))
    best = table[table["rekomenduojama"]]
    print("\nPasirinktos dimensijos: " + ", ".join(
        f"{REDUCTION_NAMES[r.mazinimas]} – {r.dimensija}D" for r in best.itertuples()))
    print("PCA 5D – tik pradinės erdvės pasukimas (Euklido atstumai nepakinta), todėl "
          "jos rezultatai sutampa su pradinės erdvės.")
    print("Pastaba: t-SNE vidiniai matai dirbtinai dideli – dimensijos lyginamos pagal "
          "silhouette_orig, ARI su pradine erdve ir stabilumą.")

    plot_dimensions(table)
    plot_clusters_by_dimension(results, background)
    print("Išsaugota: rezultatai/kmeans/dimensijos.csv")


if __name__ == "__main__":
    main()