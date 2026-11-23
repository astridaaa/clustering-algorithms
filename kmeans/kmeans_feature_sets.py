from kmeans_common import (ALGORITHM, CHOSEN_K, OUTPUT_DIR, RANDOM_STATE,
                           align_labels, cluster_colours, fit_kmeans,
                           save_figure)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import evaluation as ev
from common import load_meta, load_projection, load_space
from config import CLASSES, FEATURE_SETS, INPUT_FILE

SET_DIR = OUTPUT_DIR / "rinkiniai"
FIGURE_NAME = "rinkiniai.png"

SPACE = "originali"
MAIN = "visi"
SUBSETS = ["ritmo", "formos"]
SETS = [MAIN, *SUBSETS]
K_MAIN = CHOSEN_K[SPACE]
K_RANGE = range(2, 11)
SILHOUETTE_TOLERANCE = 0.01

SET_TITLES = {"visi": "Visi požymiai", "ritmo": "Ritmo požymiai",
              "formos": "Formos požymiai"}
META_COLUMNS = ["klase", "ritmas", "irasas"]
NOTES = "Palyginta su visų požymių k-vidurkių suskirstymu (k = 3)"


def silhouette_by_k(X):
    rows = []
    for k in K_RANGE:
        labels = fit_kmeans(X, k)[0]
        rows.append({"k": k, **ev.internal_metrics(X, labels)})
    return pd.DataFrame(rows)


def choose_k(table):
    """Mažiausias k, kurio silhouette ne daugiau kaip 0,01 mažesnis už didžiausią."""
    best = table["silhouette"].max()
    close = table[table["silhouette"] >= best - SILHOUETTE_TOLERANCE]
    return int(close["k"].min())


def align_to_main(ids, labels, main_labels):
    _, _, pairs, _ = ev.matched_points(ids, labels, ids, main_labels)
    return align_labels(labels, pairs, K_MAIN)


def crosstab(main_labels, labels):
    return pd.crosstab(pd.Series(main_labels, name="visi"),
                       pd.Series(labels, name="rinkinys"))


def meta_nmi(ids, labels):
    meta = load_meta().set_index("id").loc[ids]
    return {f"NMI_{column}": ev.external_metrics(meta[column].to_numpy(), labels)["nmi"]
            for column in META_COLUMNS}


def load_profile_data(ids):
    raw = pd.read_csv(INPUT_FILE).set_index("id").loc[ids]
    assert raw[FEATURE_SETS[MAIN]].notna().all().all(), "Trūksta požymių reikšmių"
    return raw


def profile(raw, labels):
    rows = []
    for cluster in np.unique(labels):
        members = raw[labels == cluster]
        classes = members["klase"].value_counts(normalize=True)
        row = {"klasteris": cluster, "dydis": len(members)}
        row.update({f"{name}_%": classes.get(name, 0) * 100 for name in CLASSES})
        row.update({f"{feature}_mediana": members[feature].median()
                    for feature in FEATURE_SETS[MAIN]})
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_set(feature_set, ids, labels, X, main_labels):
    return {"rinkinys": feature_set,
            "pozymiai": ", ".join(FEATURE_SETS[feature_set]),
            "k": ev.n_clusters(labels),
            **ev.internal_metrics(X, labels),
            **{f"{name}_su_visais": value
               for name, value in ev.external_metrics(main_labels, labels).items()},
            **meta_nmi(ids, labels)}


def plot_sets(projection_ids, Y, ids, labels_by_set):
    position = pd.Series(np.arange(len(projection_ids)), index=projection_ids).loc[ids]
    Y = Y[position.to_numpy()]
    fig, axes = plt.subplots(1, len(SETS), figsize=(15, 5), sharex=True, sharey=True)
    for ax, feature_set in zip(axes, SETS):
        labels = labels_by_set[feature_set]
        names = np.unique(labels)
        colours = cluster_colours(max(names) + 1)
        for name in names:
            mask = labels == name
            ax.scatter(Y[mask, 0], Y[mask, 1], s=3, color=colours[name],
                       label=f"{name} (n = {mask.sum()})", rasterized=True)
        ax.set_title(f"{SET_TITLES[feature_set]}, k = {len(names)}", fontsize=11)
        ax.set_xlabel("PC1")
        ax.grid(alpha=0.3)
        ax.legend(title="Klasteris", markerscale=4, fontsize=8, title_fontsize=8,
                  loc="lower right", frameon=True, framealpha=0.9)
    axes[0].set_ylabel("PC2")
    fig.suptitle("k-vidurkių klasteriai pagal požymių rinkinius "
                 "(visų požymių PCA 2D projekcija)", fontsize=13)
    fig.tight_layout()
    save_figure(fig, FIGURE_NAME)


def main():
    SET_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)

    ids, X_main = load_space(SPACE, MAIN)
    main_labels = fit_kmeans(X_main, K_MAIN)[0]
    spaces = {MAIN: X_main}
    labels_by_set = {MAIN: main_labels}

    for feature_set in SUBSETS:
        set_ids, X = load_space(SPACE, feature_set)
        assert np.array_equal(set_ids, ids), "Rinkinių objektai turi sutapti"
        table = silhouette_by_k(X)
        table.round(4).to_csv(SET_DIR / f"siluetas_{feature_set}.csv", index=False)
        k = choose_k(table)
        values = ", ".join(f"{row.k}: {row.silhouette:.3f}" for row in table.itertuples())
        print(f"{SET_TITLES[feature_set]}: k = {k}  (silhouette pagal k – {values})")

        labels = align_to_main(ids, fit_kmeans(X, k)[0], main_labels)
        spaces[feature_set] = X
        labels_by_set[feature_set] = labels
        crosstab(main_labels, labels).to_csv(SET_DIR / f"palyginimas_{feature_set}.csv")

        ev.log_experiment(ALGORITHM, SPACE, {"k": k}, labels, random_state=RANDOM_STATE,
                          feature_set=feature_set, internal=ev.internal_metrics(X, labels),
                          external=ev.external_metrics(main_labels, labels), notes=NOTES)

    raw = load_profile_data(ids)
    summary = []
    for feature_set in SETS:
        labels = labels_by_set[feature_set]
        summary.append(evaluate_set(feature_set, ids, labels, spaces[feature_set],
                                    main_labels))
        table = profile(raw, labels)
        table.round(3).to_csv(SET_DIR / f"profilis_{feature_set}.csv", index=False)
        print(f"\n{SET_TITLES[feature_set]} – klasterių profilis:")
        print(table.round(1).to_string(index=False))

    for feature_set in SUBSETS:
        print(f"\nVisų požymių klasteriai (eilutės) ir {feature_set} klasteriai (stulpeliai):")
        print(crosstab(main_labels, labels_by_set[feature_set]).to_string())

    summary = pd.DataFrame(summary)
    summary.round(4).to_csv(SET_DIR / "rinkiniai.csv", index=False)
    print("\nSuvestinė:")
    print(summary.drop(columns="pozymiai").round(3).to_string(index=False))

    projection_ids, Y = load_projection("pca2")
    plot_sets(projection_ids, Y, ids, labels_by_set)
    print(f"Išsaugota: {SET_DIR.relative_to(OUTPUT_DIR.parent.parent)}/")


if __name__ == "__main__":
    main()
