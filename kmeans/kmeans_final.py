from kmeans_common import (ALGORITHM, CHOSEN_K, OUTPUT_DIR, RANDOM_STATE,
                           SPACE_NAMES, background_map, fit_kmeans,
                           plot_clusters, save_figure)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import evaluation as ev
from common import load_space, load_subsample, subsample_seeds

PARAMS = {"init": "k-means++", "n_init": 10, "atstumas": "euclidean"}

REFERENCE = ("originali", CHOSEN_K["originali"])
RUNS = [REFERENCE,
        ("pca", CHOSEN_K["pca"]),
        ("tsne", CHOSEN_K["tsne"]),
        ("tsne", CHOSEN_K["originali"])]

FINAL_DIR = OUTPUT_DIR / "galutiniai"


def run_name(space, k):
    return f"{space}_k{k}"


def run_title(space, k):
    extra = " (tas pats k kaip pradinėje)" if (space, k) == ("tsne", REFERENCE[1]) else ""
    return f"{SPACE_NAMES[space]}, k = {k}{extra}"


def align_labels(labels, pairs, k_reference):
    """Suporuotiems klasteriams suteikiamas atskaitos klasterio numeris,
    nesuporuoti (kai k didesnis nei atskaitos) gauna numerius k_ref, k_ref + 1, ..."""
    mapping = dict(zip(pairs["klasteris"], pairs["atskaitos_klasteris"]))
    next_label = k_reference
    for label in np.unique(labels):
        if label not in mapping:
            mapping[label] = next_label
            next_label += 1
    return np.array([mapping[label] for label in labels])


def stability(space, k):
    samples = {}
    for seed in subsample_seeds(space):
        ids, X = load_subsample(space, seed)
        samples[seed] = (ids, X, fit_kmeans(X, k, random_state=seed)[0])
    return ev.stability(samples)


def evaluate_run(space, k, X_original, reference):
    ids, X = load_space(space)
    labels = fit_kmeans(X, k)[0]

    internal = ev.internal_metrics(X, labels)
    internal_original = ev.internal_metrics(X_original, labels)

    if reference is None:
        external, matched, share = None, len(labels), 1.0
        pairs = pd.DataFrame({"atskaitos_klasteris": np.unique(labels),
                              "klasteris": np.unique(labels)})
    else:
        ref_ids, ref_labels = reference
        external = ev.compare_with_reference(ids, labels, ref_ids, ref_labels)
        matched, share, pairs, crosstab = ev.matched_points(ids, labels, ref_ids, ref_labels)
        pairs.to_csv(FINAL_DIR / f"sutampa_{run_name(space, k)}.csv", index=False)
        crosstab.to_csv(FINAL_DIR / f"kryzmine_{run_name(space, k)}.csv")

    stability_pairs, stability_samples = stability(space, k)
    ev.log_experiment(ALGORITHM, space, {"k": k, **PARAMS}, labels,
                      random_state=RANDOM_STATE, internal=internal,
                      internal_original=None if space == "originali" else internal_original,
                      external=external, stability_pairs=stability_pairs,
                      notes="galutinis modelis" + ("" if reference is None
                                                   else "; išoriniai matai – su pradine erdve, k = 3"))

    stab = ev.summarize(stability_pairs, ["ari", "nmi", "jaccard"])
    stab_internal = ev.summarize(stability_samples, ["silhouette", "ch", "db"])
    row = {
        "erdve": space, "k": k,
        **internal,
        **{f"{name}_orig": value for name, value in internal_original.items()},
        **(external or {"ari": np.nan, "nmi": np.nan, "jaccard": np.nan}),
        "sutampa_tasku": matched, "sutampa_dalis": share,
        **{f"stab_{name}": ev.format_mean_sd(*value) for name, value in stab.items()},
        **{f"imciu_{name}": ev.format_mean_sd(*value) for name, value in stab_internal.items()},
        "klasteriu_dydziai": " / ".join(str(n) for n in np.bincount(labels)),
    }

    stability_pairs = stability_pairs.assign(variantas=run_title(space, k))
    by_n = ev.stability_vs_n(stability_pairs).assign(variantas=run_title(space, k))
    return ids, labels, pairs, row, stability_pairs, by_n


def plot_final(results, background):
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    for ax, ((space, k), (ids, aligned)) in zip(axes.ravel(), results.items()):
        plot_clusters(ax, ids, aligned, run_title(space, k), background)
    fig.suptitle("Galutiniai k-vidurkių klasteriai (bendras t-SNE žemėlapis; "
                 "ta pati spalva – suporuotas klasteris)", fontsize=13)
    fig.tight_layout()
    save_figure(fig, "galutiniai_klasteriai.png")


def plot_stability(stability_pairs, by_n):
    names = list(dict.fromkeys(stability_pairs["variantas"]))
    fig, (left, right) = plt.subplots(1, 2, figsize=(15, 5))

    left.boxplot([stability_pairs.loc[stability_pairs["variantas"] == name, "ari"]
                  for name in names], tick_labels=[name.replace(", ", "\n", 1) for name in names])
    left.set_ylabel("ARI tarp imčių porų")
    left.set_title("Stabilumas: ARI pasiskirstymas per 80 % imčių poras")
    left.grid(alpha=0.3, axis="y")

    for name in names:
        data = by_n[by_n["variantas"] == name]
        right.errorbar(data["N"], data["vidurkis"], yerr=data["sn"], fmt="o-",
                       capsize=3, label=name)
    right.set_xticks([5, 10, 15, 20])
    right.set_xlabel("Imčių skaičius N")
    right.set_ylabel("ARI vidurkis ± SN")
    right.set_title("Imčių pakankamumas")
    right.legend(fontsize=8, frameon=False)
    right.grid(alpha=0.3)

    fig.tight_layout()
    save_figure(fig, "galutiniai_stabilumas.png")


def main():
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    _, X_original = load_space("originali")
    background = background_map()

    rows, stability_tables, by_n_tables, plots = [], [], [], {}
    labels_table = pd.DataFrame({"id": load_space("originali")[0]}).set_index("id")
    reference = None

    for space, k in RUNS:
        print(f"{run_title(space, k)}...")
        ids, labels, pairs, row, stability_pairs, by_n = evaluate_run(
            space, k, X_original, reference)
        if reference is None:
            reference = (ids, labels)

        aligned = align_labels(labels, pairs, REFERENCE[1])
        labels_table.loc[ids, run_name(space, k)] = aligned
        plots[(space, k)] = (ids, aligned)
        rows.append(row)
        stability_tables.append(stability_pairs)
        by_n_tables.append(by_n)

    table = pd.DataFrame(rows)
    table.round(4).to_csv(FINAL_DIR / "galutiniai_rezultatai.csv", index=False)
    stability_pairs = pd.concat(stability_tables, ignore_index=True)
    stability_pairs.round(4).to_csv(FINAL_DIR / "stabilumas_poros.csv", index=False)
    by_n = pd.concat(by_n_tables, ignore_index=True)
    by_n.round(4).to_csv(FINAL_DIR / "stabilumas_pagal_n.csv", index=False)
    labels_table.astype(int).to_csv(FINAL_DIR / "zymes.csv")

    pd.set_option("display.width", 250)
    print("\nEksperimentų lentelė (išoriniai matai – lyginant su pradine erdve, k = 3):")
    print(table[["erdve", "k", "silhouette", "ch", "db", "silhouette_orig",
                 "ari", "nmi", "jaccard", "sutampa_tasku", "sutampa_dalis",
                 "stab_ari", "klasteriu_dydziai"]].round(3).to_string(index=False))
    print("\nImčių pakankamumas (ARI vidurkis pagal N):")
    print(by_n.pivot(index="N", columns="variantas", values="vidurkis").round(3).to_string())
    print("\nPastaba: t-SNE vidiniai matai (silhouette, ch, db) dirbtinai dideli, todėl erdves "
          "lyginkite pagal silhouette_orig ir išorinius matus.")

    plot_final(plots, background)
    plot_stability(stability_pairs, by_n)
    print(f"Išsaugota: rezultatai/kmeans/galutiniai/")


if __name__ == "__main__":
    main()