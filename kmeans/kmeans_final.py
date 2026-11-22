from kmeans_common import (ALGORITHM, CHOSEN_K, OUTPUT_DIR, RANDOM_STATE,
                           REFERENCE, SPACE_NAMES, align_labels, background_map,
                           final_spaces, fit_kmeans, load, plot_clusters,
                           save_figure, subsample_runs)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import evaluation as ev

PARAMS = {"init": "k-means++", "n_init": 10, "atstumas": "euclidean"}
FINAL_DIR = OUTPUT_DIR / "galutiniai"
K_REFERENCE = CHOSEN_K[REFERENCE]
SAME_K_NOTE = " (tas pats k kaip pradinėje)"


def final_runs():
    runs = []
    for space in final_spaces():
        runs.append((space, CHOSEN_K[space]))
        if CHOSEN_K[space] != K_REFERENCE:
            runs.append((space, K_REFERENCE))
    return runs


def run_name(space, k):
    return f"{space}_k{k}"


def run_title(space, k):
    extra = SAME_K_NOTE if k != CHOSEN_K[space] else ""
    return f"{SPACE_NAMES[space]}, k = {k}{extra}"


def evaluate_run(space, k, X_original, reference):
    ids, X = load(space)
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

    stability_pairs, stability_samples = ev.stability(subsample_runs(space, k))
    ev.log_experiment(ALGORITHM, space, {"k": k, **PARAMS}, labels,
                      random_state=RANDOM_STATE, internal=internal,
                      internal_original=None if space == REFERENCE else internal_original,
                      external=external, stability_pairs=stability_pairs,
                      notes="galutinis modelis" + ("" if reference is None else
                                                   f"; išoriniai matai – su pradine erdve, k = {K_REFERENCE}"))

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
    by_n = ev.stability_vs_n(stability_pairs, ns=[n for n in (5, 10, 15, 20)
                                                  if n <= len(stability_samples)])
    by_n = by_n.assign(variantas=run_title(space, k))
    return ids, labels, pairs, row, stability_pairs, by_n


def plot_final(results, background):
    n = len(results)
    cols = 2 if n <= 4 else 3
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6.5 * cols, 5.5 * rows), squeeze=False)
    for ax, ((space, k), (ids, aligned)) in zip(axes.ravel(), results.items()):
        plot_clusters(ax, ids, aligned, run_title(space, k), background)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    fig.suptitle("Galutiniai k-vidurkių klasteriai (bendras t-SNE žemėlapis; "
                 "ta pati spalva – suporuotas klasteris)", fontsize=13)
    fig.tight_layout()
    save_figure(fig, "galutiniai_klasteriai.png")


def plot_stability(stability_pairs, by_n):
    names = [name for name in dict.fromkeys(stability_pairs["variantas"])
             if not name.endswith(SAME_K_NOTE)]
    colours = dict(zip(names, ["#2a78d6", "#eb6834", "#1baf7a"]))
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 5), sharey=True,
                                      gridspec_kw={"width_ratios": [1, 1.2]})

    parts = left.boxplot([stability_pairs.loc[stability_pairs["variantas"] == name, "ari"]
                          for name in names],
                         tick_labels=[name.replace(", ", "\n", 1) for name in names],
                         patch_artist=True, widths=0.5,
                         medianprops={"color": "black"})
    for patch, name in zip(parts["boxes"], names):
        patch.set_facecolor(colours[name])
        patch.set_alpha(0.6)
    left.set_ylim(0.9, 1.005)
    left.set_ylabel("ARI tarp imčių porų")
    left.set_title("ARI tarp visų 80 % imčių porų")
    left.grid(alpha=0.3, axis="y")

    for name in names:
        data = by_n[by_n["variantas"] == name]
        right.errorbar(data["N"], data["vidurkis"], yerr=data["sn"], fmt="o-",
                       capsize=4, linewidth=2, color=colours[name], label=name)
    right.set_xticks([5, 10, 15, 20])
    right.set_xlabel("Imčių skaičius N (t-SNE – iki 10)")
    right.set_title("ARI vidurkis ± SN pagal imčių skaičių")
    right.legend(fontsize=9, frameon=False, loc="lower right")
    right.grid(alpha=0.3)

    fig.suptitle("Galutinių k-vidurkių suskirstymų stabilumas", fontsize=13)
    fig.tight_layout()
    save_figure(fig, "galutiniai_stabilumas.png")


def main():
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    all_ids, X_original = load(REFERENCE)
    background = background_map()

    rows, stability_tables, by_n_tables, plots = [], [], [], {}
    labels_table = pd.DataFrame({"id": all_ids}).set_index("id")
    reference = None

    for space, k in final_runs():
        print(f"{run_title(space, k)}...")
        ids, labels, pairs, row, stability_pairs, by_n = evaluate_run(
            space, k, X_original, reference)
        if reference is None:
            reference = (ids, labels)

        aligned = align_labels(labels, pairs, K_REFERENCE)
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
    print(f"\nEksperimentų lentelė (išoriniai matai – lyginant su pradine erdve, k = {K_REFERENCE}):")
    print(table[["erdve", "k", "silhouette", "ch", "db", "silhouette_orig",
                 "ari", "nmi", "jaccard", "sutampa_tasku", "sutampa_dalis",
                 "stab_ari", "klasteriu_dydziai"]].round(3).to_string(index=False))
    print("\nImčių pakankamumas (ARI vidurkis pagal N):")
    print(by_n.pivot(index="N", columns="variantas", values="vidurkis").round(3).to_string())
    print("\nPastaba: t-SNE vidiniai matai (silhouette, ch, db) dirbtinai dideli, todėl erdves "
          "lyginkite pagal silhouette_orig ir išorinius matus.")

    plot_final(plots, background)
    plot_stability(stability_pairs, by_n)
    print("Išsaugota: rezultatai/kmeans/galutiniai/")


if __name__ == "__main__":
    main()