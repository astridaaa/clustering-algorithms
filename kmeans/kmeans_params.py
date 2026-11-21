from itertools import combinations

from kmeans_common import (ALGORITHM, CHOSEN_K, OUTPUT_DIR, RANDOM_STATE,
                           REFERENCE, SKLEARN_METRIC, SPACE_NAMES, VARIANTS,
                           background_map, final_spaces, fit_kmeans, fit_variant,
                           load, load_space_subsample, plot_clusters, save_figure,
                           seeds_for)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import evaluation as ev

BASE_VARIANT = "k-vidurkiai, Euklido"
INIT_METHODS = ["k-means++", "random"]
N_INITS = [1, 10]
INIT_SEEDS = range(10)

def variant_stability(space, variant, k):
    samples = {}
    for seed in seeds_for(space):
        ids, X = load_space_subsample(space, seed)
        samples[seed] = (ids, X, fit_variant(X, variant, k, random_state=seed))
    pairs, _ = ev.stability(samples, metric=SKLEARN_METRIC[VARIANTS[variant][1]])
    return pairs


def compare_distances(space, X_original):
    ids, X = load(space)
    k = CHOSEN_K[space]
    base = fit_variant(X, BASE_VARIANT, k)

    rows, labels_by_variant = [], {}
    for variant, (method, metric) in VARIANTS.items():
        labels = fit_variant(X, variant, k)
        labels_by_variant[variant] = labels

        internal = ev.internal_metrics(X, labels, metric=SKLEARN_METRIC[metric])
        internal_original = (None if space == REFERENCE
                             else ev.internal_metrics(X_original, labels))
        external = ev.external_metrics(base, labels)
        pairs = variant_stability(space, variant, k)
        outliers = ev.outlier_share(ids, labels)

        ev.log_experiment(ALGORITHM, space,
                          {"k": k, "metodas": method, "atstumas": metric, "n_init": 10},
                          labels, random_state=RANDOM_STATE, internal=internal,
                          internal_original=internal_original, external=external,
                          stability_pairs=pairs,
                          notes=f"atstumo matas; išoriniai matai – lyginant su „{BASE_VARIANT}“")

        (stab_mean, stab_sd), = ev.summarize(pairs, ["ari"]).values()
        rows.append({
            "erdve": space, "variantas": variant, "k": k,
            "silhouette": internal["silhouette"],
            "silhouette_euklido": ev.internal_metrics(X, labels)["silhouette"],
            "ch": internal["ch"], "db": internal["db"],
            **{f"{name}_su_kvidurkiais": value for name, value in external.items()},
            "stab_ari": stab_mean, "stab_ari_sn": stab_sd,
            "klasteriu_dydziai": " / ".join(str(n) for n in np.bincount(labels)),
            "didziausia_isskirciu_dalis": outliers["dalis"].max(),
        })
        print(f"  {variant:24s} siluetas = {internal['silhouette']:.3f}, "
              f"ARI su k-vidurkiais = {external['ari']:.3f}, "
              f"stabilumas = {stab_mean:.3f} ± {stab_sd:.3f}, "
              f"dydžiai: {rows[-1]['klasteriu_dydziai']}")

    return pd.DataFrame(rows), ids, labels_by_variant


def plot_distances(results):
    spaces = list(results)
    fig, axes = plt.subplots(len(spaces), len(VARIANTS), figsize=(18, 4.4 * len(spaces)))
    background = background_map()
    for row, space in enumerate(spaces):
        ids, labels_by_variant = results[space]
        for col, variant in enumerate(VARIANTS):
            plot_clusters(axes[row, col], ids, labels_by_variant[variant],
                          f"{SPACE_NAMES[space]}: {variant}", background)
    fig.suptitle("Atstumo mato poveikis (klasteriai pavaizduoti bendrame t-SNE žemėlapyje)",
                 fontsize=13)
    fig.tight_layout()
    save_figure(fig, "parametrai_atstumai.png")

def compare_initialisation(space):
    ids, X = load(space)
    k = CHOSEN_K[space]

    rows = []
    for init in INIT_METHODS:
        for n_init in N_INITS:
            errors, runs = [], []
            for seed in INIT_SEEDS:
                labels, _, inertia = fit_kmeans(X, k, init=init, n_init=n_init,
                                                random_state=seed)
                errors.append(inertia)
                runs.append(labels)
                ev.log_experiment(ALGORITHM, space,
                                  {"k": k, "init": init, "n_init": n_init,
                                   "atstumas": "euclidean"},
                                  labels, random_state=seed,
                                  internal=ev.internal_metrics(X, labels),
                                  notes="pradinių centrų parinkimas")

            errors = np.array(errors)
            seed_ari = [ev.external_metrics(a, b)["ari"] for a, b in combinations(runs, 2)]
            rows.append({
                "erdve": space, "init": init, "n_init": n_init, "k": k,
                "E_min": errors.min(), "E_vid": errors.mean(), "E_max": errors.max(),
                "seklu_su_maziausia_E": int(np.isclose(errors, errors.min()).sum()),
                "ari_tarp_seklu": np.mean(seed_ari),
                "ari_tarp_seklu_min": np.min(seed_ari),
            })
            print(f"  init = {init:9s}, n_init = {n_init:2d}: E nuo {errors.min():.1f} "
                  f"iki {errors.max():.1f}, mažiausią E pasiekė "
                  f"{rows[-1]['seklu_su_maziausia_E']}/{len(INIT_SEEDS)} sėklų, "
                  f"ARI tarp sėklų = {rows[-1]['ari_tarp_seklu']:.3f}")
    return pd.DataFrame(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _, X_original = load(REFERENCE)
    spaces = final_spaces()
    print("Erdvės (pradinė ir geriausios dimensijos):", ", ".join(spaces))

    distance_tables, initialisation_tables, plots = [], [], {}
    for space in spaces:
        print(f"\n{SPACE_NAMES[space]} (k = {CHOSEN_K[space]}), atstumo matas:")
        table, ids, labels_by_variant = compare_distances(space, X_original)
        distance_tables.append(table)
        plots[space] = (ids, labels_by_variant)

        print(f"{SPACE_NAMES[space]}, pradinių centrų parinkimas:")
        initialisation_tables.append(compare_initialisation(space))

    distances = pd.concat(distance_tables, ignore_index=True)
    distances.round(4).to_csv(OUTPUT_DIR / "parametrai_atstumai.csv", index=False)
    initialisation = pd.concat(initialisation_tables, ignore_index=True)
    initialisation.round(4).to_csv(OUTPUT_DIR / "parametrai_inicializacija.csv", index=False)

    plot_distances(plots)
    print("\nPastaba: siluetas skaičiuojamas tuo pačiu atstumu, kuriuo klasterizuota; "
          "stulpelis silhouette_euklido – visiems variantams Euklido atstumu, "
          "kad variantus būtų galima palyginti tarpusavyje.")
    print("Išsaugota: rezultatai/kmeans/parametrai_atstumai.csv, "
          "parametrai_inicializacija.csv")


if __name__ == "__main__":
    main()