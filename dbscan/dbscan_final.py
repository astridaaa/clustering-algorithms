from dbscan_common import (ALGORITHM, MIN_SAMPLES, OUTPUT_DIR, PALETTE,
                           SPACE_NAMES, SPACES, STABILITY_SPACES, TARGET_K, VARIANTS,
                           align_labels, background_map, cluster_sizes, fit_dbscan,
                           load, load_chosen, params, plot_clusters, plot_point_types,
                           point_types, save_figure, subsample_runs)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import evaluation as ev
from common import load_meta
from config import OUTLIER_COLUMN
from evaluation import NOISE

FINAL_DIR = OUTPUT_DIR / "galutiniai"
REFERENCE = "originali"
ORDER = [REFERENCE] + [space for space in SPACES if space != REFERENCE]


def run_name(space, variant):
    return f"{space}_{variant}"


def noise_vs_outliers(ids, labels):
    """Ar DBSCAN triukšmas sutampa su pažymėtomis ekstremaliomis išskirtimis (3×IQR)."""
    outliers = load_meta().set_index("id").loc[ids, OUTLIER_COLUMN].to_numpy().astype(bool)
    noise = np.asarray(labels) == NOISE
    both = int((noise & outliers).sum())
    return {"triuksmo_tasku": int(noise.sum()), "isskirciu": int(outliers.sum()),
            "triuksmas_ir_isskirtis": both,
            "triuksmo_dalis_isskirtys_%": 100 * both / max(noise.sum(), 1),
            "isskirciu_dalis_triuksme_%": 100 * both / max(outliers.sum(), 1)}


def mean_sd(table, column, scale=1.0, digits=3):
    values = table[column].dropna() * scale
    return ev.format_mean_sd(values.mean(), values.std(ddof=1), digits) if len(values) > 1 else "–"


def evaluate_run(space, variant, eps, X_original, reference):
    ids, X = load(space)
    labels, core = fit_dbscan(X, eps, MIN_SAMPLES)
    name = run_name(space, variant)

    internal = ev.internal_metrics(X, labels)
    internal_original = ev.internal_metrics(X_original, labels)
    if reference is None:
        external, matched, share = None, len(labels), 1.0
        names = list(np.unique(labels))
        pairs = pd.DataFrame({"atskaitos_klasteris": names, "klasteris": names})
    else:
        external = ev.compare_with_reference(ids, labels, *reference)
        matched, share, pairs, crosstab = ev.matched_points(ids, labels, *reference)
        pairs.to_csv(FINAL_DIR / f"sutampa_{name}.csv", index=False)
        crosstab.to_csv(FINAL_DIR / f"kryzmine_{name}.csv")

    stability_pairs = stability_samples = by_n = None
    if space in STABILITY_SPACES:
        stability_pairs, stability_samples = ev.stability(
            subsample_runs(STABILITY_SPACES[space], eps, MIN_SAMPLES))
        by_n = ev.stability_vs_n(stability_pairs).assign(suskirstymas=name)
        stability_pairs = stability_pairs.assign(suskirstymas=name)

    ev.log_experiment(ALGORITHM, space, params(eps), labels,
                      internal=internal,
                      internal_original=None if space == REFERENCE else internal_original,
                      external=external, stability_pairs=stability_pairs,
                      notes=f"galutinis modelis; {VARIANTS[variant]}" +
                            ("" if reference is None else "; išoriniai matai – su pradine erdve"))

    types = point_types(labels, core)
    row = {
        "suskirstymas": name, "erdve": space, "variantas": variant,
        "eps": eps, "min_samples": MIN_SAMPLES, "k": ev.n_clusters(labels),
        "branduoliniu": int((types == "branduolinis").sum()),
        "ribiniu": int((types == "ribinis").sum()),
        "triuksmo": int((types == "triukšmas").sum()),
        "triuksmo_dalis_%": ev.noise_share(labels) * 100,
        **internal,
        **{f"{metric}_orig": value for metric, value in internal_original.items()},
        **(external or {"ari": np.nan, "nmi": np.nan, "jaccard": np.nan}),
        "sutampa_tasku": matched, "sutampa_dalis": share,
        "klasteriu_dydziai": " / ".join(map(str, cluster_sizes(labels))),
    }
    if stability_pairs is not None:
        row.update({f"stab_{metric}": mean_sd(stability_pairs, metric)
                    for metric in ["ari", "nmi", "jaccard"]})
        row["stab_ari_min"] = stability_pairs["ari"].min()
        row["imciu_k"] = f"{stability_samples['k'].min()}–{stability_samples['k'].max()}"
        row["imciu_triuksmas_%"] = mean_sd(stability_samples, "triuksmo_dalis", 100, 1)
        row.update({f"imciu_{metric}": mean_sd(stability_samples, metric)
                    for metric in ["silhouette", "ch", "db"]})
    row.update(noise_vs_outliers(ids, labels))
    return ids, labels, types, pairs, row, stability_pairs, by_n


def plot_final(results, background):
    fig, axes = plt.subplots(len(VARIANTS), len(ORDER), figsize=(6 * len(ORDER), 5.5 * len(VARIANTS)),
                             squeeze=False)
    for row, variant in enumerate(VARIANTS):
        for col, space in enumerate(ORDER):
            ids, aligned, _, info = results[(space, variant)]
            title = (f"{SPACE_NAMES[space]}, {variant}: eps = {info['eps']:.3f}, "
                     f"min_samples = {MIN_SAMPLES}, k = {info['k']}")
            plot_clusters(axes[row, col], ids, aligned, title, background)
    names = "; ".join(f"{name} – {text}" for name, text in VARIANTS.items())
    fig.suptitle(f"Galutiniai DBSCAN klasteriai bendrame t-SNE žemėlapyje ({names}; "
                 "pilka – triukšmas)", fontsize=13)
    fig.tight_layout()
    save_figure(fig, "galutiniai_klasteriai.png")


def plot_types(results, background):
    fig, axes = plt.subplots(len(VARIANTS), len(ORDER), figsize=(6 * len(ORDER), 5.5 * len(VARIANTS)),
                             squeeze=False)
    for row, variant in enumerate(VARIANTS):
        for col, space in enumerate(ORDER):
            ids, _, types, info = results[(space, variant)]
            plot_point_types(axes[row, col], ids, types,
                             f"{SPACE_NAMES[space]}, {variant}: eps = {info['eps']:.3f}",
                             background)
    fig.suptitle("Branduoliniai, ribiniai ir triukšmo taškai", fontsize=13)
    fig.tight_layout()
    save_figure(fig, "tasku_tipai.png")


def plot_stability(stability_pairs, by_n):
    names = list(dict.fromkeys(stability_pairs["suskirstymas"]))
    variant_names = {"kelio": "kelio metodas", **{name: f"{k} klasteriai" for name, k in TARGET_K.items()}}

    def label(name, sep="\n"):
        space, variant = name.rsplit("_", 1)
        return f"{SPACE_NAMES[space]}{sep}{variant_names[variant]}"

    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 5))
    left.boxplot([stability_pairs.loc[stability_pairs["suskirstymas"] == name, "ari"]
                  for name in names], tick_labels=[label(name) for name in names],
                 medianprops={"color": "#0b0b0b"})
    left.set_ylabel("ARI tarp imčių porų")
    left.set_title("Stabilumas: ARI per 80 % imčių poras")
    left.grid(alpha=0.3, axis="y")

    for colour, name in zip(PALETTE, names):
        data = by_n[by_n["suskirstymas"] == name]
        right.errorbar(data["N"], data["vidurkis"], yerr=data["sn"], fmt="o-", capsize=3,
                       color=colour, label=label(name, ", "))
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
    chosen = load_chosen()
    all_ids, X_original = load("originali")
    background = background_map()

    rows, stability_tables, by_n_tables, plots = [], [], [], {}
    labels_table = pd.DataFrame({"id": all_ids}).set_index("id")
    types_table = labels_table.copy()

    for variant in VARIANTS:
        reference = None
        for space in ORDER:
            name = run_name(space, variant)
            print(f"{SPACE_NAMES[space]}, {variant}...")
            ids, labels, types, pairs, row, stability_pairs, by_n = evaluate_run(
                space, variant, chosen[(space, variant)], X_original, reference)
            if reference is None:
                reference = (ids, labels)
            aligned = align_labels(labels, pairs)
            labels_table.loc[ids, name] = aligned
            types_table.loc[ids, name] = types
            plots[(space, variant)] = (ids, aligned, types, row)
            rows.append(row)
            if stability_pairs is not None:
                stability_tables.append(stability_pairs)
                by_n_tables.append(by_n)

    table = pd.DataFrame(rows)
    table.round(4).to_csv(FINAL_DIR / "galutiniai_rezultatai.csv", index=False)
    stability_pairs = pd.concat(stability_tables, ignore_index=True)
    stability_pairs.round(4).to_csv(FINAL_DIR / "stabilumas_poros.csv", index=False)
    by_n = pd.concat(by_n_tables, ignore_index=True)
    by_n.round(4).to_csv(FINAL_DIR / "stabilumas_pagal_n.csv", index=False)
    labels_table.astype(int).to_csv(FINAL_DIR / "zymes.csv")
    types_table.to_csv(FINAL_DIR / "tasku_tipai.csv")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 50)
    print("\nEksperimentų lentelė (išoriniai matai – su to paties varianto pradine erdve):")
    print(table[["suskirstymas", "eps", "k", "triuksmo_dalis_%", "silhouette",
                 "silhouette_orig", "ari", "nmi", "jaccard", "sutampa_tasku",
                 "klasteriu_dydziai"]].round(3).to_string(index=False))
    print("\nStabilumas (pradinė erdvė, PCA 3D ir t-SNE 2D):")
    print(table.dropna(subset=["stab_ari"])[["suskirstymas", "stab_ari", "stab_nmi",
                                             "stab_jaccard", "imciu_k",
                                             "imciu_triuksmas_%"]].to_string(index=False))
    print("\nTriukšmas ir pažymėtos ekstremalios išskirtys:")
    print(table[["suskirstymas", "triuksmo_tasku", "triuksmas_ir_isskirtis",
                 "triuksmo_dalis_isskirtys_%", "isskirciu_dalis_triuksme_%"]]
          .round(1).to_string(index=False))
    print("\nPastaba: vidiniai matai skaičiuojami be triukšmo; kai k = 1, jie neapibrėžti. "
          "t-SNE vidiniai matai dirbtinai dideli – erdves lyginkite pagal silhouette_orig "
          "ir išorinius matus.")

    plot_final(plots, background)
    plot_types(plots, background)
    plot_stability(stability_pairs, by_n)
    print("Išsaugota: rezultatai/dbscan/galutiniai/")


if __name__ == "__main__":
    main()
