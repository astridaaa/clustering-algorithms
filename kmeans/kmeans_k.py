"""k-vidurkių klasterių skaičiaus parinkimas kiekvienoje erdvėje.

Kiekvienai erdvei (pradinė, PCA, t-SNE) ir k = 2, ..., 15 apskaičiuojami:
  * E_k-means – kvadratinė paklaida, pagal kurią taikomas alkūnės metodas
    (k = 1, ..., 15). Papildomai pateikiama ir skaidrių var(k) – klasterių
    vidinių dispersijų suma. Ji nesvertinama klasterių dydžiais, todėl kai
    k-vidurkiai išskiria mažą, labai išsibarsčiusį išskirčių klasterį, var(k)
    didėjant k ne mažėja, o auga, ir alkūnės pagal ją nustatyti negalima;
  * vidutinis siluetas – vidutinio silueto metodas;
  * Calinski–Harabasz ir Davies–Bouldin indeksai – papildomi kriterijai;
  * stabilumas – ARI vidurkis per 80 % imtis (stiprūs klasteriai išlieka
    stabilūs, žr. paskaitos skaidres).
Papildomai įvertinamas empyrinis k ≈ sqrt(m / 2).

Kiekvienas bandymas įrašomas į eksperimentų žurnalą. Galutinį k kiekvienai
erdvei parenkate patys pagal visus kriterijus kartu (žr. k_pasiulymai.csv
ir grafikus).

Paleidimas iš repozitorijos šaknies:
    python kmeans/kmeans_k.py
"""

import time

from kmeans_common import (ALGORITHM, OUTPUT_DIR, RANDOM_STATE, SPACE_NAMES,
                           fit_kmeans, save_figure)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import evaluation as ev
from common import load_space, load_subsample, subsample_seeds
from config import SPACES

K_RANGE = range(2, 16)
PARAMS = {"init": "k-means++", "n_init": 10, "atstumas": "euclidean"}


def knee_point(ks, values):
    """Alkūnės taškas: labiausiai nuo stygos tarp galinių taškų nutolęs
    mažėjančios kreivės taškas (Kneedle algoritmo idėja)."""
    ks, values = np.asarray(ks, dtype=float), np.asarray(values, dtype=float)
    x = (ks - ks.min()) / (ks.max() - ks.min())
    y = (values - values.min()) / (values.max() - values.min())
    return int(ks[np.argmax((1 - x) - y)])


def subsample_stability(space, k):
    """k-vidurkiai kiekvienoje 80 % imtyje (sėkla = imties numeris)."""
    samples = {}
    for seed in subsample_seeds(space):
        ids, X = load_subsample(space, seed)
        labels = fit_kmeans(X, k, random_state=seed)[0]
        samples[seed] = (ids, X, labels)
    pairs, _ = ev.stability(samples)
    return pairs


def evaluate_k(space, ids, X, X_original, k):
    labels, _, inertia = fit_kmeans(X, k)
    internal = ev.internal_metrics(X, labels)
    internal_original = None if space == "originali" else ev.internal_metrics(X_original, labels)
    pairs = subsample_stability(space, k)

    ev.log_experiment(ALGORITHM, space, {"k": k, **PARAMS}, labels,
                      random_state=RANDOM_STATE, internal=internal,
                      internal_original=internal_original,
                      stability_pairs=pairs, notes="k parinkimas")

    (ari_mean, ari_sd), = ev.summarize(pairs, ["ari"]).values()
    return {"erdve": space, "k": k, "var_k": ev.within_cluster_variance(X, labels),
            "E_kmeans": inertia, **internal,
            "stab_ari": ari_mean, "stab_ari_sn": ari_sd,
            "maziausias_klasteris": np.bincount(labels).min()}


def scan_space(space, X_original):
    ids, X = load_space(space)
    squared = ((X - X.mean(axis=0)) ** 2).sum(axis=1)
    rows = [{"erdve": space, "k": 1, "var_k": squared.mean(), "E_kmeans": squared.sum()}]

    for k in K_RANGE:
        start = time.time()
        rows.append(evaluate_k(space, ids, X, X_original, k))
        row = rows[-1]
        print(f"  k = {k:2d}: E = {row['E_kmeans']:.0f}, var(k) = {row['var_k']:.3f}, siluetas = {row['silhouette']:.3f}, "
              f"CH = {row['ch']:.0f}, DB = {row['db']:.3f}, "
              f"stabilumas ARI = {row['stab_ari']:.3f} ± {row['stab_ari_sn']:.3f} "
              f"({time.time() - start:.0f} s)")

    k_empirical = ev.empirical_k(len(ids))
    empirical = evaluate_k(space, ids, X, X_original, k_empirical)
    print(f"  empyrinis k = {k_empirical}: siluetas = {empirical['silhouette']:.3f}, "
          f"stabilumas ARI = {empirical['stab_ari']:.3f} ± {empirical['stab_ari_sn']:.3f}")

    table = pd.DataFrame(rows)
    scanned = table[table["k"] >= 2]
    suggestions = {
        "erdve": space,
        "empyrinis": k_empirical,
        "alkunes": knee_point(table["k"], table["E_kmeans"]),
        "silueto": int(scanned.loc[scanned["silhouette"].idxmax(), "k"]),
        "CH": int(scanned.loc[scanned["ch"].idxmax(), "k"]),
        "DB": int(scanned.loc[scanned["db"].idxmin(), "k"]),
        "stabiliausias": int(scanned.loc[scanned["stab_ari"].idxmax(), "k"]),
    }
    return table, pd.DataFrame([empirical]), suggestions


def plot_criteria(table):
    criteria = [("E_kmeans", "E_k-means", "Alkūnės metodas"),
                ("var_k", "var(k)", "Vidinių dispersijų suma var(k)"),
                ("silhouette", "Vidutinis siluetas", "Vidutinio silueto metodas"),
                ("ch", "CH", "Calinski–Harabasz ↑"),
                ("db", "DB", "Davies–Bouldin ↓")]
    fig, axes = plt.subplots(len(SPACES), len(criteria), figsize=(19, 10))
    for row, space in enumerate(SPACES):
        data = table[table["erdve"] == space]
        for col, (column, ylabel, title) in enumerate(criteria):
            ax = axes[row, col]
            values = data.dropna(subset=[column])
            ax.plot(values["k"], values[column], "o-", color="#2a78d6", markersize=4)
            if column == "E_kmeans":
                knee = knee_point(values["k"], values[column])
                ax.axvline(knee, color="#eb6834", linestyle="--", linewidth=1,
                           label=f"alkūnė: k = {knee}")
                ax.legend(fontsize=8, frameon=False)
            ax.set_xticks(range(1, max(K_RANGE) + 1))
            ax.grid(alpha=0.3)
            if row == 0:
                ax.set_title(title, fontsize=11)
            if row == len(SPACES) - 1:
                ax.set_xlabel("k")
            ax.set_ylabel(f"{SPACE_NAMES[space]}\n{ylabel}" if col == 0 else ylabel)
    fig.suptitle("k-vidurkių klasterių skaičiaus parinkimo kriterijai", fontsize=13)
    fig.tight_layout()
    save_figure(fig, "k_kriterijai.png")


def plot_stability(table):
    fig, axes = plt.subplots(1, len(SPACES), figsize=(15, 4), sharey=True)
    for ax, space in zip(axes, SPACES):
        data = table[(table["erdve"] == space) & (table["k"] >= 2)]
        ax.errorbar(data["k"], data["stab_ari"], yerr=data["stab_ari_sn"], fmt="o-",
                    color="#2a78d6", ecolor="#9bbbe8", capsize=3, markersize=4)
        ax.set_title(SPACE_NAMES[space], fontsize=11)
        ax.set_xticks(list(K_RANGE))
        ax.set_xlabel("k")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Stabilumas: ARI vidurkis ± SN")
    fig.suptitle("k-vidurkių stabilumas per 80 % imtis pagal k", fontsize=13)
    fig.tight_layout()
    save_figure(fig, "k_stabilumas.png")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _, X_original = load_space("originali")

    tables, empirical_rows, suggestions = [], [], []
    for space in SPACES:
        print(f"\n{SPACE_NAMES[space]}:")
        table, empirical, suggestion = scan_space(space, X_original)
        tables.append(table)
        empirical_rows.append(empirical)
        suggestions.append(suggestion)

    table = pd.concat(tables, ignore_index=True)
    table.round(4).to_csv(OUTPUT_DIR / "k_parinkimas.csv", index=False)
    pd.concat(empirical_rows, ignore_index=True).round(4).to_csv(
        OUTPUT_DIR / "k_empyrinis.csv", index=False)
    suggestions = pd.DataFrame(suggestions)
    suggestions.to_csv(OUTPUT_DIR / "k_pasiulymai.csv", index=False)

    print("\nKlasterių skaičius pagal kiekvieną kriterijų:")
    print(suggestions.to_string(index=False))
    print("\nPastaba: t-SNE erdvės vidiniai matai dirbtinai didesni, todėl jie "
          "lyginami tik toje pačioje erdvėje, ne su kitomis erdvėmis.")

    plot_criteria(table)
    plot_stability(table)
    print(f"\nIšsaugota: {OUTPUT_DIR.relative_to(OUTPUT_DIR.parents[1])}/k_parinkimas.csv, "
          "k_empyrinis.csv, k_pasiulymai.csv")


if __name__ == "__main__":
    main()