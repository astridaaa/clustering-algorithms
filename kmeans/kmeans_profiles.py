from kmeans_common import OUTPUT_DIR, cluster_colours, save_figure

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import evaluation as ev
from common import load_meta
from config import CLASS_COLUMN, CLASSES, FEATURES, INPUT_FILE, OUTLIER_COLUMN

FINAL_DIR = OUTPUT_DIR / "galutiniai"
PROFILE_DIR = OUTPUT_DIR / "profiliai"

UNITS = {"RR_sant_vid": "", "RR_pre": "ms", "RR_sant_post": "", "QRS": "ms", "QTc": "ms"}
TOP_N = 3

RUN_TITLES = {"originali_k3": "Pradinė erdvė, k = 3", "pca_k3": "PCA erdvė, k = 3",
              "tsne_k4": "t-SNE erdvė, k = 4", "tsne_k3": "t-SNE erdvė, k = 3"}


def load_data():
    labels = pd.read_csv(FINAL_DIR / "zymes.csv").set_index("id")
    raw = pd.read_csv(INPUT_FILE).set_index("id")[FEATURES]
    meta = load_meta().set_index("id")
    data = meta.join(raw).loc[labels.index]
    assert data[FEATURES].notna().all().all(), "Visi galutiniai objektai turi turėti požymius"
    return data, labels


def top_values(series, n=TOP_N):
    shares = series.value_counts(normalize=True).head(n)
    return ", ".join(f"{value} {share * 100:.0f} %" for value, share in shares.items())


def profile_run(data, labels, run):
    rows = []
    for cluster in sorted(labels[run].unique()):
        members = data[labels[run] == cluster]
        row = {"suskirstymas": run, "klasteris": cluster, "dydis": len(members),
               "dalis_%": len(members) / len(data) * 100,
               "isskirciu_dalis_%": members[OUTLIER_COLUMN].mean() * 100}
        for feature in FEATURES:
            q1, median, q3 = members[feature].quantile([0.25, 0.5, 0.75])
            row[f"{feature}_mediana"] = median
            row[f"{feature}_Q1"] = q1
            row[f"{feature}_Q3"] = q3
        classes = members[CLASS_COLUMN].value_counts(normalize=True)
        row.update({f"{name}_%": classes.get(name, 0) * 100 for name in CLASSES})
        row["dazniausi_ritmai"] = top_values(members["ritmas"])
        row["dazniausi_irasai"] = top_values(members["irasas"])
        rows.append(row)
    return rows


def plot_composition(data, labels):
    runs = list(labels.columns)
    fig, axes = plt.subplots(1, len(runs), figsize=(4.5 * len(runs), 4.5), sharey=True)
    class_colours = {"N": "#2a78d6", "L": "#eb6834", "R": "#1baf7a", "V": "#eda100", "A": "#e87ba4"}
    for ax, run in zip(axes, runs):
        table = pd.crosstab(labels[run], data[CLASS_COLUMN], normalize="index")[CLASSES] * 100
        bottom = np.zeros(len(table))
        for name in CLASSES:
            ax.bar(table.index.astype(str), table[name], bottom=bottom,
                   color=class_colours[name], label=name)
            bottom += table[name].to_numpy()
        sizes = labels[run].value_counts().sort_index()
        ax.set_xticks(range(len(table)))
        ax.set_xticklabels([f"{c}\n({sizes[c]})" for c in table.index])
        ax.set_title(RUN_TITLES.get(run, run), fontsize=10)
        ax.set_xlabel("Klasteris (dydis)")
    axes[0].set_ylabel("Dūžių klasių dalis klasteryje, %")
    axes[-1].legend(title="Klasė", fontsize=8, frameon=False, bbox_to_anchor=(1, 1))
    fig.suptitle("Klasterių sudėtis pagal dūžių klases", fontsize=13)
    fig.tight_layout()
    save_figure(fig, "profiliai_klases.png")


def plot_features(data, labels, run="originali_k3"):
    clusters = sorted(labels[run].unique())
    colours = cluster_colours(len(clusters))
    fig, axes = plt.subplots(1, len(FEATURES), figsize=(18, 4))
    for ax, feature in zip(axes, FEATURES):
        values = [data.loc[labels[run] == c, feature] for c in clusters]
        parts = ax.boxplot(values, tick_labels=[str(c) for c in clusters],
                           patch_artist=True, showfliers=False)
        for patch, colour in zip(parts["boxes"], colours):
            patch.set_facecolor(colour)
            patch.set_alpha(0.6)
        unit = f", {UNITS[feature]}" if UNITS[feature] else ""
        ax.set_title(f"{feature}{unit}", fontsize=10)
        ax.set_xlabel("Klasteris")
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"Požymių pasiskirstymas klasteriuose ({RUN_TITLES[run]}; be išskirčių ūsų)",
                 fontsize=12)
    fig.tight_layout()
    save_figure(fig, "profiliai_pozymiai.png")


def main():
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    data, labels = load_data()

    profiles = pd.DataFrame([row for run in labels.columns
                             for row in profile_run(data, labels, run)])
    profiles.round(3).to_csv(PROFILE_DIR / "profiliai.csv", index=False)

    agreement = pd.concat([ev.compare_with_meta(labels.index.to_numpy(),
                                                labels[run].to_numpy()).assign(suskirstymas=run)
                           for run in labels.columns], ignore_index=True)
    agreement = agreement[["suskirstymas", "zyme", "ari", "nmi", "jaccard"]]
    agreement.round(4).to_csv(PROFILE_DIR / "atitikimas_zymems.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    medians = [f"{feature}_mediana" for feature in FEATURES]
    for run in labels.columns:
        table = profiles[profiles["suskirstymas"] == run]
        print(f"\n{RUN_TITLES.get(run, run)}:")
        print(table[["klasteris", "dydis", "isskirciu_dalis_%", *medians,
                     *[f"{name}_%" for name in CLASSES]]].round(2).to_string(index=False))
        print(table[["klasteris", "dazniausi_ritmai", "dazniausi_irasai"]].to_string(index=False))

    print("\nSuskirstymų atitikimas dūžių klasėms, ritmams ir įrašams:")
    print(agreement.pivot(index="suskirstymas", columns="zyme", values="ari")
          .round(3).add_prefix("ARI_").to_string())

    plot_composition(data, labels)
    plot_features(data, labels)
    print("Išsaugota: rezultatai/kmeans/profiliai/profiliai.csv, atitikimas_zymems.csv")


if __name__ == "__main__":
    main()