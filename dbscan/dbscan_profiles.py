from dbscan_common import (NOISE_COLOUR, OUTPUT_DIR, PALETTE, SPACE_NAMES,
                           VARIANTS, save_figure)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import evaluation as ev
from common import load_meta
from config import CLASS_COLUMN, CLASSES, FEATURES, INPUT_FILE, OUTLIER_COLUMN
from evaluation import NOISE

FINAL_DIR = OUTPUT_DIR / "galutiniai"
PROFILE_DIR = OUTPUT_DIR / "profiliai"

UNITS = {"RR_sant_vid": "", "RR_pre": "ms", "RR_sant_post": "", "QRS": "ms", "QTc": "ms"}
CLASS_COLOURS = dict(zip(CLASSES, PALETTE))
TOP_N = 3


def run_title(run):
    space, variant = run.rsplit("_", 1)
    return f"{SPACE_NAMES[space]}, {variant}"


def load_data():
    labels = pd.read_csv(FINAL_DIR / "zymes.csv").set_index("id")
    raw = pd.read_csv(INPUT_FILE).set_index("id")[FEATURES]
    meta = load_meta().set_index("id")
    data = meta.join(raw).loc[labels.index]
    assert data[FEATURES].notna().all().all(), "Visi objektai turi turėti požymius"
    return data, labels


def cluster_name(cluster):
    return "triukšmas" if cluster == NOISE else str(cluster)


def top_values(series, n=TOP_N):
    shares = series.value_counts(normalize=True).head(n)
    return ", ".join(f"{value} {share * 100:.0f} %" for value, share in shares.items())


def profile_run(data, labels, run):
    rows = []
    for cluster in sorted(labels[run].unique()):
        members = data[labels[run] == cluster]
        row = {"suskirstymas": run, "klasteris": cluster_name(cluster), "dydis": len(members),
               "dalis_%": len(members) / len(data) * 100,
               "isskirciu_dalis_%": members[OUTLIER_COLUMN].mean() * 100}
        for feature in FEATURES:
            q1, median, q3 = members[feature].quantile([0.25, 0.5, 0.75])
            row[f"{feature}_mediana"] = median
            row[f"{feature}_Q1"] = q1
            row[f"{feature}_Q3"] = q3
        classes = members[CLASS_COLUMN].value_counts(normalize=True)
        row.update({f"{name}_%": classes.get(name, 0) * 100 for name in CLASSES})
        row["dominuojanti_klase"] = classes.idxmax()
        row["dominuojancios_dalis_%"] = classes.max() * 100
        row["dazniausi_ritmai"] = top_values(members["ritmas"])
        row["dazniausi_irasai"] = top_values(members["irasas"])
        rows.append(row)
    return rows


def descriptive_statistics(data, labels):
    """Kiekvieno klasterio požymių aprašomoji statistika pradiniais vienetais."""
    rows = []
    for run in labels.columns:
        for cluster in sorted(labels[run].unique()):
            members = data.loc[labels[run] == cluster, FEATURES]
            for feature in FEATURES:
                values = members[feature]
                rows.append({"suskirstymas": run, "klasteris": cluster_name(cluster),
                             "pozymis": feature, "n": len(values), "min": values.min(),
                             "Q1": values.quantile(0.25), "mediana": values.median(),
                             "vidurkis": values.mean(), "Q3": values.quantile(0.75),
                             "max": values.max(), "dispersija": values.var(ddof=1)})
    return pd.DataFrame(rows)


def class_counts(data, labels):
    """Kiek kiekvienos klasės dūžių pateko į kiekvieną klasterį (kryžminė lentelė)."""
    tables = []
    for run in labels.columns:
        table = pd.crosstab(labels[run], data[CLASS_COLUMN]).reindex(columns=CLASSES, fill_value=0)
        table.index = [cluster_name(c) for c in table.index]
        tables.append(table.rename_axis("klasteris").reset_index().assign(suskirstymas=run))
    return pd.concat(tables, ignore_index=True)[["suskirstymas", "klasteris", *CLASSES]]


def plot_composition(data, labels, variant):
    runs = [run for run in labels.columns if run.endswith(f"_{variant}")]
    widths = [labels[run].nunique() + 1 for run in runs]
    fig, axes = plt.subplots(1, len(runs), figsize=(max(14, 0.45 * sum(widths)), 4.8),
                             sharey=True, gridspec_kw={"width_ratios": widths})
    for ax, run in zip(axes, runs):
        table = pd.crosstab(labels[run], data[CLASS_COLUMN], normalize="index")
        table = table.reindex(columns=CLASSES, fill_value=0) * 100
        bottom = np.zeros(len(table))
        for name in CLASSES:
            ax.bar(range(len(table)), table[name], bottom=bottom, color=CLASS_COLOURS[name],
                   edgecolor="#fcfcfb", linewidth=0.8, label=name)
            bottom += table[name].to_numpy()
        sizes = labels[run].value_counts()
        ax.set_xticks(range(len(table)))
        ax.set_xticklabels([f"{'tr.' if c == NOISE else c}\n({sizes[c]})" for c in table.index],
                           fontsize=7 if len(table) > 10 else 9)
        ax.set_xlim(-0.6, len(table) - 0.4)
        ax.set_title(run_title(run), fontsize=10)
    fig.supxlabel("Klasteris (dydis); tr. – triukšmas", fontsize=10)
    axes[0].set_ylabel("Dūžių klasių dalis klasteryje, %")
    axes[-1].legend(title="Klasė", fontsize=8, frameon=False, bbox_to_anchor=(1, 1))
    fig.suptitle(f"DBSCAN klasterių ir triukšmo sudėtis pagal dūžių klases ({VARIANTS[variant]})",
                 fontsize=13)
    fig.tight_layout()
    save_figure(fig, f"profiliai_klases_{variant}.png")


def plot_features(data, labels, run):
    clusters = sorted(labels[run].unique())
    colours = [NOISE_COLOUR if c == NOISE else PALETTE[c % len(PALETTE)] for c in clusters]
    fig, axes = plt.subplots(1, len(FEATURES), figsize=(18, 4))
    for ax, feature in zip(axes, FEATURES):
        values = [data.loc[labels[run] == c, feature] for c in clusters]
        parts = ax.boxplot(values, tick_labels=[cluster_name(c) for c in clusters],
                           patch_artist=True, showfliers=False,
                           medianprops={"color": "#0b0b0b"})
        for patch, colour in zip(parts["boxes"], colours):
            patch.set_facecolor(colour)
            patch.set_alpha(0.6)
        unit = f", {UNITS[feature]}" if UNITS[feature] else ""
        ax.set_title(f"{feature}{unit}", fontsize=10)
        ax.set_xlabel("Klasteris")
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"Požymių pasiskirstymas DBSCAN klasteriuose ({run_title(run)}; "
                 "be išskirčių ūsų)", fontsize=12)
    fig.tight_layout()
    save_figure(fig, f"profiliai_pozymiai_{run}.png")


def main():
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    data, labels = load_data()

    profiles = pd.DataFrame([row for run in labels.columns
                             for row in profile_run(data, labels, run)])
    profiles.round(3).to_csv(PROFILE_DIR / "profiliai.csv", index=False)
    counts = class_counts(data, labels)
    counts.to_csv(PROFILE_DIR / "klases_klasteriuose.csv", index=False)
    descriptive_statistics(data, labels).round(3).to_csv(
        PROFILE_DIR / "aprasomoji_statistika.csv", index=False)

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
        print(f"\n{run_title(run)}:")
        print(table[["klasteris", "dydis", "isskirciu_dalis_%", *medians,
                     "dominuojanti_klase", "dominuojancios_dalis_%"]].round(2).to_string(index=False))

    print("\nSuskirstymų atitikimas dūžių klasėms, ritmams ir įrašams (triukšmas – atskira grupė):")
    print(agreement.pivot(index="suskirstymas", columns="zyme", values="ari")
          .round(3).add_prefix("ARI_").to_string())

    for variant in VARIANTS:
        plot_composition(data, labels, variant)
    for variant in VARIANTS:
        plot_features(data, labels, f"originali_{variant}")
    print("Išsaugota: rezultatai/dbscan/profiliai/profiliai.csv, klases_klasteriuose.csv, "
          "aprasomoji_statistika.csv, atitikimas_zymems.csv")


if __name__ == "__main__":
    main()
