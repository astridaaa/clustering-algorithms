import time

from dbscan_common import (ALGORITHM, BLUE_RAMP, CHOSEN_FILE, MAX_NOISE,
                           MIN_SAMPLES, OUTPUT_DIR, PALETTE, SPACE_NAMES, SPACES,
                           STABILITY_SPACE, VARIANTS, cluster_sizes, fit_dbscan,
                           knee_eps, largest_share, load, params, save_figure,
                           stability_pairs, subsample_runs)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

import evaluation as ev
from common import load_meta
from config import CLASS_COLUMN

EPS_FACTORS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
KNEE_COLOUR = PALETTE[1]
MIN_SAMPLES_GRID = [5, 10, 20]

HEATMAPS = [("k", "Klasterių skaičius", "{:.0f}"),
            ("triuksmo_dalis_%", "Triukšmo dalis, %", "{:.1f}"),
            ("didziausio_dalis_%", "Didžiausio klasterio dalis, %", "{:.1f}"),
            ("ari_klase", "Atitikimas klasėms (ARI)", "{:.2f}"),
            ("stab_ari", "Stabilumas (ARI vidurkis)", "{:.2f}")]


def evaluate(space, X, X_original, classes, eps, min_samples, factor):
    labels, core = fit_dbscan(X, eps, min_samples)
    internal = ev.internal_metrics(X, labels)
    internal_original = (None if space == "originali"
                         else ev.internal_metrics(X_original, labels))
    # Stabilumas – tik pradinėje erdvėje. Kai visi taškai – triukšmas, ARI tarp imčių
    # formaliai lygus 1, bet nieko nereiškia, todėl neskaičiuojamas.
    pairs = (stability_pairs(subsample_runs(space, eps, min_samples))
             if space == STABILITY_SPACE and ev.n_clusters(labels) else None)

    ev.log_experiment(ALGORITHM, space, params(eps, min_samples), labels,
                      internal=internal, internal_original=internal_original,
                      stability_pairs=pairs, notes="parametrų įtaka")

    sizes = cluster_sizes(labels)
    return {
        "erdve": space, "eps_daugiklis": factor, "eps": eps, "min_samples": min_samples,
        "k": ev.n_clusters(labels),
        "branduoliniu_dalis_%": core.mean() * 100,
        "triuksmo_dalis_%": ev.noise_share(labels) * 100,
        "didziausio_dalis_%": largest_share(labels) * 100,
        "silhouette": internal["silhouette"],
        "ch": internal["ch"],
        "db": internal["db"],
        "silhouette_orig": (internal if internal_original is None
                            else internal_original)["silhouette"],
        "ari_klase": ev.external_metrics(classes, labels)["ari"],
        "stab_ari": np.nan if pairs is None else pairs["ari"].mean(),
        "stab_ari_sn": np.nan if pairs is None else pairs["ari"].std(ddof=1),
        "klasteriu_dydziai": " / ".join(map(str, sizes[:6])) + (" / …" if len(sizes) > 6 else ""),
    }


def scan_space(space, X_original):
    ids, X = load(space)
    classes = load_meta().set_index("id").loc[ids, CLASS_COLUMN].to_numpy()
    base = knee_eps(X, MIN_SAMPLES)
    rows = []
    for min_samples in MIN_SAMPLES_GRID:
        for factor in EPS_FACTORS:
            start = time.time()
            rows.append(evaluate(space, X, X_original, classes, round(base * factor, 3),
                                 min_samples, factor))
            row = rows[-1]
            print(f"  min_samples = {min_samples:2d}, eps = {row['eps']:.3f} (×{factor}): "
                  f"k = {row['k']:3d}, triukšmas = {row['triuksmo_dalis_%']:5.1f} %, "
                  f"didžiausias = {row['didziausio_dalis_%']:5.1f} %, "
                  f"ARI su klase = {row['ari_klase']:.3f}, "
                  f"stabilumas = {row['stab_ari']:.3f} ({time.time() - start:.0f} s)")
    return rows


def suggestions(table):
    """Kokį klasterių skaičių siūlo kiekvienas kriterijus. Tinklelio kriterijams imami tik
    deriniai, kuriuose k ≥ 2 ir triukšmo ne daugiau kaip 50 %."""
    knee = pd.read_csv(OUTPUT_DIR / "eps_kelio.csv")
    chosen = pd.read_csv(CHOSEN_FILE)
    rows = []

    def add(space, criterion, row, eps_column="eps", noise_scale=1.0, noise_column="triuksmo_dalis_%"):
        rows.append({"erdve": space, "kriterijus": criterion, "eps": row[eps_column],
                     "min_samples": int(row["min_samples"]), "k": int(row["k"]),
                     "triuksmo_dalis_%": row[noise_column] * noise_scale})

    for space in SPACES:
        for _, row in knee[knee["erdve"] == space].iterrows():
            add(space, f"kelio metodas, min_samples = {row['min_samples']}", row,
                eps_column="eps_kelio", noise_scale=100, noise_column="triuksmo_dalis")
        data = table[(table["erdve"] == space) & (table["k"] >= 2)
                     & (table["triuksmo_dalis_%"] <= MAX_NOISE * 100)]
        for column, criterion, best in [("silhouette", "didžiausias siluetas", "max"),
                                        ("ch", "didžiausias CH", "max"),
                                        ("db", "mažiausias DB", "min"),
                                        ("stab_ari", "stabiliausias", "max")]:
            values = data[column].dropna()
            if len(values):
                add(space, criterion, data.loc[values.idxmax() if best == "max" else values.idxmin()])
        for _, row in chosen[(chosen["erdve"] == space) & (chosen["variantas"] != "kelio")].iterrows():
            add(space, VARIANTS[row["variantas"]], row, noise_scale=100, noise_column="triuksmo_dalis")
    return pd.DataFrame(rows)


def plot_heatmaps(table):
    cmap = LinearSegmentedColormap.from_list("melyna", BLUE_RAMP)
    fig, axes = plt.subplots(len(SPACES), len(HEATMAPS), figsize=(23, 3.4 * len(SPACES)))
    for row, space in enumerate(SPACES):
        data = table[table["erdve"] == space]
        for col, (column, title, fmt) in enumerate(HEATMAPS):
            ax = axes[row, col]
            if row == 0:
                ax.set_title(title, fontsize=11)
            grid = data.pivot(index="min_samples", columns="eps", values=column)
            values = grid.to_numpy(dtype=float)
            if np.isnan(values).all():
                ax.axis("off")
                ax.text(0.5, 0.5, "Stabilumas tikrinamas\ntik pradinėje erdvėje",
                        ha="center", va="center", fontsize=10, color="#52514e",
                        transform=ax.transAxes)
                continue
            image = ax.imshow(values, cmap=cmap, aspect="auto")
            low, high = np.nanmin(values), np.nanmax(values)
            for i in range(values.shape[0]):
                for j in range(values.shape[1]):
                    value = values[i, j]
                    dark = high > low and (value - low) / (high - low) > 0.55
                    ax.text(j, i, "–" if np.isnan(value) else fmt.format(value),
                            ha="center", va="center", fontsize=8,
                            color="#ffffff" if dark else "#0b0b0b")
            ax.set_xticks(range(len(grid.columns)))
            ax.set_xticklabels([f"{eps:.2f}".replace(".", ",") for eps in grid.columns],
                               fontsize=8)
            ax.set_yticks(range(len(grid.index)))
            ax.set_yticklabels(grid.index)
            ax.set_xlabel("eps")
            ax.set_ylabel(f"{SPACE_NAMES[space]}\nmin_samples" if col == 0 else "min_samples")
            knee_column = list(grid.columns).index(data.loc[data["eps_daugiklis"] == 1.0, "eps"].iloc[0])
            ax.add_patch(plt.Rectangle((knee_column - 0.5, -0.5), 1, len(grid.index), fill=False,
                                       edgecolor=KNEE_COLOUR, linewidth=2))
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("DBSCAN parametrų įtaka (eps – kelio eps × 0,25 … 2; apvestas stulpelis – kelio eps)",
                 fontsize=13)
    fig.tight_layout()
    save_figure(fig, "parametrai.png")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _, X_original = load("originali")

    rows = []
    for space in SPACES:
        print(f"\n{SPACE_NAMES[space]}:")
        rows.extend(scan_space(space, X_original))

    table = pd.DataFrame(rows)
    table.round(4).to_csv(OUTPUT_DIR / "parametrai.csv", index=False)
    proposals = suggestions(table)
    proposals.round(3).to_csv(OUTPUT_DIR / "k_pasiulymai.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nKlasterių skaičius pagal kiekvieną kriterijų (galutinį sutariame grupėje):")
    print(proposals.round(3).to_string(index=False))
    plot_heatmaps(table)
    print("\nPastaba: siluetas skaičiuojamas be triukšmo taškų; kai k < 2, jis neapibrėžtas. "
          "Stabilumas – tik pradinėje erdvėje.")
    print("Išsaugota: rezultatai/dbscan/parametrai.csv")


if __name__ == "__main__":
    main()
