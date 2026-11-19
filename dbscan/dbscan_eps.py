from dbscan_common import (BLUE_RAMP, CHOSEN_FILE, MIN_SAMPLES, OUTPUT_DIR,
                           PALETTE, SPACE_NAMES, SPACES, TARGET_K, VARIANTS,
                           cluster_sizes, eps_for_k, eps_scan, fit_dbscan,
                           k_distances, knee_index, largest_share, load, save_figure)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import evaluation as ev

MIN_SAMPLES_GRID = [5, 10, 20]
ZOOM_FROM = 0.85
COLOURS = {5: BLUE_RAMP[2], 10: BLUE_RAMP[4], 20: BLUE_RAMP[6]}
KNEE_COLOUR = PALETTE[1]


def scan_space(space):
    ids, X = load(space)
    curves, rows = {}, []
    for min_samples in MIN_SAMPLES_GRID:
        distances = k_distances(X, min_samples)
        knee = knee_index(distances)
        eps = round(float(distances[knee]), 3)
        labels, core = fit_dbscan(X, eps, min_samples)
        curves[min_samples] = (distances, knee)
        sizes = cluster_sizes(labels)
        rows.append({
            "erdve": space, "min_samples": min_samples, "eps_kelio": eps,
            "kelio_vieta": knee / (len(distances) - 1),
            "k": ev.n_clusters(labels),
            "branduoliniu_dalis": core.mean(),
            "triuksmo_dalis": ev.noise_share(labels),
            "didziausio_dalis": largest_share(labels),
            "klasteriu_dydziai": " / ".join(map(str, sizes[:6])) + (" / …" if len(sizes) > 6 else ""),
        })
    return curves, rows


def plot_curves(all_curves):
    fig, axes = plt.subplots(2, len(SPACES), figsize=(5.3 * len(SPACES), 8))
    for col, space in enumerate(SPACES):
        for row, start in enumerate([0.0, ZOOM_FROM]):
            ax = axes[row, col]
            for min_samples, (distances, knee) in all_curves[space].items():
                share = np.linspace(0, 100, len(distances))
                keep = share >= start * 100
                ax.plot(share[keep], distances[keep], color=COLOURS[min_samples],
                        label=f"min_samples = {min_samples}")
                ax.plot(share[knee], distances[knee], "o", color=KNEE_COLOUR, markersize=5)
            chosen = all_curves[space][MIN_SAMPLES]
            eps = chosen[0][chosen[1]]
            ax.axhline(eps, color=KNEE_COLOUR, linestyle="--", linewidth=1,
                       label=f"eps = {eps:.3f} (min_samples = {MIN_SAMPLES})")
            if row == 1:
                visible = chosen[0][int(start * (len(chosen[0]) - 1)):]
                ax.set_ylim(0, np.quantile(visible, 0.95) * 1.5)
                ax.set_xlabel("Taškai, surikiuoti pagal k-atstumą, %")
            ax.set_title(f"{SPACE_NAMES[space]}" + (" (paskutiniai 15 % taškų)" if row else ""),
                         fontsize=11)
            ax.set_ylabel("k-atstumas")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8, frameon=False)
    fig.suptitle("k-atstumų grafikai: oranžinis taškas – kelias, punktyras – pasirinktas eps",
                 fontsize=13)
    fig.tight_layout()
    save_figure(fig, "k_atstumai.png")


def plot_scans(scans, chosen):
    """Klasterių skaičius ir triukšmo dalis pagal eps (dvi eilutės, ne dvi y ašys)."""
    fig, axes = plt.subplots(2, len(SPACES), figsize=(5.3 * len(SPACES), 7), sharex="col")
    for col, space in enumerate(SPACES):
        scan = scans[space]
        top, bottom = axes[0, col], axes[1, col]
        top.plot(scan["eps"], scan["k"], color=BLUE_RAMP[4], linewidth=1.5)
        for k in sorted(set(TARGET_K.values())):
            top.axhline(k, color="#898781", linestyle=":", linewidth=1, label=f"k = {k}")
        bottom.plot(scan["eps"], scan["triuksmo_dalis"] * 100, color=BLUE_RAMP[4], linewidth=1.5)
        for i, variant in enumerate(VARIANTS):
            colour, style = PALETTE[(1 + i) % len(PALETTE)], "--" if i == 0 else "-."
            eps = chosen[(space, variant)]
            for ax in (top, bottom):
                ax.axvline(eps, color=colour, linestyle=style, linewidth=1.2,
                           label=f"{variant}: eps = {eps:.3f}" if ax is top else None)
        top.set_yscale("log")
        top.set_yticks([1, 2, 5, 10, 20, 50, 100])
        top.set_yticklabels(["1", "2", "5", "10", "20", "50", "100"])
        top.set_title(SPACE_NAMES[space], fontsize=11)
        top.set_ylabel("Klasterių skaičius")
        bottom.set_ylabel("Triukšmo dalis, %")
        bottom.set_xlabel(f"eps (min_samples = {MIN_SAMPLES})")
        for ax in (top, bottom):
            ax.grid(alpha=0.3)
        top.legend(fontsize=8, frameon=False)
    fig.suptitle("Kaip klasterių skaičius ir triukšmas priklauso nuo eps", fontsize=13)
    fig.tight_layout()
    save_figure(fig, "k_pagal_eps.png")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_curves, rows = {}, []
    for space in SPACES:
        all_curves[space], space_rows = scan_space(space)
        rows.extend(space_rows)

    table = pd.DataFrame(rows)
    table.round(4).to_csv(OUTPUT_DIR / "eps_kelio.csv", index=False)
    pd.set_option("display.width", 250)
    print(table.round(3).to_string(index=False))
    plot_curves(all_curves)

    scans, chosen, chosen_rows = {}, {}, []
    for space in SPACES:
        _, X = load(space)
        scans[space] = eps_scan(X, MIN_SAMPLES)
        knee = table[(table["erdve"] == space) & (table["min_samples"] == MIN_SAMPLES)]["eps_kelio"].iloc[0]
        chosen[(space, "kelio")] = knee
        for name, target in TARGET_K.items():
            chosen[(space, name)], found = eps_for_k(scans[space], target)
            if found != target:
                print(f"{SPACE_NAMES[space]}: lygiai {target} klasterių negaunama, "
                      f"imamas artimiausias – {found}.")
        for variant in VARIANTS:
            eps = chosen[(space, variant)]
            labels, _ = fit_dbscan(X, eps, MIN_SAMPLES)
            chosen_rows.append({"erdve": space, "variantas": variant, "eps": eps,
                                "min_samples": MIN_SAMPLES, "k": ev.n_clusters(labels),
                                "triuksmo_dalis": ev.noise_share(labels),
                                "klasteriu_dydziai": " / ".join(map(str, cluster_sizes(labels)[:8]))})

    pd.concat([scan.assign(erdve=space) for space, scan in scans.items()]).round(4).to_csv(
        OUTPUT_DIR / "k_pagal_eps.csv", index=False)
    chosen_table = pd.DataFrame(chosen_rows)
    chosen_table.round(4).to_csv(CHOSEN_FILE, index=False)
    print(f"\nPasirinkti parametrai (min_samples = {MIN_SAMPLES}):")
    print(chosen_table.round(3).to_string(index=False))
    plot_scans(scans, chosen)


if __name__ == "__main__":
    main()
