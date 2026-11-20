from itertools import combinations

from hier_common import (CHOSEN, DIMENSIONS_FILE, GRID, OUTPUT_DIR,
                         RESULTS_FILE, SPACES, build_tree, cut_tree_k,
                         load_space_subsample, seeds_for)

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

import evaluation as ev
from common import common_labels
from hier_metrics import k_by_gap, recommend_dimension

STABILITY_DIR = OUTPUT_DIR / "stabilumas"


def chosen_stability(space):
    config = CHOSEN[space]
    method, metric, k = config["jungimas"], config["atstumas"], config["k"]
    samples, gap_ks = {}, {}
    for seed in seeds_for(space):
        ids, X = load_space_subsample(space, seed)
        Z = build_tree(X, method, metric)
        samples[seed] = (ids, X, cut_tree_k(Z, k))
        gap_ks[seed] = k_by_gap(Z)[0]

    pairs, per_sample = ev.stability(samples, metric=metric)
    per_sample["k_tarpas"] = per_sample["imtis"].map(gap_ks)
    per_sample["dydziai"] = [" / ".join(map(str, np.bincount(samples[s][2])))
                             for s in per_sample["imtis"]]
    by_n = ev.stability_vs_n(pairs, ns=[n for n in (5, 10, 15, 20) if n <= len(samples)])

    stab = ev.summarize(pairs, ["ari", "nmi", "jaccard"])
    internal = ev.summarize(per_sample, ["silhouette", "ch", "db"])
    summary = {"erdve": space, "jungimas": method, "atstumas": metric, "k": k,
               "N": len(samples), "poru": len(pairs)}
    for name, label in [("ari", "ARI"), ("nmi", "NMI"), ("jaccard", "Jaccard"),
                        ("silhouette", "siluetas"), ("ch", "CH"), ("db", "DB")]:
        mean, sd = stab[name] if name in stab else internal[name]
        summary[f"{label}_vid"], summary[f"{label}_sn"] = mean, sd
    summary["k_tarpas_sutampa"] = np.mean([gap_ks[s] == k for s in gap_ks])

    tag = {"erdve": space}
    return (pairs.assign(**tag), per_sample.assign(**tag), by_n.assign(**tag), summary)


def grid_stability(space, rows):
    labels = {}
    wanted = rows.groupby(["jungimas", "atstumas"])["k"].apply(list).to_dict()
    for seed in seeds_for(space):
        ids, X = load_space_subsample(space, seed)
        for method, metric in GRID:
            if (method, metric) not in wanted:
                continue
            Z = build_tree(X, method, metric)
            for k in wanted[(method, metric)]:
                labels.setdefault((method, metric, k), {})[seed] = (ids, cut_tree_k(Z, k))

    out = {}
    for key, by_seed in labels.items():
        values = [adjusted_rand_score(*common_labels(*by_seed[a], *by_seed[b]))
                  for a, b in combinations(sorted(by_seed), 2)]
        out[key] = (np.mean(values), np.std(values, ddof=1))
    return out


def main():
    STABILITY_DIR.mkdir(parents=True, exist_ok=True)
    if not RESULTS_FILE.exists():
        raise SystemExit("Pirmiausia paleiskite hier_params.py")
    results = pd.read_csv(RESULTS_FILE)

    pairs, per_sample, by_n, summaries = [], [], [], []
    for space in SPACES:
        print(f"{space}: pasirinkta konfigūracija {CHOSEN[space]}...")
        p, s, n, summary = chosen_stability(space)
        pairs.append(p)
        per_sample.append(s)
        by_n.append(n)
        summaries.append(summary)

        print(f"{space}: visas tinklelis...")
        stab = grid_stability(space, results[results["erdve"] == space])
        for (method, metric, k), (mean, sd) in stab.items():
            mask = ((results["erdve"] == space) & (results["jungimas"] == method)
                    & (results["atstumas"] == metric) & (results["k"] == k))
            results.loc[mask, ["stabilumas_ARI_vid", "stabilumas_ARI_sn"]] = mean, sd

    pd.concat(pairs, ignore_index=True).round(4).to_csv(
        STABILITY_DIR / "stabilumas_poros.csv", index=False)
    pd.concat(per_sample, ignore_index=True).round(4).to_csv(
        STABILITY_DIR / "stabilumas_imtys.csv", index=False)
    by_n = pd.concat(by_n, ignore_index=True)
    by_n.round(4).to_csv(STABILITY_DIR / "stabilumas_pagal_n.csv", index=False)
    summary = pd.DataFrame(summaries)
    summary.round(4).to_csv(STABILITY_DIR / "stabilumas_santrauka.csv", index=False)
    results.round(4).to_csv(RESULTS_FILE, index=False)

    dims = pd.read_csv(DIMENSIONS_FILE)
    dims = dims[[c for c in dims.columns if not c.startswith("stab")]]
    stab_columns = {f"{name}_{stat}": f"stab_{name}_{stat}"
                    for name in ("ARI", "NMI", "Jaccard") for stat in ("vid", "sn")}
    dims = dims.merge(summary[["erdve", "jungimas", "atstumas", "k", *stab_columns]]
                      .rename(columns=stab_columns), on=["erdve", "jungimas", "atstumas", "k"],
                      how="left")
    dims = recommend_dimension(dims)
    dims.round(4).to_csv(DIMENSIONS_FILE, index=False)

    pd.set_option("display.width", 250)
    print("\nStabilumas (pasirinkta konfigūracija, vid. ± SN per imčių poras / imtis):")
    for row in summaries:
        print(f"  {row['erdve']:9s} {row['jungimas']}, {row['atstumas']}, k = {row['k']} "
              f"(N = {row['N']}, porų {row['poru']}):")
        print("    " + "; ".join(f"{name} {ev.format_mean_sd(row[f'{name}_vid'], row[f'{name}_sn'])}"
                                 for name in ("ARI", "NMI", "Jaccard", "siluetas", "CH", "DB")))
        print(f"    tarpo taisyklė imtyje duoda tą patį k: {row['k_tarpas_sutampa']:.0%}")
    print("\nDimensijų palyginimas su stabilumu:")
    print(dims[["erdve", "k", "siluetas_orig", "ARI", "stab_ARI_vid", "stab_ARI_sn",
                "rekomenduojama"]].round(3).to_string(index=False))
    print("\nARI vidurkis pagal N:")
    print(by_n.pivot(index="N", columns="erdve", values="vidurkis").round(3).to_string())
    print("\nVisas tinklelis – stabilumas (ARI vid.):")
    print(results.pivot_table(index=["jungimas", "atstumas"], columns="erdve",
                              values="stabilumas_ARI_vid").round(3).to_string())
    print("Pastaba: aukštas ARI degeneruotiems variantams (vienas milžiniškas klasteris) "
          "nereiškia gerų klasterių – tik tai, kad tos pačios išskirtys atskiriamos kas kartą.")


if __name__ == "__main__":
    main()
