from datetime import datetime

from hier_common import (ALGORITHM, BY_K_FILE, CHOSEN, DETAILS_FILE,
                         DIMENSIONS_FILE, GRID, K_RANGE, METRICS, OUTPUT_DIR,
                         REDUCTIONS, RESULTS_FILE, SPACE_NAMES, SPACES,
                         build_tree, cut_tree_k, load, reduction_of)

import numpy as np
import pandas as pd

import evaluation as ev
from hier_metrics import (distance_matrix, internal_measures, k_by_gap,
                          recommend_dimension, size_summary)

RESULT_COLUMNS = ["erdve", "algoritmas", "jungimas", "atstumas", "k",
                  "siluetas", "CH", "DB", "ARI", "NMI", "Jaccard",
                  "stabilumas_ARI_vid", "stabilumas_ARI_sn", "random_state", "data"]


def external(labels, reference):
    if reference is None:
        return {"ARI": np.nan, "NMI": np.nan, "Jaccard": np.nan}
    values = ev.external_metrics(reference, labels)
    return {"ARI": values["ari"], "NMI": values["nmi"], "Jaccard": values["jaccard"]}


def dimension_table(details):
    rows = []
    for space in SPACES:
        c = CHOSEN[space]
        row = details[(details["erdve"] == space) & (details["jungimas"] == c["jungimas"])
                      & (details["atstumas"] == c["atstumas"]) & (details["k"] == c["k"])].iloc[0]
        rows.append({"erdve": space, "mazinimas": reduction_of(space),
                     "dimensija": 5 if space == "originali" else int(space[-1]),
                     **{name: row[name] for name in ["jungimas", "atstumas", "k", "k_tarpas",
                                                     "k_siluetas", "siluetas", "CH", "DB",
                                                     "ARI", "NMI", "Jaccard", "dydziai"]},
                     "siluetas_orig": row["siluetas"] if space == "originali" else row["siluetas_orig"],
                     "CH_orig": row["CH"] if space == "originali" else row["CH_orig"],
                     "DB_orig": row["DB"] if space == "originali" else row["DB_orig"]})
    return pd.DataFrame(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = {space: load(space) for space in SPACES}
    ids_original, X_original = data["originali"]
    for space in SPACES:
        assert np.array_equal(data[space][0], ids_original)

    by_k_rows, detail_rows = [], []
    for metric in METRICS:
        methods = [method for method, m in GRID if m == metric]
        D_original = distance_matrix(X_original, metric)
        references = {}
        for space in SPACES:
            X = data[space][1]
            D = D_original if space == "originali" else distance_matrix(X, metric)
            for method in methods:
                print(f"{space:9s} {method:8s} {metric}")
                Z = build_tree(X, method, metric)
                k_gap, gaps = k_by_gap(Z)
                labels_by_k = {k: cut_tree_k(Z, k) for k in K_RANGE}
                reference = references.get(method)

                for k, labels in labels_by_k.items():
                    row = {"erdve": space, "jungimas": method, "atstumas": metric, "k": k,
                           "tarpas": gaps[k],
                           **internal_measures(X, labels, D=D),
                           **external(labels, reference)}
                    if space != "originali":
                        orig = internal_measures(X_original, labels, D=D_original)
                        row.update({f"{name}_orig": value for name, value in orig.items()})
                    row.update(size_summary(labels))
                    by_k_rows.append(row)

                by_k = pd.DataFrame([r for r in by_k_rows if r["erdve"] == space
                                     and r["jungimas"] == method and r["atstumas"] == metric])
                k_silhouette = int(by_k.loc[by_k["siluetas"].idxmax(), "k"])
                chosen = CHOSEN[space]
                ks = [k_gap]
                if (chosen["jungimas"], chosen["atstumas"]) == (method, metric) \
                        and chosen["k"] != k_gap:
                    ks.append(chosen["k"])
                for k in ks:
                    detail_rows.append({
                        **by_k[by_k["k"] == k].iloc[0].to_dict(),
                        "k_tarpas": k_gap, "k_siluetas": k_silhouette,
                        "k_saltinis": "tarpas" if k == k_gap else "pasirinktas",
                        "siluetas_ties_k_siluetas": by_k["siluetas"].max(),
                    })

                if space == "originali":
                    references[method] = labels_by_k[k_gap]
        del D_original

    by_k = pd.DataFrame(by_k_rows)
    by_k.round(4).to_csv(BY_K_FILE, index=False)

    details = pd.DataFrame(detail_rows)
    order = {space: i for i, space in enumerate(SPACES)}
    details = details.sort_values(["erdve", "jungimas", "atstumas", "k"],
                                  key=lambda c: c.map(order) if c.name == "erdve" else c,
                                  ignore_index=True)
    details.round(4).to_csv(DETAILS_FILE, index=False)

    previous = pd.read_csv(RESULTS_FILE) if RESULTS_FILE.exists() else None
    results = details.assign(algoritmas=ALGORITHM, stabilumas_ARI_vid=np.nan,
                             stabilumas_ARI_sn=np.nan, random_state=np.nan,
                             data=datetime.now().strftime("%Y-%m-%d %H:%M"))[RESULT_COLUMNS]
    if previous is not None:
        key = ["erdve", "jungimas", "atstumas", "k"]
        old = previous.set_index(key)[["stabilumas_ARI_vid", "stabilumas_ARI_sn"]]
        results = results.set_index(key)
        results.update(old, overwrite=True)
        results = results.reset_index()[RESULT_COLUMNS]
    results.round(4).to_csv(RESULTS_FILE, index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    print("\nParametrų tinklelis (k – didžiausio tarpo taisyklė; išoriniai matai – "
          "lyginant su ta pačia konfigūracija pradinėje erdvėje):")
    columns = ["erdve", "jungimas", "atstumas", "k", "k_siluetas", "siluetas", "CH", "DB",
               "ARI", "NMI", "Jaccard", "siluetas_orig", "didziausio_dalis", "dydziai"]
    print(details[columns].round(3).to_string(index=False))
    print("\nPastaba: t-SNE vidiniai matai dirbtinai išpūsti; t-SNE lyginkite pagal "
          "ARI/NMI/Jaccard ir siluetas_orig.")
    dims = dimension_table(details)
    if DIMENSIONS_FILE.exists():
        old = pd.read_csv(DIMENSIONS_FILE)
        keep = [c for c in old.columns if c.startswith("stab")]
        if keep:
            dims = dims.merge(old[["erdve", "jungimas", "atstumas", "k", *keep]],
                              on=["erdve", "jungimas", "atstumas", "k"], how="left")
    dims = recommend_dimension(dims, REDUCTIONS)
    dims.round(4).to_csv(DIMENSIONS_FILE, index=False)
    print("\nDimensijų palyginimas (pasirinkta konfigūracija kiekvienoje erdvėje; "
          "lyginama pagal siluetas_orig ir ARI su pradine):")
    print(dims[["erdve", "jungimas", "atstumas", "k", "k_tarpas", "k_siluetas", "siluetas",
                "siluetas_orig", "ARI", "NMI", "Jaccard", "dydziai", "rekomenduojama"]]
          .round(3).to_string(index=False))
    print("PCA 5D = pradinės erdvės pasukimas (Euklido atstumai identiški).")
    print(f"Išsaugota: {RESULTS_FILE.relative_to(ev.ROOT)}, {DETAILS_FILE.name}, "
          f"{BY_K_FILE.name}, {DIMENSIONS_FILE.name}")


if __name__ == "__main__":
    main()
