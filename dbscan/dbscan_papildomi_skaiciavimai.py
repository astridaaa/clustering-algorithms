from dbscan_common import OUTPUT_DIR, fit_dbscan, load, load_chosen

import numpy as np
import pandas as pd
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                             pairwise_distances)

import evaluation as ev
from common import load_meta
from evaluation import NOISE, RESULTS_DIR

FINAL_DIR = OUTPUT_DIR / "galutiniai"
KMEANS_LABELS = RESULTS_DIR / "kmeans" / "galutiniai" / "zymes.csv"
GROUP = 1  # pradinės erdvės kelio metodo 25 objektų (231 įrašo) klasteris


def new_cluster_after_knee(rows, chosen, X):
    """4 sk.: pirmas Eps > Eps_k, kai klasterių daugėja, ir iš ko susidaro naujas klasteris."""
    knee = chosen[("originali", "kelio")]
    scan = pd.read_csv(OUTPUT_DIR / "k_pagal_eps.csv")
    scan = scan[(scan["erdve"] == "originali") & (scan["eps"] > knee)]
    base, _ = fit_dbscan(X, knee)
    base_k = ev.n_clusters(base)
    eps = float(scan.loc[scan["k"] > base_k, "eps"].iloc[0])
    labels, _ = fit_dbscan(X, eps)
    sizes = pd.Series(labels[labels != NOISE]).value_counts()
    members = labels == sizes.index[-1]
    rows += [("4", "Eps, kai už Eps_k atsiranda papildomas klasteris", eps),
             ("4", "mažiausio klasterio dydis", int(members.sum())),
             ("4", "iš jų triukšmas, kai Eps = Eps_k", int((base[members] == NOISE).sum()))]


def pca_changes(rows, labels):
    """6.1: kiek objektų PCA 3D erdvėje pakeitė klasterį ir ARI be triukšmo."""
    original, pca = labels["originali_kelio"], labels["pca3_kelio"]
    _, _, pairs, _ = ev.matched_points(labels.index.to_numpy(), pca.to_numpy(),
                                       labels.index.to_numpy(), original.to_numpy())
    mapping = dict(zip(pairs["klasteris"], pairs["atskaitos_klasteris"]))
    changed = original != pca.map(mapping).fillna(-2)
    both = (original != NOISE) & (pca != NOISE)
    rows += [("6.1", "klasterį pakeitusių objektų", int(changed.sum())),
             ("6.1", "triukšmas -> klasteris", int(((original == NOISE) & (pca != NOISE)).sum())),
             ("6.1", "klasteris -> triukšmas", int(((original != NOISE) & (pca == NOISE)).sum())),
             ("6.1", "ARI be triukšmo objektų", adjusted_rand_score(original[both], pca[both]))]


def ami_table(labels):
    """7.2: AMI su klasėmis, ritmais ir įrašais (triukšmas – atskira grupė, kaip ARI ir NMI)."""
    meta = load_meta().set_index("id").loc[labels.index]
    return pd.DataFrame([{"suskirstymas": run, "zyme": target,
                          "ami": adjusted_mutual_info_score(meta[target], labels[run])}
                         for run in labels.columns for target in ev.META_TARGETS])


def kmeans_overlap(rows, labels):
    """5.3: t-SNE 2D varianto su 5 klasteriais sutapimas su k-vidurkių suskirstymu."""
    kmeans = pd.read_csv(KMEANS_LABELS).set_index("id").loc[labels.index, "originali_k3"]
    dbscan = labels["tsne2_k5"]
    table = pd.crosstab(kmeans, dbscan)
    for cluster, row in table.iterrows():
        best = row.idxmax()
        rows.append(("5.3", f"k-vidurkių klasteris {cluster} ({row.sum()}): daugiausia DBSCAN "
                            f"klasteryje {best} ({int((dbscan == best).sum())})", int(row.max())))
    rows.append(("5.3", "ARI tarp k-vidurkių (pradinė, k = 3) ir DBSCAN (t-SNE 2D, 5 kl.)",
                 adjusted_rand_score(kmeans, dbscan)))


def third_component(rows, labels, chosen):
    """7.3: 25 objektų grupę atskiria trečioji pagrindinė komponentė."""
    group = (labels["originali_kelio"] == GROUP).to_numpy()
    _, Y3 = load("pca3")
    rows += [("7.3", "grupės PC3 mažiausia reikšmė", Y3[group, 2].min()),
             ("7.3", "grupės PC3 didžiausia reikšmė", Y3[group, 2].max()),
             ("7.3", "kitų objektų PC3 99-asis procentilis", np.percentile(Y3[~group, 2], 99))]
    for space in ["pca3", "pca2"]:
        _, Y = load(space)
        eps = chosen[(space, "kelio")]
        _, core = fit_dbscan(Y, eps)
        distance = pairwise_distances(Y[group & core], Y[~group & core]).min()
        rows.append(("7.3", f"{space}: mažiausias atstumas tarp grupės ir kitų branduolinių "
                            f"objektų (Eps = {eps})", distance))


def main():
    chosen = load_chosen()
    labels = pd.read_csv(FINAL_DIR / "zymes.csv").set_index("id")
    _, X = load("originali")

    rows = []
    new_cluster_after_knee(rows, chosen, X)
    pca_changes(rows, labels)
    kmeans_overlap(rows, labels)
    third_component(rows, labels, chosen)
    table = pd.DataFrame(rows, columns=["poskyris", "rodiklis", "reiksme"])
    table.to_csv(OUTPUT_DIR / "papildomi_skaiciavimai.csv", index=False)

    ami = ami_table(labels)
    ami.round(4).to_csv(FINAL_DIR / "ami_zymems.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 120)
    print(table.assign(reiksme=table["reiksme"].map(lambda v: f"{v:.4g}")).to_string(index=False))
    print("\nAMI (triukšmas – atskira grupė):")
    print(ami.pivot(index="suskirstymas", columns="zyme", values="ami").round(3).to_string())
    print(f"\nIšsaugota: {OUTPUT_DIR / 'papildomi_skaiciavimai.csv'}, {FINAL_DIR / 'ami_zymems.csv'}")


if __name__ == "__main__":
    main()
