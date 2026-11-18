import json
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score,
                             normalized_mutual_info_score, silhouette_score)
from sklearn.metrics.cluster import contingency_matrix, pair_confusion_matrix

from common import common_labels, load_meta
from config import ROOT

NOISE = -1

RESULTS_DIR = ROOT / "rezultatai"

PREPARATION = "RobustScaler, 4835 dūžiai (be 2 RR_vid klaidų), išskirtys paliktos"

LOG_COLUMNS = [
    "data", "algoritmas", "erdve", "rinkinys", "parametrai", "k",
    "random_state", "paruosimas", "objektu", "triuksmo_dalis",
    "silhouette", "ch", "db",
    "silhouette_orig", "ch_orig", "db_orig",
    "ari", "nmi", "jaccard",
    "stab_n", "stab_ari", "stab_ari_sn", "stab_nmi", "stab_nmi_sn",
    "stab_jaccard", "stab_jaccard_sn",
    "pastabos",
]

LOG_KEY = ["algoritmas", "erdve", "rinkinys", "parametrai", "random_state"]

def empirical_k(m):
    """Empyrinis klasterių skaičius k ≈ sqrt(m / 2)."""
    return int(round(np.sqrt(m / 2)))

def within_cluster_variance(X, labels):
    """var(k) = sum_l (1 / m_l) sum_j ||X_j^l - C^l||^2 (alkūnės metodas).

    Tai ataskaitoje ir skaidrėse apibrėžta klasterių vidinių dispersijų suma:
    kiekvieno klasterio kvadratinių atstumų iki centro suma dalinama iš
    klasterio dydžio. Ji skiriasi nuo sklearn KMeans.inertia_, kuri yra
    E_k-means (be dalybos iš m_l). Triukšmo taškai neįtraukiami.
    """
    X, labels = _without_noise(np.asarray(X), np.asarray(labels))
    total = 0.0
    for label in np.unique(labels):
        members = X[labels == label]
        total += ((members - members.mean(axis=0)) ** 2).sum(axis=1).mean()
    return total

def n_clusters(labels):
    """Klasterių skaičius be triukšmo."""
    return len(set(np.asarray(labels).tolist()) - {NOISE})

def noise_share(labels):
    return float(np.mean(np.asarray(labels) == NOISE))

def _without_noise(X, labels):
    keep = labels != NOISE
    return X[keep], labels[keep]

def internal_metrics(X, labels, metric="euclidean"):
    X, labels = _without_noise(np.asarray(X), np.asarray(labels))
    k = len(np.unique(labels))
    if not 2 <= k <= len(labels) - 1:
        return {"silhouette": np.nan, "ch": np.nan, "db": np.nan}
    return {
        "silhouette": silhouette_score(X, labels, metric=metric),
        "ch": calinski_harabasz_score(X, labels),
        "db": davies_bouldin_score(X, labels),
    }


def jaccard_index(labels_u, labels_v):
    (_, fp), (fn, tp) = pair_confusion_matrix(labels_u, labels_v)
    denominator = tp + fp + fn
    return float(tp / denominator) if denominator else 1.0


def external_metrics(labels_u, labels_v):
    return {
        "ari": adjusted_rand_score(labels_u, labels_v),
        "nmi": normalized_mutual_info_score(labels_u, labels_v,
                                            average_method="arithmetic"),
        "jaccard": jaccard_index(labels_u, labels_v),
    }


def compare_with_reference(ids, labels, ref_ids, ref_labels):
    u, v = common_labels(ref_ids, ref_labels, ids, labels)
    return external_metrics(u, v)


def matched_points(ids, labels, ref_ids, ref_labels):
    ref, other = common_labels(ref_ids, ref_labels, ids, labels)
    ref_names, ref_codes = np.unique(ref, return_inverse=True)
    names, codes = np.unique(other, return_inverse=True)
    table = contingency_matrix(ref_codes, codes)

    ref_noise = np.flatnonzero(ref_names == NOISE)
    noise = np.flatnonzero(names == NOISE)
    ref_rest = np.flatnonzero(ref_names != NOISE)
    rest = np.flatnonzero(names != NOISE)

    sub_rows, sub_cols = linear_sum_assignment(table[np.ix_(ref_rest, rest)],
                                               maximize=True)
    rows, cols = list(ref_rest[sub_rows]), list(rest[sub_cols])
    if len(ref_noise) and len(noise):
        rows.append(ref_noise[0])
        cols.append(noise[0])
    rows, cols = np.array(rows, dtype=int), np.array(cols, dtype=int)

    together = table[rows, cols]
    ref_sizes = table.sum(axis=1)[rows]
    sizes = table.sum(axis=0)[cols]
    pairs = pd.DataFrame({
        "atskaitos_klasteris": ref_names[rows],
        "klasteris": names[cols],
        "sutampa": together,
        "atskaitos_dydis": ref_sizes,
        "dydis": sizes,
        "persidengimas": together / (ref_sizes + sizes - together),
    }).sort_values("atskaitos_klasteris", ignore_index=True)

    crosstab = pd.DataFrame(table, index=pd.Index(ref_names, name="atskaita"),
                            columns=pd.Index(names, name="klasteris"))
    matched = int(together.sum())
    return matched, matched / len(ref), pairs, crosstab

META_TARGETS = ["klase", "ritmas", "irasas"]

def _meta_for(ids):
    return load_meta().set_index("id").loc[ids]

def compare_with_meta(ids, labels):
    meta = _meta_for(ids)
    return pd.DataFrame([{"zyme": column,
                          **external_metrics(meta[column].to_numpy(), labels)}
                         for column in META_TARGETS])


def cluster_composition(ids, labels, column="klase", normalize=True):
    meta = _meta_for(ids)
    return pd.crosstab(pd.Series(labels, name="klasteris"),
                       pd.Series(meta[column].to_numpy(), name=column),
                       normalize="index" if normalize else False)


def outlier_share(ids, labels):
    meta = _meta_for(ids)
    outliers = pd.Series(meta["ekstremali_isskirtis"].to_numpy(), name="isskirtis")
    return outliers.groupby(pd.Series(labels, name="klasteris")).agg(
        objektu="size", isskirciu="sum", dalis="mean")

def stability(samples, metric="euclidean"):
    pair_rows = []
    for seed_a, seed_b in combinations(sorted(samples), 2):
        ids_a, _, labels_a = samples[seed_a]
        ids_b, _, labels_b = samples[seed_b]
        u, v = common_labels(ids_a, labels_a, ids_b, labels_b)
        pair_rows.append({"imtis_a": seed_a, "imtis_b": seed_b,
                          "bendru": len(u), **external_metrics(u, v)})

    sample_rows = []
    for seed in sorted(samples):
        _, X, labels = samples[seed]
        sample_rows.append({"imtis": seed, "k": n_clusters(labels),
                            "triuksmo_dalis": noise_share(labels),
                            **internal_metrics(X, labels, metric=metric)})

    return pd.DataFrame(pair_rows), pd.DataFrame(sample_rows)


def summarize(table, columns):
    return {column: (table[column].mean(), table[column].std(ddof=1))
            for column in columns}

def format_mean_sd(mean, sd, digits=3):
    return f"{mean:.{digits}f} ± {sd:.{digits}f}".replace(".", ",")


def stability_vs_n(pairs, ns=(5, 10, 15, 20), column="ari"):
    seeds = set(pairs["imtis_a"]) | set(pairs["imtis_b"])
    rows = []
    for n in ns:
        if not set(range(n)) <= seeds:
            continue
        subset = pairs[(pairs["imtis_a"] < n) & (pairs["imtis_b"] < n)]
        rows.append({"N": n, "poru": len(subset),
                     "vidurkis": subset[column].mean(),
                     "sn": subset[column].std(ddof=1)})
    return pd.DataFrame(rows)


def experiments_file(algorithm):
    name = algorithm.lower().replace(" ", "_")
    return RESULTS_DIR / f"eksperimentai_{name}.csv"


def log_experiment(algorithm, space, params, labels, random_state=None,
                   feature_set="visi", internal=None, internal_original=None,
                   external=None, stability_pairs=None, notes="",
                   preparation=PREPARATION, path=None):
    path = path or experiments_file(algorithm)

    row = dict.fromkeys(LOG_COLUMNS, np.nan)
    row.update({
        "data": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "algoritmas": algorithm,
        "erdve": space,
        "rinkinys": feature_set,
        "parametrai": json.dumps(params, ensure_ascii=False, sort_keys=True),
        "k": n_clusters(labels),
        "random_state": random_state,
        "paruosimas": preparation,
        "objektu": len(labels),
        "triuksmo_dalis": noise_share(labels),
        "pastabos": notes,
    })
    if internal:
        row.update(internal)
    if internal_original:
        row.update({f"{name}_orig": value for name, value in internal_original.items()})
    if external:
        row.update(external)
    if stability_pairs is not None and len(stability_pairs):
        row["stab_n"] = len(set(stability_pairs["imtis_a"]) | set(stability_pairs["imtis_b"]))
        for name, (mean, sd) in summarize(stability_pairs, ["ari", "nmi", "jaccard"]).items():
            row[f"stab_{name}"] = mean
            row[f"stab_{name}_sn"] = sd

    new = pd.DataFrame([row], columns=LOG_COLUMNS)
    if path.exists():
        log = pd.read_csv(path)
        same = (_log_key(log) == _log_key(new).iloc[0])
        log = pd.concat([log[~same], new], ignore_index=True)
    else:
        log = new

    path.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(path, index=False)
    return row


def _log_key(table):
    key = table[LOG_KEY].copy()
    key["random_state"] = pd.to_numeric(key["random_state"], errors="coerce").astype("Int64")
    return key.astype(str).agg("|".join, axis=1)


def load_experiments():
    files = sorted(RESULTS_DIR.glob("eksperimentai_*.csv"))
    if not files:
        return pd.DataFrame(columns=LOG_COLUMNS)
    return pd.concat([pd.read_csv(file) for file in files], ignore_index=True)