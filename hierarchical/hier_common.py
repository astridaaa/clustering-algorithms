import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from common import load_projection, load_subsample
from config import (DIMENSION_SPACE_NAMES, DIMENSION_SPACES, N_SUBSAMPLES,
                    TSNE_PARAMS)
from evaluation import RESULTS_DIR

OUTPUT_DIR = RESULTS_DIR / "hierarchinis"
FIGURES_DIR = OUTPUT_DIR / "grafikai"
PROFILES_DIR = OUTPUT_DIR / "profiliai"

RESULTS_FILE = OUTPUT_DIR / "rezultatai.csv"
DETAILS_FILE = OUTPUT_DIR / "parametrai_detaliai.csv"
BY_K_FILE = OUTPUT_DIR / "matai_pagal_k.csv"
LABELS_FILE = OUTPUT_DIR / "zymes.csv"

SUBSAMPLE_CACHE_DIR = OUTPUT_DIR / "imtys"
DIMENSIONS_FILE = OUTPUT_DIR / "dimensijos.csv"

ALGORITHM = "hierarchinis"

SPACES = DIMENSION_SPACES
SPACE_NAMES = DIMENSION_SPACE_NAMES
REDUCTIONS = ["pca", "tsne"]

LINKAGES = ["single", "complete", "average", "centroid", "ward"]
METRICS = ["euclidean", "cityblock", "chebyshev"]
EUCLIDEAN_ONLY = {"centroid", "ward"}

LINKAGE_NAMES = {"single": "artimiausio kaimyno", "complete": "tolimiausio kaimyno",
                 "average": "vidutinis", "centroid": "centroidų", "ward": "Ward"}
METRIC_NAMES = {"euclidean": "Euklido", "cityblock": "Manheteno", "chebyshev": "Čebyševo"}

K_RANGE = range(2, 11)

GRID = [(method, metric) for method in LINKAGES for metric in METRICS
        if method not in EUCLIDEAN_ONLY or metric == "euclidean"]

CHOSEN = {space: {"jungimas": "ward", "atstumas": "euclidean",
                  "k": 2 if space.startswith("tsne") else 3}
          for space in SPACES}

BEST_DIMENSION = {"pca": "pca2", "tsne": "tsne3"}
FINAL_SPACES = ["originali"] + [BEST_DIMENSION[r] for r in REDUCTIONS]


def reduction_of(space):
    return space.rstrip("0123456789")


def load(space):
    return load_projection(space)


def seeds_for(space):
    return range(N_SUBSAMPLES[reduction_of(space)])


PREPARED_SUBSAMPLES = {"originali": "originali", "pca3": "pca", "tsne2": "tsne"}


def load_space_subsample(space, seed):
    if space in PREPARED_SUBSAMPLES:
        return load_subsample(PREPARED_SUBSAMPLES[space], seed)
    ids, X = load_subsample("originali", seed)
    method, dim = reduction_of(space), int(space[len(reduction_of(space)):])
    if method == "pca":
        return ids, PCA(n_components=dim).fit_transform(X)
    path = SUBSAMPLE_CACHE_DIR / f"{space}_imtis{seed}.csv"
    if not path.exists():
        model = TSNE(n_components=dim, random_state=seed,
                     method="barnes_hut" if dim <= 3 else "exact", **TSNE_PARAMS)
        table = pd.DataFrame(model.fit_transform(X))
        table.insert(0, "id", ids)
        SUBSAMPLE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False, float_format="%.6f")
    data = pd.read_csv(path)
    return data["id"].to_numpy(), data.drop(columns="id").to_numpy()


def build_tree(X, method, metric):
    if method in EUCLIDEAN_ONLY and metric != "euclidean":
        raise ValueError(f"{method} apibrėžtas tik Euklido atstumui")
    return linkage(X, method=method, metric=metric)


def cut_tree_k(Z, k):
    n = len(Z) + 1
    members = {i: [i] for i in range(n)}
    for step in range(n - k):
        a, b = int(Z[step, 0]), int(Z[step, 1])
        merged = members.pop(a)
        merged.extend(members.pop(b))
        members[n + step] = merged
    groups = sorted(members.values(), key=len, reverse=True)
    labels = np.empty(n, dtype=int)
    for label, group in enumerate(groups):
        labels[group] = label
    return labels


def cut_height(Z, k):
    n = len(Z) + 1
    return (Z[n - k - 1, 2] + Z[n - k, 2]) / 2


def config_title(method, metric, k=None):
    title = f"{LINKAGE_NAMES[method]}, {METRIC_NAMES[metric]}"
    return f"{title}, k = {k}" if k is not None else title


def save_figure(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Išsaugota: {path.relative_to(ROOT)}")
