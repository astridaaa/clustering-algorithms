import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault("OMP_NUM_THREADS", "4")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from common import load_projection, load_space, load_subsample
from config import (DIMENSION_SPACE_NAMES, DIMENSION_SPACES, N_SUBSAMPLES,
                    TSNE_PARAMS)
from evaluation import RESULTS_DIR

OUTPUT_DIR = RESULTS_DIR / "kmeans"
FIGURES_DIR = OUTPUT_DIR / "grafikai"
DIMENSIONS_FILE = OUTPUT_DIR / "dimensijos.csv"

ALGORITHM = "k-means"
RANDOM_STATE = 0

# 7 erdvės, kaip ir kitų algoritmų: pradinė (5D), PCA ir t-SNE 2D, 3D, 5D.
SPACES = DIMENSION_SPACES
SPACE_NAMES = DIMENSION_SPACE_NAMES
REDUCTIONS = ["pca", "tsne"]
REFERENCE = "originali"

# Pasirinktas k kiekvienoje erdvėje (žr. kmeans_k.py rezultatus, k_pasiulymai.csv):
#   pradinė ir PCA – beveik visi kriterijai rodo k = 3;
#   t-SNE 2D – alkūnė ir stabilumas k = 4;
#   t-SNE 3D – siluetas, DB ir stabilumas k = 8;
#   t-SNE 5D – siluetas, CH ir stabilumas k = 2.
CHOSEN_K = {"originali": 3,
            "pca2": 3, "pca3": 3, "pca5": 3,
            "tsne2": 4, "tsne3": 8, "tsne5": 2}

# Imtys, kurios jau paruoštos prepare.py (tos pačios dimensijos).
PREPARED_SUBSAMPLES = {"originali": "originali", "pca3": "pca", "tsne2": "tsne"}
# t-SNE 3D ir 5D imtis jau sukūrė hierarchinio klasterizavimo skriptai – naudojamos tos
# pačios, kad visi algoritmai būtų tikrinami su tomis pačiomis imtimis.
SHARED_SUBSAMPLE_DIR = RESULTS_DIR / "hierarchinis" / "imtys"
OWN_SUBSAMPLE_DIR = OUTPUT_DIR / "imtys"

VARIANTS = {
    "k-vidurkiai, Euklido": ("kmeans", "euclidean"),
    "k-medoidai, Euklido": ("kmedoids", "euclidean"),
    "k-medoidai, Manheteno": ("kmedoids", "cityblock"),
    "k-medoidai, Čebyševo": ("kmedoids", "chebyshev"),
}

SKLEARN_METRIC = {"euclidean": "euclidean", "cityblock": "manhattan",
                  "chebyshev": "chebyshev"}


# --- Erdvės -----------------------------------------------------------------

def reduction_of(space):
    return space.rstrip("0123456789")


def dimension_of(space):
    return 5 if space == REFERENCE else int(space[len(reduction_of(space)):])


def load(space):
    """Visa aibė erdvėje: (ids, X)."""
    return load_projection(space)


def seeds_for(space):
    return range(N_SUBSAMPLES[reduction_of(space)])


def load_space_subsample(space, seed):
    """80 % imtis `seed` erdvėje `space`: (ids, X).

    Imtis normuota iš naujo (prepare.py), o PCA ir t-SNE taikomi kiekvienai imčiai
    atskirai – kaip hierarchinio klasterizavimo skriptuose."""
    if space in PREPARED_SUBSAMPLES:
        return load_subsample(PREPARED_SUBSAMPLES[space], seed)
    ids, X = load_subsample(REFERENCE, seed)
    method, dim = reduction_of(space), dimension_of(space)
    if method == "pca":
        return ids, PCA(n_components=dim).fit_transform(X)

    name = f"{space}_imtis{seed}.csv"
    path = SHARED_SUBSAMPLE_DIR / name
    if not path.exists():
        path = OWN_SUBSAMPLE_DIR / name
    if not path.exists():
        model = TSNE(n_components=dim, random_state=seed,
                     method="barnes_hut" if dim <= 3 else "exact", **TSNE_PARAMS)
        table = pd.DataFrame(model.fit_transform(X))
        table.insert(0, "id", ids)
        OWN_SUBSAMPLE_DIR.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False, float_format="%.6f")
    data = pd.read_csv(path)
    return data["id"].to_numpy(), data.drop(columns="id").to_numpy()


def load_best_dimensions():
    """kmeans_dimensions.py parinktos dimensijos: {"pca": "pca2", "tsne": "tsne5"}."""
    if not DIMENSIONS_FILE.exists():
        raise SystemExit("Pirmiausia paleiskite kmeans/kmeans_dimensions.py")
    table = pd.read_csv(DIMENSIONS_FILE)
    best = table[table["rekomenduojama"]]
    return dict(zip(best["mazinimas"], best["erdve"]))


def final_spaces():
    """Pradinė erdvė ir geriausia PCA bei t-SNE dimensija."""
    best = load_best_dimensions()
    return [REFERENCE] + [best[reduction] for reduction in REDUCTIONS]


# --- k-vidurkių metodas -----------------------------------------------------

def fit_kmeans(X, k, init="k-means++", n_init=10, random_state=RANDOM_STATE):
    """k-vidurkiai. Grąžina (žymės, centrai, E_k-means)."""
    model = KMeans(n_clusters=k, init=init, n_init=n_init,
                   random_state=random_state).fit(X)
    return model.labels_, model.cluster_centers_, model.inertia_


def subsample_runs(space, k):
    """k-vidurkiai kiekvienoje 80 % imtyje (sėkla = imties numeris): {imtis: (ids, X, žymės)}."""
    runs = {}
    for seed in seeds_for(space):
        ids, X = load_space_subsample(space, seed)
        runs[seed] = (ids, X, fit_kmeans(X, k, random_state=seed)[0])
    return runs


# --- k-medoidų metodas ------------------------------------------------------

def _distance_sums(points, metric, chunk=1000):
    """Kiekvieno taško atstumų iki visų kitų taškų suma (dalimis, taupant atmintį)."""
    sums = np.empty(len(points))
    for start in range(0, len(points), chunk):
        sums[start:start + chunk] = cdist(points[start:start + chunk], points,
                                          metric=metric).sum(axis=1)
    return sums


def _kmedoids_init(X, k, metric, rng):
    """k-medoids++: kitas medoidas renkamas su tikimybe, proporcinga atstumui
    iki artimiausio jau parinkto medoido (kaip k-means++)."""
    medoids = [rng.integers(len(X))]
    nearest = cdist(X, X[medoids], metric=metric).ravel()
    for _ in range(1, k):
        new = rng.choice(len(X), p=nearest / nearest.sum())
        medoids.append(new)
        nearest = np.minimum(nearest, cdist(X, X[[new]], metric=metric).ravel())
    return np.array(medoids)


def fit_kmedoids(X, k, metric="cityblock", n_init=10, max_iter=100,
                 random_state=RANDOM_STATE):
    rng = np.random.default_rng(random_state)
    best = None
    for _ in range(n_init):
        medoids = _kmedoids_init(X, k, metric, rng)
        for _ in range(max_iter):
            labels = cdist(X, X[medoids], metric=metric).argmin(axis=1)
            new = medoids.copy()
            for cluster in range(k):
                members = np.flatnonzero(labels == cluster)
                new[cluster] = members[_distance_sums(X[members], metric).argmin()]
            if np.array_equal(new, medoids):
                break
            medoids = new
        distances = cdist(X, X[medoids], metric=metric)
        labels = distances.argmin(axis=1)
        cost = distances.min(axis=1).sum()
        if best is None or cost < best[2]:
            best = (labels, medoids, cost)
    return best


def fit_variant(X, variant, k, n_init=10, random_state=RANDOM_STATE):
    method, metric = VARIANTS[variant]
    if method == "kmeans":
        return fit_kmeans(X, k, n_init=n_init, random_state=random_state)[0]
    return fit_kmedoids(X, k, metric=metric, n_init=n_init,
                        random_state=random_state)[0]


# --- Klasterių numerių suderinimas --------------------------------------------

def align_labels(labels, pairs, k_reference):
    """Suporuotiems klasteriams suteikiamas atskaitos klasterio numeris (ta pati spalva
    visuose grafikuose), nesuporuoti gauna numerius k_ref, k_ref + 1, ..."""
    mapping = dict(zip(pairs["klasteris"], pairs["atskaitos_klasteris"]))
    next_label = k_reference
    for label in np.unique(labels):
        if label not in mapping:
            mapping[label] = next_label
            next_label += 1
    return np.array([mapping[label] for label in labels])


# --- Grafikai ---------------------------------------------------------------

def background_map():
    """Visų erdvių klasteriai rodomi tame pačiame 2D t-SNE žemėlapyje."""
    ids, Y = load_space("tsne")
    return pd.DataFrame({"id": ids, "x": Y[:, 0], "y": Y[:, 1]}).set_index("id")


def cluster_colours(k):
    cmap = plt.get_cmap("tab10" if k <= 10 else "tab20")
    return [cmap(i % cmap.N) for i in range(k)]


def plot_clusters(ax, ids, labels, title, background=None):
    background = background if background is not None else background_map()
    points = background.loc[ids]
    names = np.unique(labels)
    colours = cluster_colours(max(names) + 1)
    for name in names:
        mask = labels == name
        ax.scatter(points["x"].to_numpy()[mask], points["y"].to_numpy()[mask],
                   s=3, color=colours[name], label=f"{name} ({mask.sum()})", rasterized=True)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(markerscale=4, fontsize=7, loc="best", frameon=False)


def save_figure(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Išsaugota: {path.relative_to(ROOT)}")