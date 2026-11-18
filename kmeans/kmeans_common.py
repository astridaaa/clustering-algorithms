"""Bendros k-vidurkių dalies funkcijos: modeliai, keliai ir grafikai.

Modeliai:
  * k-vidurkių metodas (Euklido atstumas, centras – klasterio vidurkis),
    sklearn KMeans realizacija;
  * k-medoidų metodas (centras – klasterio objektas, medoidas), kurį galima
    taikyti su bet kuriuo Minkovskio grupės atstumu: Euklido, miesto kvartalų
    (Manheteno) ir Čebyševo. Juo tiriama, kaip rezultatus keičia panašumo
    matas. k-vidurkių metodas kitų atstumų nepalaiko, nes vidurkis
    minimizuoja tik kvadratinį Euklido atstumą.

Skriptai leidžiami iš repozitorijos šaknies, pvz.:
    python kmeans/kmeans_k.py
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# Su visais branduoliais sklearn KMeans gijos viena kitai trukdo (~2 kartus
# lėčiau), todėl ribojama iki 4. Turi būti nustatyta prieš importuojant numpy.
os.environ.setdefault("OMP_NUM_THREADS", "4")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

from common import load_space
from evaluation import RESULTS_DIR

OUTPUT_DIR = RESULTS_DIR / "kmeans"
FIGURES_DIR = OUTPUT_DIR / "grafikai"

ALGORITHM = "k-means"
RANDOM_STATE = 0

# Pasirinktas klasterių skaičius kiekvienoje erdvėje (žr. kmeans_k.py rezultatus):
# pradinėje ir PCA erdvėse visi kriterijai rodo k = 3, t-SNE erdvėje alkūnės
# metodas ir stabilumas – k = 4 (t-SNE vidiniai matai dirbtinai dideli).
CHOSEN_K = {"originali": 3, "pca": 3, "tsne": 4}

SPACE_NAMES = {"originali": "Pradinė erdvė", "pca": "PCA erdvė", "tsne": "t-SNE erdvė"}

# Variantai: (metodas, atstumas). k-vidurkių metodas – tik Euklido atstumas.
VARIANTS = {
    "k-vidurkiai, Euklido": ("kmeans", "euclidean"),
    "k-medoidai, Euklido": ("kmedoids", "euclidean"),
    "k-medoidai, Manheteno": ("kmedoids", "cityblock"),
    "k-medoidai, Čebyševo": ("kmedoids", "chebyshev"),
}

# scipy (cdist) ir sklearn (silhouette_score) tą patį atstumą vadina skirtingai.
SKLEARN_METRIC = {"euclidean": "euclidean", "cityblock": "manhattan",
                  "chebyshev": "chebyshev"}


# --- k-vidurkių metodas -----------------------------------------------------

def fit_kmeans(X, k, init="k-means++", n_init=10, random_state=RANDOM_STATE):
    """k-vidurkiai. Grąžina (žymės, centrai, E_k-means)."""
    model = KMeans(n_clusters=k, init=init, n_init=n_init,
                   random_state=random_state).fit(X)
    return model.labels_, model.cluster_centers_, model.inertia_


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
    """k-medoidai (kintamasis algoritmas, angl. alternate).

    1. Parenkami k medoidų (k-medoids++).
    2. Kiekvienas objektas priskiriamas artimiausiam medoidui.
    3. Kiekvieno klasterio medoidu tampa objektas, kurio atstumų iki kitų
       to klasterio objektų suma mažiausia.
    4. 2–3 žingsniai kartojami, kol medoidai nebesikeičia.
    Kaip ir k-vidurkiams, algoritmas vykdomas n_init kartų iš skirtingų
    pradžių ir paliekamas sprendinys su mažiausia atstumų iki medoidų suma.
    Grąžina (žymės, medoidų indeksai, atstumų suma).
    """
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
    """Paleidžia vieną iš VARIANTS. Grąžina žymes."""
    method, metric = VARIANTS[variant]
    if method == "kmeans":
        return fit_kmeans(X, k, n_init=n_init, random_state=random_state)[0]
    return fit_kmedoids(X, k, metric=metric, n_init=n_init,
                        random_state=random_state)[0]


# --- Grafikai ---------------------------------------------------------------

def background_map():
    """Bendras 2D t-SNE žemėlapis (2 darbas), ant kurio rodomi visų erdvių
    klasteriai, kad grafikai būtų palyginami. Grąžina DataFrame su id."""
    ids, Y = load_space("tsne")
    return pd.DataFrame({"id": ids, "x": Y[:, 0], "y": Y[:, 1]}).set_index("id")


def cluster_colours(k):
    cmap = plt.get_cmap("tab10" if k <= 10 else "tab20")
    return [cmap(i % cmap.N) for i in range(k)]


def plot_clusters(ax, ids, labels, title, background=None):
    """Klasteriai ant bendro t-SNE žemėlapio."""
    background = background if background is not None else background_map()
    points = background.loc[ids]
    names = np.unique(labels)
    colours = cluster_colours(len(names))
    for colour, name in zip(colours, names):
        mask = labels == name
        ax.scatter(points["x"].to_numpy()[mask], points["y"].to_numpy()[mask],
                   s=3, color=colour, label=f"{name} ({mask.sum()})", rasterized=True)
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