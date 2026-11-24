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
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

import evaluation as ev
from common import (common_labels, load_projection as load, load_space, load_subsample,
                    subsample_seeds)
from config import CLASSES, DIMENSION_SPACE_NAMES as SPACE_NAMES, DIMENSION_SPACES as SPACES
from evaluation import NOISE, RESULTS_DIR

OUTPUT_DIR = RESULTS_DIR / "dbscan"
FIGURES_DIR = OUTPUT_DIR / "grafikai"
CHOSEN_FILE = OUTPUT_DIR / "pasirinkti_parametrai.csv"

ALGORITHM = "DBSCAN"

MIN_SAMPLES = 10

N_CLASSES = len(CLASSES)
TARGET_K = {"k5": N_CLASSES}
VARIANTS = {"kelio": "eps kelio metodu",
            **{name: f"k = {k}" + (" (tiek, kiek klasių)" if k == N_CLASSES else "")
               for name, k in TARGET_K.items()}}
MAX_NOISE = 0.5

# Galutinių suskirstymų stabilumas: erdvė -> paruoštų 80 % imčių rinkinys
# (data/paruosta/stabilumas/: pca – 3 komponentės, tsne – 2D, kaip pasirinktos DBSCAN erdvės).
STABILITY_SPACES = {"originali": "originali", "pca3": "pca", "tsne2": "tsne"}
# Parametrų tinklelio (dbscan_params.py) stabilumas – tik pradinėje erdvėje.
STABILITY_SPACE = "originali"

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OTHER_COLOUR = "#c3c2b7"
NOISE_COLOUR = "#52514e"
TYPE_COLOURS = {"branduolinis": PALETTE[0], "ribinis": PALETTE[1], "triukšmas": NOISE_COLOUR}
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def k_distances(X, min_samples=MIN_SAMPLES):
    """Kiekvieno taško atstumas iki min_samples-o artimiausio taško (įskaitant jį patį),
    surikiuotas didėjančiai – k-atstumų grafiko reikšmės.

    kneighbors(X) pirmuoju kaimynu grąžina patį tašką (atstumas 0), todėl paskutinis
    stulpelis – mažiausias eps, su kuriuo taškas tampa branduoliniu: sklearn
    min_samples skaičiuoja ir patį tašką.
    """
    distances, _ = NearestNeighbors(n_neighbors=min_samples).fit(X).kneighbors(X)
    return np.sort(distances[:, -1])


def knee_index(values):
    """Kelio taškas didėjančioje kreivėje (Kneedle idėja): abi ašys normuojamos į [0, 1],
    kelias – taškas, labiausiai nutolęs žemyn nuo stygos tarp galinių taškų."""
    values = np.asarray(values, dtype=float)
    x = np.linspace(0, 1, len(values))
    y = (values - values.min()) / (values.max() - values.min())
    return int(np.argmax(x - y))


def knee_eps(X, min_samples=MIN_SAMPLES):
    """eps kelio metodu iš k-atstumų grafiko, suapvalintas iki 0,001."""
    distances = k_distances(X, min_samples)
    return round(float(distances[knee_index(distances)]), 3)


def eps_scan(X, min_samples=MIN_SAMPLES, start=0.2, stop=4.0, step=0.01):
    """DBSCAN su eps = kelio eps × 0,20; 0,21; …: klasterių skaičius ir triukšmo dalis.
    Sustojama, kai eps jau didesnis nei 1,5 kelio eps, o klasteris liko vienas."""
    base = knee_eps(X, min_samples)
    rows = []
    for factor in np.arange(start, stop + step / 2, step):
        eps = round(base * factor, 3)
        labels, _ = fit_dbscan(X, eps, min_samples)
        k = ev.n_clusters(labels)
        rows.append({"eps": eps, "eps_daugiklis": round(factor, 2), "k": k,
                     "triuksmo_dalis": ev.noise_share(labels)})
        if k <= 1 and factor > 1.5:
            break
    return pd.DataFrame(rows).drop_duplicates("eps", ignore_index=True)


def eps_for_k(scan, k=N_CLASSES, max_noise=MAX_NOISE):
    """eps, su kuriuo klasterių k: iš variantų, kur triukšmo ≤ 50 %, imamas mažiausio
    triukšmo. Jei lygiai k klasterių negaunama – artimiausias klasterių skaičius."""
    usable = scan[scan["triuksmo_dalis"] <= max_noise]
    distance = (usable["k"] - k).abs()
    best = usable[distance == distance.min()].sort_values(["triuksmo_dalis", "eps"]).iloc[0]
    return float(best["eps"]), int(best["k"])


def load_chosen():
    """dbscan_eps.py parinkti parametrai: {(erdvė, variantas): eps}."""
    table = pd.read_csv(CHOSEN_FILE)
    return {(row.erdve, row.variantas): float(row.eps) for row in table.itertuples()}


def fit_dbscan(X, eps, min_samples=MIN_SAMPLES):
    """DBSCAN su Euklido atstumu. Grąžina (žymės, branduolinių taškų kaukė); triukšmas = -1."""
    model = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    core = np.zeros(len(X), dtype=bool)
    core[model.core_sample_indices_] = True
    return model.labels_, core


def point_types(labels, core):
    """Kiekvieno taško tipas: branduolinis, ribinis arba triukšmas."""
    return np.where(core, "branduolinis",
                    np.where(np.asarray(labels) == NOISE, "triukšmas", "ribinis"))


def params(eps, min_samples=MIN_SAMPLES):
    return {"eps": eps, "min_samples": min_samples, "atstumas": "euclidean"}


def subsample_runs(space, eps, min_samples=MIN_SAMPLES):
    """DBSCAN kiekvienoje 80 % imtyje su tais pačiais parametrais: {imtis: (ids, X, žymės)}.
    DBSCAN deterministinis, todėl random_state nereikalingas."""
    runs = {}
    for seed in subsample_seeds(space):
        ids, X = load_subsample(space, seed)
        runs[seed] = (ids, X, fit_dbscan(X, eps, min_samples)[0])
    return runs


def stability_pairs(runs):
    """Tik imčių porų ARI, NMI ir Jaccard (be vidinių matų – greičiau nei ev.stability)."""
    seeds = sorted(runs)
    rows = []
    for i, seed_a in enumerate(seeds):
        for seed_b in seeds[i + 1:]:
            ids_a, _, labels_a = runs[seed_a]
            ids_b, _, labels_b = runs[seed_b]
            u, v = common_labels(ids_a, labels_a, ids_b, labels_b)
            rows.append({"imtis_a": seed_a, "imtis_b": seed_b, "bendru": len(u),
                         **ev.external_metrics(u, v)})
    return pd.DataFrame(rows)


def cluster_sizes(labels):
    """Klasterių dydžiai mažėjimo tvarka (be triukšmo)."""
    labels = np.asarray(labels)
    return sorted(np.bincount(labels[labels != NOISE]).tolist(), reverse=True)


def largest_share(labels):
    sizes = cluster_sizes(labels)
    return sizes[0] / len(labels) if sizes else 0.0


def background_map():
    """Visi suskirstymai vaizduojami tame pačiame t-SNE žemėlapyje (kaip k-vidurkių)."""
    ids, Y = load_space("tsne")
    return pd.DataFrame({"id": ids, "x": Y[:, 0], "y": Y[:, 1]}).set_index("id")


def align_labels(labels, pairs):
    """Suporuoti klasteriai gauna atskaitos klasterio numerį (ta pati spalva visuose
    grafikuose), nesuporuoti – kitus numerius dydžio mažėjimo tvarka; triukšmas lieka -1."""
    labels = np.asarray(labels)
    mapping = {NOISE: NOISE}
    mapping.update(dict(zip(pairs["klasteris"], pairs["atskaitos_klasteris"])))
    next_label = max([value for value in mapping.values()] + [-1]) + 1
    unmatched = [name for name in np.unique(labels) if name not in mapping]
    for name in sorted(unmatched, key=lambda name: -(labels == name).sum()):
        mapping[name] = next_label
        next_label += 1
    return np.array([mapping[name] for name in labels])


def plot_clusters(ax, ids, labels, title, background=None):
    """Klasteriai fiksuotos paletės spalvomis (pagal numerį), kai jų > 8 – 7 pirmi
    atskirai, likę kaip „kiti klasteriai“; triukšmas – tamsiai pilkas, piešiamas paskutinis."""
    background = background if background is not None else background_map()
    points = background.loc[ids]
    x, y = points["x"].to_numpy(), points["y"].to_numpy()
    labels = np.asarray(labels)

    names = [name for name in np.unique(labels) if name != NOISE]
    limit = len(PALETTE) - 1 if len(names) > len(PALETTE) else len(PALETTE)
    for name in (name for name in names if name < limit):
        mask = labels == name
        ax.scatter(x[mask], y[mask], s=3, color=PALETTE[name],
                   label=f"{name} ({mask.sum()})", rasterized=True)
    other = (labels != NOISE) & (labels >= limit)
    if other.any():
        n_other = len(np.unique(labels[other]))
        ax.scatter(x[other], y[other], s=3, color=OTHER_COLOUR, rasterized=True,
                   label=f"kiti klasteriai, {n_other} vnt. ({other.sum()})")
    noise = labels == NOISE
    ax.scatter(x[noise], y[noise], s=4, color=NOISE_COLOUR,
               label=f"triukšmas ({noise.sum()})", rasterized=True)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(markerscale=4, fontsize=7, loc="best", frameon=False)


def plot_point_types(ax, ids, types, title, background=None):
    background = background if background is not None else background_map()
    points = background.loc[ids]
    for name, colour in TYPE_COLOURS.items():
        mask = types == name
        ax.scatter(points["x"].to_numpy()[mask], points["y"].to_numpy()[mask], s=3,
                   color=colour, label=f"{name} ({mask.sum()})", rasterized=True)
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
