import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from config import (FEATURE_SETS, MAIN_SET, N_COMPONENTS, N_SUBSAMPLES,
                    PREPARED_DIR, STABILITY_DIR, SUBSAMPLE_FRACTION,
                    TSNE_PARAMS)


META_FILE = PREPARED_DIR / "meta.csv"
SUBSAMPLES_FILE = PREPARED_DIR / "imtys.csv"


def space_file(space, feature_set=MAIN_SET):
    return PREPARED_DIR / f"{space}_{feature_set}.csv"


def stability_file(space, feature_set=MAIN_SET):
    return STABILITY_DIR / f"{space}_{feature_set}.csv"


def load_meta():
    """Žymės pagal id: klasė, ritmas, įrašas, išskirčių požymiai."""
    return pd.read_csv(META_FILE)


def load_space(space, feature_set=MAIN_SET):
    """Visa aibė pasirinktoje erdvėje: (ids, X)."""
    data = pd.read_csv(space_file(space, feature_set))
    return data["id"].to_numpy(), data.drop(columns="id").to_numpy()


def subsample_seeds(space):
    """Stabilumo imčių sėklos šiai erdvei: 0, ..., N-1."""
    return range(N_SUBSAMPLES[space])


def load_subsample(space, seed, feature_set=MAIN_SET):
    """80 % imtis `seed`, normuota ir projektuota iš naujo: (ids, X)."""
    data = pd.read_csv(stability_file(space, feature_set))
    sample = data[data["imtis"] == seed]
    return sample["id"].to_numpy(), sample.drop(columns=["imtis", "id"]).to_numpy()


def robust_scale(X):
    """x' = (x - mediana) / IQR, parametrai skaičiuojami iš paties X."""
    median = np.median(X, axis=0)
    iqr = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)
    return (X - median) / iqr, median, iqr


def reduce_dimension(X_scaled, space, random_state):
    """Projekcija į N_COMPONENTS dimensijų. 'originali' grąžina X be pakeitimų."""
    if space == "originali":
        return X_scaled
    if space == "pca":
        return PCA(n_components=N_COMPONENTS).fit_transform(X_scaled)
    if space == "tsne":
        model = TSNE(n_components=N_COMPONENTS, random_state=random_state,
                     **TSNE_PARAMS)
        return model.fit_transform(X_scaled)
    raise ValueError(f"Nežinoma erdvė: {space}")


def coordinate_names(space, feature_set=MAIN_SET):
    if space == "originali":
        return FEATURE_SETS[feature_set]
    prefix = {"pca": "PC", "tsne": "tSNE"}[space]
    return [f"{prefix}{i + 1}" for i in range(N_COMPONENTS)]


def draw_subsample(all_ids, seed):
    """80 % objektų be grąžinimo; id surikiuoti didėjimo tvarka."""
    rng = np.random.default_rng(seed)
    size = round(SUBSAMPLE_FRACTION * len(all_ids))
    return np.sort(rng.choice(all_ids, size=size, replace=False))


def common_labels(ids_a, labels_a, ids_b, labels_b):
    """Dviejų imčių klasterių žymės tik bendriems objektams.

    Objektai derinami pagal id, ne pagal eilės numerį imtyje,
    todėl grąžinami vektoriai yra tos pačios tvarkos ir ilgio.
    """
    a = pd.Series(labels_a, index=ids_a)
    b = pd.Series(labels_b, index=ids_b)
    shared = a.index.intersection(b.index)
    return a.loc[shared].to_numpy(), b.loc[shared].to_numpy()
