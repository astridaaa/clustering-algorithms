"""Hierarchiniam metodui specifiniai matai.

Bendri matai (porų Jaccard per pair_confusion_matrix, ARI, NMI, stabilumas,
sutampantys taškai, kryžminės lentelės) imami iš bendro evaluation.py, kad
visi metodai skaičiuotų juos vienodai. Čia – tik tai, ko ten nėra.
"""
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)

from hier_common import K_RANGE


def distance_matrix(X, metric):
    """Kvadratinė atstumų matrica – siluetui skaičiuoti daug kartų (k = 2…10)."""
    return squareform(pdist(X, metric=metric))


def internal_measures(X, labels, D=None, metric="euclidean"):
    """Siluetas – su tuo pačiu atstumu, kuriuo klasterizuota (D arba metric).
    CH ir DB apibrėžti per centroidus, todėl visada Euklido erdvėje."""
    if len(np.unique(labels)) < 2:
        return {"siluetas": np.nan, "CH": np.nan, "DB": np.nan}
    silhouette = (silhouette_score(D, labels, metric="precomputed") if D is not None
                  else silhouette_score(X, labels, metric=metric))
    return {"siluetas": silhouette,
            "CH": calinski_harabasz_score(X, labels),
            "DB": davies_bouldin_score(X, labels)}


def merge_gaps(Z, k_range=K_RANGE):
    """Tarpas tarp gretimų sujungimų aukščių kiekvienam k.

    Pjaunant į k klasterių, nepritaikomi paskutiniai k - 1 sujungimai;
    tarpas_k = h[n-k] - h[n-k-1] (h – Z[:, 2], 0 indeksavimas).
    Centroid metodui tarpas gali būti neigiamas (inversija).
    """
    n = len(Z) + 1
    heights = Z[:, 2]
    return {k: heights[n - k] - heights[n - k - 1] for k in k_range}


def k_by_gap(Z, k_range=K_RANGE):
    gaps = merge_gaps(Z, k_range)
    return max(gaps, key=gaps.get), gaps


def size_summary(labels):
    """Klasterių dydžiai (mažėjančiai), didžiausio dalis ir mažų klasterių skaičius."""
    sizes = np.sort(np.bincount(labels))[::-1]
    return {
        "dydziai": " / ".join(str(s) for s in sizes),
        "didziausio_dalis": sizes[0] / sizes.sum(),
        "mazu_klasteriu": int((sizes < 0.01 * sizes.sum()).sum()),
        "pavieniu_tasku": int((sizes == 1).sum()),
    }


TOLERANCE = 0.01


def recommend_dimension(table, reductions=("pca", "tsne")):
    """Geriausia dimensija kiekvienam mažinimo metodui (stulpelis „rekomenduojama“).

    Kandidatai – dimensijos, kurių siluetas pradinėje erdvėje (siluetas_orig)
    nuo geriausio skiriasi mažiau nei TOLERANCE. Iš jų imama stabiliausia
    (stab_ARI_vid), o kol stabilumas nesuskaičiuotas – mažiausia dimensija.
    """
    table = table.copy()
    table["rekomenduojama"] = False
    for reduction in reductions:
        group = table[table["mazinimas"] == reduction]
        close = group[group["siluetas_orig"] >= group["siluetas_orig"].max() - TOLERANCE]
        if "stab_ARI_vid" in close and close["stab_ARI_vid"].notna().all():
            best = close["stab_ARI_vid"].idxmax()
        else:
            best = close["dimensija"].idxmin()
        table.loc[best, "rekomenduojama"] = True
    return table
