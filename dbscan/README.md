# DBSCAN

Paleidžiama Python 3.12 aplinkoje su `requirements.txt` versijomis.
Duomenų (`data/paruosta`) iš naujo negeneruoti – DBSCAN skaičiuojamas ant repozitorijoje esančių.

Erdvės: pradinė (5D), PCA ir t-SNE 2D, 3D, 5D (`DIMENSION_SPACES` – `config.py`, `load_projection` – `common.py`).
t-SNE 3D/5D imami iš `rezultatai/erdves/`; jei failo nėra, apskaičiuojami (3D ~1 min, 5D ~1 val.).

| Eilė | Komanda | Trukmė | Ką daro |
|---|---|---|---|
| 1 | `python dbscan/dbscan_eps.py` | ~5–8 min | k-atstumų grafikai, eps kelio metodu, klasterių skaičius pagal eps, parinkti parametrai |
| 2 | `python dbscan/dbscan_params.py` | ~10 min | eps × min_samples tinklelis (k, triukšmas, vidiniai matai, ARI su klasėmis, stabilumas), k pasiūlymų lentelė |
| 3 | `python dbscan/dbscan_final.py` | ~2 min | galutiniai modeliai 7 erdvėse: vidiniai ir išoriniai matai, sutampantys taškai, triukšmas ir išskirtys, stabilumas (tik originalioje erdvėje), eksperimentų žurnalas |
| 4 | `python dbscan/dbscan_profiles.py` | <1 min | klasterių profiliai (medianos originaliais vienetais), klasių sudėtis, aprašomoji statistika |

Tvarka svarbi: 2–4 skriptai naudoja 1 skripto parinktus parametrus.

Rezultatai: `rezultatai/dbscan/` (lentelės, `grafikai/`, `galutiniai/`, `profiliai/`),
eksperimentų žurnalas `rezultatai/eksperimentai_dbscan.csv`.

## Variantai (`dbscan_common.py`)

- `kelio` – eps kelio metodu, min_samples = 2·D = 10;
- `k5` – eps, su kuriuo gaunama tiek klasterių, kiek klasių (5).

Kai grupė sutars galutinį k, įrašyk jį į `TARGET_K`, pvz. `TARGET_K = {"k5": N_CLASSES, "k3": 3}`,
ir paleisk visus 4 skriptus iš naujo.
