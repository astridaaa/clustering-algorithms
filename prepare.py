import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from common import (META_FILE, SUBSAMPLES_FILE, coordinate_names,
                    draw_subsample, reduce_dimension, robust_scale,
                    space_file, stability_file)
from config import (CLASS_COLUMN, CLASSES, CORRELATION_THRESHOLD,
                    EXPECTED_OBJECTS, FEATURE_SETS, FEATURES, INPUT_FILE,
                    MAIN_SET, META_COLUMNS, N_SUBSAMPLES, OUTLIER_COLUMN,
                    PREPARED_DIR, RR_ERROR_COLUMN, RR_VID_ERROR_MS, SPACES,
                    STABILITY_DIR, TSNE_RANDOM_STATE)


def load_input():
    data = pd.read_csv(INPUT_FILE)
    assert len(data) == EXPECTED_OBJECTS, f"Tikėtasi {EXPECTED_OBJECTS} objektų, gauta {len(data)}"
    assert data["id"].is_unique, "id turi būti unikalūs"
    assert data[FEATURES].notna().all().all(), "Neturi būti trūkstamų reikšmių"

    print(f"Įkelta: {INPUT_FILE.name}, {len(data)} dūžiai, "
          f"{data[OUTLIER_COLUMN].sum()} pažymėtos ekstremalios išskirtys")
    print(data[CLASS_COLUMN].value_counts().reindex(CLASSES).to_string())
    return data


def mark_rr_errors(data):
    rr_vid = data["RR_pre"] / data["RR_sant_vid"]
    data[RR_ERROR_COLUMN] = rr_vid > RR_VID_ERROR_MS

    errors = data[data[RR_ERROR_COLUMN]]
    print(f"\nRR_vid skaičiavimo klaidos (RR_vid > {RR_VID_ERROR_MS} ms): {len(errors)}")
    print(errors[["id", "irasas", CLASS_COLUMN, "RR_pre", "RR_sant_vid"]]
          .assign(RR_vid=rr_vid[errors.index].round(0)).to_string(index=False))
    assert errors[OUTLIER_COLUMN].all(), "Klaidos turi būti tarp ekstremalių išskirčių"
    return data


def check_correlation(data):
    correlation = data[FEATURES].corr(method="spearman")
    correlation.round(3).to_csv(PREPARED_DIR / "koreliacija.csv", encoding="utf-8-sig")

    strong = [(first, second, correlation.loc[first, second])
              for i, first in enumerate(FEATURES) for second in FEATURES[i + 1:]
              if abs(correlation.loc[first, second]) > CORRELATION_THRESHOLD]
    strongest = correlation.where(~np.eye(len(FEATURES), dtype=bool)).abs().stack().idxmax()

    print(f"\nSpearman koreliacija, riba |rho| > {CORRELATION_THRESHOLD}:")
    print(correlation.round(3).to_string())
    print(f"Stipriausia pora: {strongest[0]} – {strongest[1]}, "
          f"rho = {correlation.loc[strongest]:.3f}")
    if strong:
        raise SystemExit(f"Rasta stipriai koreliuotų porų: {strong}. "
                         "Aptarkite su komanda prieš tęsdami.")
    print("Stipriai koreliuotų porų nėra – visi požymiai paliekami.")


def save_space(ids, coordinates, path, columns):
    table = pd.DataFrame(coordinates, columns=columns)
    table.insert(0, "id", ids)
    table.to_csv(path, index=False)


def prepare_full(data):
    ids = data["id"].to_numpy()
    parameters = []

    print("\nNormavimas (x - mediana) / IQR:")
    for feature_set, features in FEATURE_SETS.items():
        X_scaled, median, iqr = robust_scale(data[features].to_numpy())
        assert np.allclose(np.median(X_scaled, axis=0), 0)
        save_space(ids, X_scaled, space_file("originali", feature_set), features)
        parameters += [{"rinkinys": feature_set, "pozymis": f, "mediana": m, "IQR": q}
                       for f, m, q in zip(features, median, iqr)]
        print(f"  {feature_set}: {len(features)} požymiai -> {space_file('originali', feature_set).name}")

    pd.DataFrame(parameters).round(4).to_csv(PREPARED_DIR / "normavimo_parametrai.csv",
                                             index=False, encoding="utf-8-sig")

    X_main = pd.read_csv(space_file("originali", MAIN_SET)).drop(columns="id").to_numpy()

    pca = PCA().fit(X_main)
    variance = pd.DataFrame({
        "komponente": [f"PC{i + 1}" for i in range(len(FEATURES))],
        "dispersijos_dalis": pca.explained_variance_ratio_,
        "sukaupta": np.cumsum(pca.explained_variance_ratio_),
    })
    variance.round(4).to_csv(PREPARED_DIR / "pca_dispersija.csv", index=False)
    print(f"\nPCA: 2 komponentės išlaiko {variance['sukaupta'][1] * 100:.1f} % dispersijos")

    for space in ["pca", "tsne"]:
        start = time.time()
        Y = reduce_dimension(X_main, space, TSNE_RANDOM_STATE)
        save_space(ids, Y, space_file(space, MAIN_SET), coordinate_names(space))
        print(f"  {space}: {space_file(space, MAIN_SET).name} ({time.time() - start:.0f} s)")


def prepare_subsamples(data):
    all_ids = data["id"].to_numpy()
    features = FEATURE_SETS[MAIN_SET]
    by_id = data.set_index("id")[features]

    max_seeds = max(N_SUBSAMPLES.values())
    samples = {seed: draw_subsample(all_ids, seed) for seed in range(max_seeds)}
    pd.concat([pd.DataFrame({"imtis": seed, "id": ids}) for seed, ids in samples.items()]
              ).to_csv(SUBSAMPLES_FILE, index=False)

    sizes = {len(ids) for ids in samples.values()}
    shared = [len(np.intersect1d(samples[a], samples[b]))
              for a in samples for b in samples if a < b]
    print(f"\nStabilumo imtys: {max_seeds} imčių po {sizes.pop()} objektų, "
          f"bendrų objektų porose vidutiniškai {np.mean(shared) / len(all_ids) * 100:.1f} %")

    for space in SPACES:
        start = time.time()
        rows = []
        for seed in range(N_SUBSAMPLES[space]):
            ids = samples[seed]
            X_scaled, _, _ = robust_scale(by_id.loc[ids].to_numpy())
            Y = reduce_dimension(X_scaled, space, random_state=seed)
            table = pd.DataFrame(Y, columns=coordinate_names(space))
            table.insert(0, "id", ids)
            table.insert(0, "imtis", seed)
            rows.append(table)
        pd.concat(rows).to_csv(stability_file(space, MAIN_SET), index=False)
        print(f"  {space}: {N_SUBSAMPLES[space]} imčių -> "
              f"stabilumas/{stability_file(space, MAIN_SET).name} ({time.time() - start:.0f} s)")


def main():
    STABILITY_DIR.mkdir(parents=True, exist_ok=True)

    data = mark_rr_errors(load_input())
    check_correlation(data)
    data[META_COLUMNS].to_csv(META_FILE, index=False)

    prepare_full(data)
    prepare_subsamples(data)
    print(f"\nParuošta: {PREPARED_DIR}")


if __name__ == "__main__":
    main()
