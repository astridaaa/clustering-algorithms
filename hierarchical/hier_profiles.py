"""Pasirinktų klasterių aprašymas ir pradinės bei sumažintų erdvių palyginimas.

* Klasterių dydžiai ir požymių medianos pradiniais vienetais (data/A.csv).
* Kryžminės lentelės su klase, ritmu, įrašu (dažniausi įrašai) ir ekstremalių
  išskirčių dalis – žymės naudojamos TIK interpretacijai.
* Sutampantys taškai tarp pradinės ir PCA / t-SNE erdvės klasterių
  (kryžminė lentelė, Vengrų metodu suporuoti klasteriai), maži klasteriai.

Profiliuojamos pradinė erdvė ir geriausios PCA / t-SNE dimensijos
(hier_common.FINAL_SPACES, BEST_DIMENSION).

PCA ir t-SNE klasteriai pernumeruojami pagal suporuotą pradinės erdvės
klasterį, kad ta pati spalva / numeris reikštų tą pačią grupę.
"""
from hier_common import (CHOSEN, FINAL_SPACES, LABELS_FILE, PROFILES_DIR, build_tree, load,
                         config_title, cut_tree_k)

import numpy as np
import pandas as pd

import evaluation as ev
from common import load_meta
from config import FEATURES, INPUT_FILE

TOP_RECORDS = 3


def chosen_labels(space):
    config = CHOSEN[space]
    ids, X = load(space)
    Z = build_tree(X, config["jungimas"], config["atstumas"])
    return ids, cut_tree_k(Z, config["k"])


def align_labels(labels, pairs, k_reference):
    """Suporuotas klasteris gauna atskaitos numerį; nesuporuoti – k_ref, k_ref + 1, ..."""
    mapping = dict(zip(pairs["klasteris"], pairs["atskaitos_klasteris"]))
    next_label = k_reference
    for label in np.unique(labels):
        if label not in mapping:
            mapping[label] = next_label
            next_label += 1
    return np.array([mapping[label] for label in labels])


def profile(space, ids, labels, raw, meta):
    """Dydis, išskirčių dalis, dominuojanti klasė ir medianos pradiniais vienetais."""
    frame = raw.loc[ids, FEATURES].assign(klasteris=labels)
    medians = frame.groupby("klasteris")[FEATURES].median()
    outliers = ev.outlier_share(ids, labels)
    classes = pd.crosstab(labels, meta.loc[ids, "klase"].to_numpy(), normalize="index")
    table = pd.DataFrame({
        "erdve": space,
        "dydis": outliers["objektu"],
        "dalis": outliers["objektu"] / len(labels),
        "isskirciu": outliers["isskirciu"],
        "isskirciu_dalis": outliers["dalis"],
        "dominuojanti_klase": classes.idxmax(axis=1),
        "dominuojancios_dalis": classes.max(axis=1),
    }).join(medians)
    overall = pd.DataFrame({"erdve": space, "dydis": len(labels), "dalis": 1.0,
                            "isskirciu": int(meta.loc[ids, "ekstremali_isskirtis"].sum()),
                            "isskirciu_dalis": meta.loc[ids, "ekstremali_isskirtis"].mean(),
                            **raw.loc[ids, FEATURES].median().to_dict()},
                           index=pd.Index(["visi"], name="klasteris"))
    return pd.concat([table, overall]).rename_axis("klasteris").reset_index()


def crosstab_long(space, ids, labels, column):
    counts = ev.cluster_composition(ids, labels, column, normalize=False)
    shares = ev.cluster_composition(ids, labels, column, normalize=True)
    return (counts.stack().rename("objektu").to_frame()
            .join(shares.stack().rename("dalis_klasteryje"))
            .reset_index().rename(columns={column: "reiksme"})
            .assign(erdve=space, zyme=column)
            .query("objektu > 0")[["erdve", "zyme", "klasteris", "reiksme", "objektu",
                                   "dalis_klasteryje"]])


def top_records(space, ids, labels, meta):
    frame = pd.DataFrame({"klasteris": labels, "irasas": meta.loc[ids, "irasas"].to_numpy()})
    rows = []
    for cluster, group in frame.groupby("klasteris"):
        counts = group["irasas"].value_counts()
        for rank, (record, count) in enumerate(counts.head(TOP_RECORDS).items(), 1):
            rows.append({"erdve": space, "klasteris": cluster, "vieta": rank, "irasas": record,
                         "objektu": count, "dalis_klasteryje": count / len(group),
                         "skirtingu_irasu": counts.size})
    return pd.DataFrame(rows)


def describe(row):
    medians = ", ".join(f"{f} {row[f]:.3g}" for f in FEATURES)
    dominant = ("" if row["klasteris"] == "visi" else
                f", dominuoja {row['dominuojanti_klase']} ({row['dominuojancios_dalis']:.0%})")
    return (f"  {row['klasteris']!s:>4}: n = {int(row['dydis'])} ({row['dalis']:.1%}), "
            f"išskirčių {row['isskirciu_dalis']:.1%}{dominant}; medianos: {medians}")


def main():
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(INPUT_FILE).set_index("id")
    meta = load_meta().set_index("id")

    ref_ids, ref_labels = chosen_labels("originali")
    k_ref = CHOSEN["originali"]["k"]
    labels = {"originali": (ref_ids, ref_labels)}
    matched_rows, pair_tables, contingency = [], [], {}
    for space in FINAL_SPACES[1:]:
        ids, raw_labels = chosen_labels(space)
        matched, share, pairs, crosstab = ev.matched_points(ids, raw_labels, ref_ids, ref_labels)
        aligned = align_labels(raw_labels, pairs, k_ref)
        matched, share, pairs, crosstab = ev.matched_points(ids, aligned, ref_ids, ref_labels)
        labels[space] = (ids, aligned)
        matched_rows.append({"erdve": space, "sutampa_tasku": matched, "sutampa_dalis": share,
                             "nesutampa_tasku": len(ids) - matched})
        pair_tables.append(pairs.assign(erdve=space))
        contingency[space] = crosstab

    label_table = pd.DataFrame({space: pd.Series(l, index=i) for space, (i, l) in labels.items()})
    label_table.rename_axis("id").to_csv(LABELS_FILE)

    profiles, crosstabs, records, meta_fit = [], [], [], []
    for space, (ids, l) in labels.items():
        profiles.append(profile(space, ids, l, raw, meta))
        for column in ("klase", "ritmas"):
            crosstabs.append(crosstab_long(space, ids, l, column))
        records.append(top_records(space, ids, l, meta))
        meta_fit.append(ev.compare_with_meta(ids, l).assign(erdve=space))

    profiles = pd.concat(profiles, ignore_index=True)
    profiles.round(4).to_csv(PROFILES_DIR / "profiliai_medianos.csv", index=False)
    crosstabs = pd.concat(crosstabs, ignore_index=True)
    crosstabs.round(4).to_csv(PROFILES_DIR / "kryzmine_klase_ritmas.csv", index=False)
    for space, (ids, l) in labels.items():
        ev.cluster_composition(ids, l, "klase", normalize=False).to_csv(
            PROFILES_DIR / f"kryzmine_klase_{space}.csv")
    records = pd.concat(records, ignore_index=True)
    records.round(4).to_csv(PROFILES_DIR / "top_irasai.csv", index=False)
    pd.concat(meta_fit, ignore_index=True).round(4).to_csv(
        PROFILES_DIR / "atitikimas_zymems.csv", index=False)

    matched = pd.DataFrame(matched_rows)
    matched.round(4).to_csv(PROFILES_DIR / "sutampantys_taskai.csv", index=False)
    pd.concat(pair_tables, ignore_index=True).round(4).to_csv(
        PROFILES_DIR / "suporuoti_klasteriai.csv", index=False)
    for space, table in contingency.items():
        table.to_csv(PROFILES_DIR / f"sutapimas_originali_{space}.csv")

    medians = profiles[profiles["klasteris"] != "visi"].set_index(["erdve", "klasteris"])[FEATURES]
    changes = []
    for space in FINAL_SPACES[1:]:
        for cluster in medians.loc[space].index:
            if cluster in medians.loc["originali"].index:
                diff = medians.loc[(space, cluster)] - medians.loc[("originali", cluster)]
                changes.append({"erdve": space, "klasteris": cluster, **diff.to_dict()})
    pd.DataFrame(changes).round(4).to_csv(PROFILES_DIR / "medianu_pokytis.csv", index=False)

    moved = []
    for space in FINAL_SPACES[1:]:
        both = label_table[["originali", space]].dropna()
        changed = both.index[both["originali"] != both[space]]
        sub = meta.loc[changed]
        moved.append({"erdve": space, "pakeite": len(changed),
                      "is_ju_isskirciu": int(sub["ekstremali_isskirtis"].sum()),
                      **{f"klase_{c}": int((sub["klase"] == c).sum()) for c in sorted(meta["klase"].unique())},
                      **{f"is_{int(a)}_i_{int(b)}": int(((both.loc[changed, "originali"] == a)
                                                        & (both.loc[changed, space] == b)).sum())
                         for a in np.unique(both["originali"]) for b in np.unique(both[space])
                         if a != b}})
    moved = pd.DataFrame(moved)
    moved.to_csv(PROFILES_DIR / "pakeite_klasteri.csv", index=False)

    pd.set_option("display.width", 250)
    for space in FINAL_SPACES:
        c = CHOSEN[space]
        print(f"\n{space}: {config_title(c['jungimas'], c['atstumas'], c['k'])}")
        for _, row in profiles[profiles["erdve"] == space].iterrows():
            print(describe(row))
        print("  Dažniausi įrašai: " + "; ".join(
            f"{cl}: " + ", ".join(f"{int(r.irasas)} ({r.dalis_klasteryje:.0%})"
                                  for r in g.itertuples())
            for cl, g in records[records["erdve"] == space].groupby("klasteris")))
        print("  Klasės (objektų):")
        print(ev.cluster_composition(*labels[space], "klase", normalize=False)
              .to_string().replace("\n", "\n    ").join(["    ", ""]))
    print("\nSutampantys taškai su pradine erdve:")
    print(matched.round(3).to_string(index=False))
    for space, table in contingency.items():
        print(f"\nKryžminė lentelė originali × {space}:\n{table.to_string()}")
    print("\nPakeitę klasterį objektai:")
    print(moved.T.to_string())
    print(f"\nIšsaugota: {PROFILES_DIR.relative_to(ev.ROOT)}/, {LABELS_FILE.name}")


if __name__ == "__main__":
    main()
