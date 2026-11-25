"""Check: does the choice of success label change the 2025 pre-draft forecast?

Compares three over/under labels with the forecast the main script keeps (algorithm and inputs chosen on the study
label), each split at the 2023-24 median per position:
  Play + starts  the study's label (half first-season play score, half career starts per season)
  Starts only    career starts per season; a player with 0 starts can't be an over-performer, because many players
                 tie at 0 and would otherwise all land on one side of the median
  Play only      first-season play score

Usage
  python src/compare_labels.py                      # downloads with kagglehub
  python src/compare_labels.py --data-dir <folder>  # local copy of the competition CSVs
Writes results/label_comparison.csv.
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import combine_rookie_classifier as c  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Compare success labels for the 2025 forecast")
    ap.add_argument("--data-dir", help="folder with the competition CSVs (skips the kagglehub download)")
    ap.add_argument("--out", default="results", help="output folder")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df, games, plays, career = c.load(c.get_data_dir(args.data_dir))
    df = c.add_combine_traits(df)
    df = df.join(c.build_rookie_success(df, games, plays), on="nfl_id")
    df = c.add_labels(df, career)

    z = lambda s: (s - s.mean()) / s.std()
    grp = df.groupby(["combine_position", "draft_year"])
    labels = {"Play + starts": df.success,
              "Starts only": grp.starts_per_season.transform(z),
              "Play only": grp.rookie_score.transform(z)}
    algo, inputs, _ = c.choose(df)  # the forecast kept by the main script (chosen on the study label)
    print(f"Forecast: {algo}, inputs = {inputs}")
    rows = []
    for name, score in labels.items():
        d = df.assign(success=score)
        cut = d[d.draft_year <= 2024].groupby("combine_position").success.median()
        d["over"] = (d.success > d.combine_position.map(cut)).astype(int)
        if name == "Starts only":
            d["over"] = d.over * (d.starts_per_season > 0)
        clf, _ = c.classifier(d, algo, inputs)
        rows.append(clf.assign(label=name))
    out = pd.concat(rows)[["label", "position", "n_test", "n_test_over", "baseline", "accuracy", "acc_lo", "acc_hi",
                           "auc", "precision", "recall", "f1"]]
    out.to_csv(f"{args.out}/label_comparison.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.round(3).to_string(index=False))
    agree = df.groupby("combine_position").apply(
        lambda g: g.rookie_score.corr(g.starts_per_season, method="spearman"), include_groups=False)
    print("\nHow closely the two halves agree (Spearman, first-season play vs starts per season):")
    print(agree.round(2).to_string())


if __name__ == "__main__":
    main()
