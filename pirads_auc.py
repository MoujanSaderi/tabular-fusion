"""
AUC of maxPIRADS used as a semi-continuous (ordinal 1-5) score for the
configured target (e.g. csPCa), on the same exams and labels the trained
models are scored on.

Unlike pirads_rule.py, which turns PI-RADS into a hard 0/1 decision at a single
cutoff, this ranks exams directly by maxPIRADS. The ROC curve therefore has one
point per PI-RADS cutoff (>=2, >=3, >=4, >=5), and the AUC is the probability
that a random positive exam has a higher maxPIRADS than a random negative one,
with ties counted as 1/2 (Mann-Whitney / trapezoidal AUC).

Cohort and labels come from train.build_dataset, exactly as in eval.py, so
training.mode: cspca keeps only biopsied exams (csPCa not NA) with
label = csPCa (ISUP GG >= 2). maxPIRADS is read from the split CSV; no H5
volumes or tabular CSV are needed.

Usage:
    python pirads_auc.py --config configs/pirads_rule.yaml --out preds_pirads_score.csv

    # a different split than paths.valid_csv (e.g. the held-out test set):
    python pirads_auc.py --config configs/pirads_rule.yaml \
        --val-csv /path/to/test_split.csv --out preds_pirads_score_test.csv

The 95% CI is a patient-level bootstrap (patients resampled with replacement,
all their exams kept together), since a patient can contribute several exams.

The output CSV follows eval.py's per-exam format. prob_1 is maxPIRADS rescaled
to [0, 1] ((maxPIRADS - 1) / 4); this is a monotone transform, so it has the
same AUC, and it lets the CSV be used anywhere eval.py's output is (e.g.
late_fusion.py --imaging-preds). It is a ranking score, not a calibrated
probability, so pred_label uses a PI-RADS cutoff (--cutoff, default >= 3)
instead of a probability threshold.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from train import SERIES_NAME_TO_ENUM, build_dataset

VALID_PIRADS = (1, 2, 3, 4, 5)


def bootstrap_auc_ci(df, n_boot, seed, alpha=0.05):
    """Patient-level bootstrap CI for the AUC of maxPIRADS vs target."""
    rng = np.random.default_rng(seed)
    groups = {pid: g.index.to_numpy() for pid, g in df.groupby("PatientID")}
    patients = np.array(list(groups))
    y = df["target"].to_numpy()
    s = df["maxPIRADS"].to_numpy()

    aucs = []
    for _ in range(n_boot):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        idx = np.concatenate([groups[p] for p in sampled])
        if np.unique(y[idx]).size < 2:
            continue  # AUC undefined for a single-class resample
        aucs.append(roc_auc_score(y[idx], s[idx]))
    aucs = np.asarray(aucs)
    lo, hi = np.quantile(aucs, [alpha / 2, 1 - alpha / 2])
    return lo, hi, len(aucs)


def per_score_table(df):
    """Exams, positives and positive rate at each maxPIRADS value."""
    t = df.groupby("maxPIRADS")["target"].agg(n="size", positives="sum")
    t["pos_rate"] = t["positives"] / t["n"]
    return t


def operating_points(df):
    """Sensitivity / specificity / PPV / NPV of the rule maxPIRADS >= k, per k."""
    y = df["target"].to_numpy()
    s = df["maxPIRADS"].to_numpy()
    rows = []
    for k in VALID_PIRADS[1:]:
        pred = s >= k
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        fn = int(np.sum(~pred & (y == 1)))
        tn = int(np.sum(~pred & (y == 0)))
        rows.append({
            "rule": f">= {k}",
            "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
            "specificity": tn / (tn + fp) if tn + fp else np.nan,
            "ppv": tp / (tp + fp) if tp + fp else np.nan,
            "npv": tn / (tn + fn) if tn + fn else np.nan,
        })
    return pd.DataFrame(rows).set_index("rule")


def main():
    parser = argparse.ArgumentParser(
        description="AUC of maxPIRADS as a semi-continuous score on a CSV split."
    )
    parser.add_argument("--config", required=True,
                        help="YAML config defining splits, training.mode and data "
                             "settings (e.g. configs/pirads_rule.yaml)")
    parser.add_argument("--val-csv", help="Split CSV to evaluate on (default: paths.valid_csv)")
    parser.add_argument("--out", required=True, help="Where to write the per-exam CSV")
    parser.add_argument("--cutoff", type=int, default=3,
                        help="PI-RADS cutoff for the pred_label column: 1 if "
                             "maxPIRADS >= cutoff (default 3)")
    parser.add_argument("--n-boot", type=int, default=2000,
                        help="Patient-level bootstrap resamples for the 95%% CI "
                             "(default 2000; 0 to skip)")
    parser.add_argument("--seed", type=int, help="Bootstrap seed (default: training.seed)")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    # Same defaults train.py / eval.py fill in.
    config.setdefault("paths", {}).setdefault("extra_valid_csv", None)
    config["paths"].setdefault("dce_dirs", [])
    config["training"].setdefault("class_weights", None)
    # Read maxPIRADS / labels from the split CSV only: never open H5 volumes
    # or a tabular CSV, whatever model the config was written for.
    config["model"]["model"] = "pirads_rule"
    config["data"]["tabular_csv"] = None

    mode = config["training"].get("mode", "pirads")
    if mode == "pirads":
        print("WARNING: training.mode is 'pirads', so the target is itself derived "
              "from maxPIRADS and the AUC is trivially high. Use mode: cspca.")

    val_csv = args.val_csv or config["paths"]["valid_csv"]
    series = [SERIES_NAME_TO_ENUM[s] for s in config["data"]["series"]]
    # mode="val" -> same filtering/labels as eval.py. The cohort is fixed by
    # the dataset's metadata, so there is no need to iterate the loader.
    dataset = build_dataset(val_csv, config, "val", series)
    meta = dataset.df_metadata

    df = pd.DataFrame({
        "AccessionNumber": meta["AccessionNumber"].astype(int),
        "PatientID": meta["PatientID"].astype(int),
        "target": meta["target"].astype(int),
        "maxPIRADS": pd.to_numeric(meta["maxPIRADS"], errors="coerce"),
    })
    bad = ~df["maxPIRADS"].isin(VALID_PIRADS)
    if bad.any():
        print(f"Dropping {int(bad.sum())} exams with maxPIRADS outside 1-5 "
              f"(values: {sorted(df.loc[bad, 'maxPIRADS'].unique().tolist())})")
        df = df.loc[~bad]
    df = df.reset_index(drop=True)
    df["maxPIRADS"] = df["maxPIRADS"].astype(int)

    n_pos = int(df["target"].sum())
    print(f"Evaluating on {len(df)} exams ({df['PatientID'].nunique()} patients, "
          f"{n_pos} positive / {len(df) - n_pos} negative) from {val_csv} [mode: {mode}]")

    # Per-exam CSV in eval.py's format.
    score = (df["maxPIRADS"] - 1) / 4.0
    out = df.copy()
    out["prob_0"] = 1.0 - score
    out["prob_1"] = score
    out["pred_label"] = (df["maxPIRADS"] >= args.cutoff).astype(int)
    out["Predictions"] = [str([p0, p1]) for p0, p1 in zip(out["prob_0"], out["prob_1"])]
    out["Targets"] = out["target"]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Saved {len(out)} predictions to {out_path}")

    if df["target"].nunique() < 2:
        print("Only one class present in targets; skipping AUC.")
        return

    print("\nmaxPIRADS breakdown:")
    print(per_score_table(df).to_string(float_format=lambda x: f"{x:.3f}"))
    print("\nOperating points (predict positive when maxPIRADS >= k):")
    print(operating_points(df).to_string(float_format=lambda x: f"{x:.3f}"))

    auc = roc_auc_score(df["target"], df["maxPIRADS"])
    line = f"\nAUC (maxPIRADS as ordinal score): {auc:.4f}"
    if args.n_boot > 0:
        seed = args.seed if args.seed is not None else config["training"].get("seed", 42)
        lo, hi, n_ok = bootstrap_auc_ci(df, args.n_boot, seed)
        line += f" | 95% CI [{lo:.4f}, {hi:.4f}] (patient-level bootstrap, {n_ok} resamples)"
    print(line)
    acc = (out["pred_label"] == out["target"]).mean()
    print(f"accuracy @ maxPIRADS >= {args.cutoff}: {acc:.4f}")


if __name__ == "__main__":
    main()
