"""Recall of fire-only, smoke-only and fire-and-smoke images (no retraining).

Joins the saved predictions with the four-class labels of the split file and
writes fire_smoke_recall_runs.csv and fire_smoke_recall_summary.csv.

Usage:
    python 04_experiments/src/fire_smoke_recall.py --results RESULTS_FOLDER [--splits SPLIT_FILE]
"""

import argparse
import csv
import glob
import os
import statistics

DEFAULT_SPLIT_FILE = "03_datasets/processed/upload/splits/within_dataset.csv"
SUB_LABELS = ["fire", "smoke", "fire_and_smoke"]
THRESHOLD = 0.5

DATASET_NAMES = {"ALT_WILDFIRE": "S4"}


def read_options():
    parser = argparse.ArgumentParser(description="Fire vs smoke recall")
    parser.add_argument("--results", default="04_experiments/results_tf",
                        help="folder with the run folders (within_<R>_<model>_<mode>_seed<N>)")
    parser.add_argument("--splits", default=DEFAULT_SPLIT_FILE, help="split file with the l4 column")
    return parser.parse_args()


def read_sub_labels(path):
    """image path -> fire / smoke / fire_and_smoke (positives only)."""
    sub = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["l2"] != "fire_or_smoke":
                continue
            label = row["l4"]
            if not label and row["dataset"] == "S2":
                label = "smoke"
            if not label and row["dataset"] == "S3":
                label = "fire"
            if label in SUB_LABELS:
                sub[row["image"]] = label
    return sub


def parse_run_name(folder):
    """within_R4_vit_b16_finetuned_seed42 -> (R4, vit_b16, finetuned, 42)."""
    parts = os.path.basename(folder).split("_")
    train = parts[1]
    seed = parts[-1].replace("seed", "")
    mode = parts[-2]
    model = "_".join(parts[2:-2])
    return train, model, mode, seed


def main():
    options = read_options()
    sub = read_sub_labels(options.splits)
    print(f"positive images with a fire/smoke sub-label: {len(sub)}")

    run_rows = []
    folders = sorted(glob.glob(os.path.join(options.results, "within_*_seed*")))
    for folder in folders:
        path = os.path.join(folder, "predictions.csv")
        if not os.path.exists(path):
            continue
        train, model, mode, seed = parse_run_name(folder)

        counts = {}
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                label = sub.get(row["image"])
                if label is None:
                    continue
                test = DATASET_NAMES.get(row["test_dataset"], row["test_dataset"])
                found_total = counts.setdefault((test, label), [0, 0])
                found_total[1] += 1
                if float(row["fire_probability"]) >= THRESHOLD:
                    found_total[0] += 1

        for (test, label), (found, total) in sorted(counts.items()):
            run_rows.append({
                "model": model, "mode": mode, "seed": seed, "train_dataset": train,
                "test_dataset": test, "kind": "within" if test == train else "cross",
                "sub_label": label, "images": total, "detected": found,
                "recall": round(found / total, 4),
            })

    out_runs = os.path.join(options.results, "fire_smoke_recall_runs.csv")
    with open(out_runs, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(run_rows[0].keys()))
        writer.writeheader()
        writer.writerows(run_rows)

    groups = {}
    for r in run_rows:
        key = (r["model"], r["mode"], r["kind"], r["sub_label"])
        groups.setdefault(key, {}).setdefault(r["train_dataset"], []).append(r["recall"])

    summary = []
    for (model, mode, kind, label), per_train in sorted(groups.items()):
        means = [statistics.mean(v) for v in per_train.values()]
        summary.append({"model": model, "mode": mode, "kind": kind, "sub_label": label,
                        "recall": round(statistics.mean(means), 4),
                        "training_datasets": len(means)})

    out_summary = os.path.join(options.results, "fire_smoke_recall_summary.csv")
    with open(out_summary, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    for s in summary:
        print(s)
    print(f"written: {out_runs}")
    print(f"written: {out_summary}")


if __name__ == "__main__":
    main()
