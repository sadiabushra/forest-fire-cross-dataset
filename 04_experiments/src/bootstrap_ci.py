"""Bootstrap 95% intervals for macro-F1 (no retraining).

Paired resampling of every test set for the fine-tuned seed-42 runs: intervals per
test set, for the within and cross scores, and for the difference to ViT-B/16.
Writes bootstrap_test_sets.csv, bootstrap_summary.csv and bootstrap_vs_vit.csv.

Usage:
    python 04_experiments/src/bootstrap_ci.py --results RESULTS_FOLDER
"""

import argparse
import csv
import os

import numpy as np

MODELS = ["resnet50", "mobilenet_v2", "efficientnet_b3", "densenet121", "vit_b16"]
TRAIN_DATASETS = ["R1", "R2", "R3", "R4"]
REFERENCE_MODEL = "vit_b16"
MODE = "finetuned"
SEED = 42

BOOTSTRAP_SAMPLES = 1000
RANDOM_SEED = 0
THRESHOLD = 0.5

DATASET_NAMES = {"ALT_WILDFIRE": "S4"}


def read_options():
    parser = argparse.ArgumentParser(description="Bootstrap confidence intervals")
    parser.add_argument("--results", default="04_experiments/results_tf",
                        help="folder with the run folders")
    return parser.parse_args()


def read_predictions(path):
    """test dataset -> {image: (label, predicted)}"""
    per_test = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            test = DATASET_NAMES.get(row["test_dataset"], row["test_dataset"])
            predicted = 1 if float(row["fire_probability"]) >= THRESHOLD else 0
            per_test.setdefault(test, {})[row["image"]] = (int(row["label"]), predicted)
    return per_test


def macro_f1(labels, predicted):
    """Macro-F1 for many resamples at once."""
    tp = np.sum((labels == 1) & (predicted == 1), axis=-1)
    tn = np.sum((labels == 0) & (predicted == 0), axis=-1)
    fp = np.sum((labels == 0) & (predicted == 1), axis=-1)
    fn = np.sum((labels == 1) & (predicted == 0), axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1_fire = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
        f1_none = np.where(2 * tn + fn + fp > 0, 2 * tn / (2 * tn + fn + fp), 0.0)
    return (f1_fire + f1_none) / 2


def interval(values):
    return np.percentile(values, 2.5), np.percentile(values, 97.5)


def main():
    options = read_options()
    rng = np.random.default_rng(RANDOM_SEED)

    predictions = {}
    for model in MODELS:
        for train in TRAIN_DATASETS:
            path = os.path.join(options.results, f"within_{train}_{model}_{MODE}_seed{SEED}", "predictions.csv")
            if not os.path.exists(path):
                print(f"missing {path}")
                continue
            predictions.setdefault(model, {})[train] = read_predictions(path)

    boot = {}
    point = {}
    test_rows = []
    for train in TRAIN_DATASETS:
        tests = sorted(predictions[REFERENCE_MODEL][train].keys())
        for test in tests:
            images = sorted(predictions[REFERENCE_MODEL][train][test].keys())
            n = len(images)
            index = rng.integers(0, n, size=(BOOTSTRAP_SAMPLES, n))
            for model in MODELS:
                pairs = predictions[model][train][test]
                labels = np.array([pairs[i][0] for i in images])
                predicted = np.array([pairs[i][1] for i in images])
                values = macro_f1(labels[index], predicted[index])
                score = float(macro_f1(labels[None, :], predicted[None, :])[0])
                boot.setdefault(model, {}).setdefault(train, {})[test] = values
                point.setdefault(model, {}).setdefault(train, {})[test] = score
                low, high = interval(values)
                test_rows.append({"model": model, "train_dataset": train, "test_dataset": test,
                                  "kind": "within" if test == train else "cross", "images": n,
                                  "macro_f1": round(score, 4), "ci_low": round(low, 4), "ci_high": round(high, 4)})

    def headline(model, kind, use_boot):
        """Within or cross score: mean over test sets per training set, then over training sets."""
        source = boot if use_boot else point
        per_train = []
        for train in TRAIN_DATASETS:
            tests = [t for t in source[model][train] if (t == train) == (kind == "within")]
            per_train.append(np.mean([source[model][train][t] for t in tests], axis=0))
        return np.mean(per_train, axis=0)

    summary_rows = []
    diff_rows = []
    for kind in ["within", "cross"]:
        reference = headline(REFERENCE_MODEL, kind, True)
        for model in MODELS:
            values = headline(model, kind, True)
            low, high = interval(values)
            summary_rows.append({"model": model, "kind": kind,
                                 "macro_f1": round(float(headline(model, kind, False)), 4),
                                 "ci_low": round(low, 4), "ci_high": round(high, 4)})
            if model != REFERENCE_MODEL:
                diff = reference - values
                low, high = interval(diff)
                diff_rows.append({"kind": kind, "comparison": f"{REFERENCE_MODEL} minus {model}",
                                  "difference": round(float(np.mean(diff)), 4),
                                  "ci_low": round(low, 4), "ci_high": round(high, 4),
                                  "clear": "yes" if low > 0 or high < 0 else "no"})

    outputs = [("bootstrap_test_sets.csv", test_rows), ("bootstrap_summary.csv", summary_rows),
               ("bootstrap_vs_vit.csv", diff_rows)]
    for name, rows in outputs:
        path = os.path.join(options.results, name)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"written: {path}")

    for row in summary_rows + diff_rows:
        print(row)


if __name__ == "__main__":
    main()
