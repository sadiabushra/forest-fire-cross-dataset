"""Extra metrics from the saved baseline results (no retraining).

MCC, NPV and Youden's J from the confusion matrices in <model>_scores.csv, and
recall at 3%, 5% and 10% false alarms from the per-image predictions.
Writes extra_metrics.csv.

Usage:
    python 04_experiments/src/extra_metrics.py --results RESULTS_FOLDER
"""

import argparse
import csv
import math
import os
import statistics

MODELS = ["resnet50", "mobilenet_v2", "efficientnet_b3", "densenet121", "vit_b16"]
MODES = ["frozen", "finetuned"]
TRAIN_DATASETS = ["R1", "R2", "R3", "R4"]

FALSE_ALARM_LEVELS = [0.03, 0.05, 0.10]


def read_options():
    parser = argparse.ArgumentParser(description="Extra metrics from saved results")
    parser.add_argument("--results", default="04_experiments/results_tf",
                        help="folder with <model>_scores.csv and the run folders")
    return parser.parse_args()


def confusion_metrics(row):
    """MCC, NPV and Youden's J from one row of a scores file."""
    tn = float(row["true_negatives"])
    fp = float(row["false_positives"])
    fn = float(row["false_negatives"])
    tp = float(row["true_positives"])

    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denominator if denominator > 0 else 0.0

    npv = tn / (tn + fn) if tn + fn > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    specificity = tn / (tn + fp) if tn + fp > 0 else 0.0
    youden = recall + specificity - 1
    return {"mcc": mcc, "npv": npv, "youden_j": youden}


def recall_at_false_alarms(probs, labels, level):
    """Share of fire images caught when only `level` of no-fire images are called fire."""
    negatives = sorted([p for p, y in zip(probs, labels) if y == 0], reverse=True)
    positives = [p for p, y in zip(probs, labels) if y == 1]
    if not negatives or not positives:
        return None

    allowed = int(len(negatives) * level)
    threshold = negatives[allowed] if allowed < len(negatives) else -1.0

    caught = 0
    for p in positives:
        if p > threshold:
            caught += 1
    return caught / len(positives)


def mean_or_blank(values):
    values = [v for v in values if v is not None]
    return round(statistics.mean(values), 4) if values else ""


def main():
    options = read_options()
    table = {}

    def add(model, mode, kind, train, metric, value):
        key = (model, mode, kind)
        table.setdefault(key, {}).setdefault(metric, {}).setdefault(train, []).append(value)

    for model in MODELS:
        path = os.path.join(options.results, f"{model}_scores.csv")
        if not os.path.exists(path):
            print(f"missing {path}, skipped")
            continue
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                for metric, value in confusion_metrics(row).items():
                    add(model, row["mode"], row["kind"], row["train_dataset"], metric, value)

    for model in MODELS:
        for mode in MODES:
            for train in TRAIN_DATASETS:
                path = os.path.join(options.results, f"within_{train}_{model}_{mode}_seed42", "predictions.csv")
                if not os.path.exists(path):
                    continue
                per_test = {}
                with open(path, newline="") as f:
                    for row in csv.DictReader(f):
                        probs, labels = per_test.setdefault(row["test_dataset"], ([], []))
                        probs.append(float(row["fire_probability"]))
                        labels.append(int(row["label"]))
                for test, (probs, labels) in per_test.items():
                    kind = "within" if test == train else "cross"
                    for level in FALSE_ALARM_LEVELS:
                        name = f"recall_at_{int(level * 100)}pct_false_alarms"
                        add(model, mode, kind, train, name, recall_at_false_alarms(probs, labels, level))

    metrics = ["mcc", "npv", "youden_j"] + [f"recall_at_{int(l * 100)}pct_false_alarms" for l in FALSE_ALARM_LEVELS]
    out_rows = []
    for (model, mode, kind), per_metric in sorted(table.items()):
        out = {"model": model, "mode": mode, "kind": kind}
        for metric in metrics:
            per_train = [mean_or_blank(v) for v in per_metric.get(metric, {}).values()]
            out[metric] = mean_or_blank([v for v in per_train if v != ""])
        out_rows.append(out)

    out_path = os.path.join(options.results, "extra_metrics.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "mode", "kind"] + metrics)
        writer.writeheader()
        writer.writerows(out_rows)

    for row in out_rows:
        print(row)
    print(f"written: {out_path}")


if __name__ == "__main__":
    main()
