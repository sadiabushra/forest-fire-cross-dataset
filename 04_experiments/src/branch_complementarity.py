"""How complementary each CNN is to ViT-B/16 (no retraining).

On the test parts of the other training datasets (R1-R4, not S1-S4): share of
images both models get wrong, disagreement, and macro-F1 of the averaged fire
probabilities. Writes branch_complementarity.csv.

Usage:
    python 04_experiments/src/branch_complementarity.py --results RESULTS_FOLDER
"""

import argparse
import csv
import os
import statistics

from sklearn.metrics import f1_score

CNNS = ["resnet50", "mobilenet_v2", "efficientnet_b3", "densenet121"]
REFERENCE = "vit_b16"
DEV_TESTS = ["R1", "R2", "R3", "R4"]
TRAIN_DATASETS = ["R1", "R2", "R3", "R4"]
THRESHOLD = 0.5


def read_options():
    parser = argparse.ArgumentParser(description="Complementarity of each CNN with ViT-B/16")
    parser.add_argument("--results", default="04_experiments/results_tf")
    return parser.parse_args()


def read_predictions(path):
    """(test dataset, image) -> (label, fire probability)."""
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[(row["test_dataset"], row["image"])] = (int(row["label"]), float(row["fire_probability"]))
    return out


def macro_f1(labels, probs):
    predicted = [1 if p >= THRESHOLD else 0 for p in probs]
    return f1_score(labels, predicted, average="macro", labels=[0, 1], zero_division=0)


def main():
    options = read_options()
    rows = []
    for cnn in CNNS:
        per_train = {"double_fault": [], "disagreement": [], "vit_f1": [], "cnn_f1": [], "average_f1": []}
        for train in TRAIN_DATASETS:
            vit = read_predictions(os.path.join(options.results, f"within_{train}_{REFERENCE}_finetuned_seed42", "predictions.csv"))
            other = read_predictions(os.path.join(options.results, f"within_{train}_{cnn}_finetuned_seed42", "predictions.csv"))
            keys = [k for k in vit if k in other and k[0] != train and k[0] in DEV_TESTS]
            tests = sorted({k[0] for k in keys})
            scores = {name: [] for name in per_train}
            both_wrong = one_right = 0
            for test in tests:
                test_keys = [k for k in keys if k[0] == test]
                labels = [vit[k][0] for k in test_keys]
                p_vit = [vit[k][1] for k in test_keys]
                p_cnn = [other[k][1] for k in test_keys]
                p_avg = [(a + b) / 2 for a, b in zip(p_vit, p_cnn)]
                for y, a, b in zip(labels, p_vit, p_cnn):
                    vit_right = (a >= THRESHOLD) == (y == 1)
                    cnn_right = (b >= THRESHOLD) == (y == 1)
                    both_wrong += (not vit_right) and (not cnn_right)
                    one_right += vit_right != cnn_right
                scores["vit_f1"].append(macro_f1(labels, p_vit))
                scores["cnn_f1"].append(macro_f1(labels, p_cnn))
                scores["average_f1"].append(macro_f1(labels, p_avg))
            per_train["double_fault"].append(both_wrong / len(keys))
            per_train["disagreement"].append(one_right / len(keys))
            for name in ("vit_f1", "cnn_f1", "average_f1"):
                per_train[name].append(statistics.mean(scores[name]))
        row = {"cnn": cnn}
        for name, values in per_train.items():
            row[name] = round(statistics.mean(values), 4)
        row["average_minus_vit"] = round(row["average_f1"] - row["vit_f1"], 4)
        rows.append(row)
        print(row)

    out = os.path.join(options.results, "branch_complementarity.csv")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"written: {out}")


if __name__ == "__main__":
    main()
