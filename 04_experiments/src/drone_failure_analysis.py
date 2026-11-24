"""Analysis of models trained on FASDD_UAV and tested on UAVs-FFDB.

AUC per time of day, image statistics of both datasets, and a figure of the most
confident mistakes. Writes drone_failure_auc.csv and a figure.

Usage:
    python 04_experiments/src/drone_failure_analysis.py --results RESULTS_FOLDER
"""

import argparse
import csv
import os
import random

import numpy as np
from PIL import Image

MODELS = ["resnet50", "mobilenet_v2", "efficientnet_b3", "densenet121", "vit_b16"]
TRAIN_DATASETS = ["R1", "R2", "R3", "R4"]
FIGURE_MODEL = "densenet121"

UPLOAD = "03_datasets/processed/upload"
SPLIT_FILE = UPLOAD + "/splits/within_dataset.csv"
MANIFEST_FOLDER = "03_datasets/processed/manifest"
FIGURE_PATH = "10_thesis_draft/figures/drone_failure.jpg"

THUMB = 220
PER_ROW = 5
GAP = 6
RANDOM_SEED = 1


def read_options():
    parser = argparse.ArgumentParser(description="R3 -> S3 failure analysis")
    parser.add_argument("--results", default="04_experiments/results_tf", help="folder with the run folders")
    return parser.parse_args()


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def auc(labels, scores):
    """Area under the ROC curve from ranks (0.5 = guessing, below 0.5 = reversed)."""
    labels = np.array(labels)
    scores = np.array(scores)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def image_stats(manifest_rows, label):
    rows = [r for r in manifest_rows if r["l2"] == label and r["mean_lum"]]
    return {
        "images": len(rows),
        "brightness": round(np.mean([float(r["mean_lum"]) for r in rows]), 1),
        "contrast": round(np.mean([float(r["std_lum"]) for r in rows]), 1),
        "sharpness_median": round(float(np.median([float(r["lap_var"]) for r in rows])), 1),
    }


def make_figure(rows_of_images, path):
    width = PER_ROW * THUMB + (PER_ROW - 1) * GAP
    height = len(rows_of_images) * THUMB + (len(rows_of_images) - 1) * GAP
    figure = Image.new("RGB", (width, height), "white")
    for r, images in enumerate(rows_of_images):
        for c, image_path in enumerate(images[:PER_ROW]):
            thumb = Image.open(os.path.join(UPLOAD, image_path)).convert("RGB")
            side = min(thumb.size)
            left = (thumb.width - side) // 2
            top = (thumb.height - side) // 2
            thumb = thumb.crop((left, top, left + side, top + side)).resize((THUMB, THUMB))
            figure.paste(thumb, (c * (THUMB + GAP), r * (THUMB + GAP)))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    figure.save(path, quality=90)


def main():
    options = read_options()

    s3_manifest = {r["rel_path"]: r for r in read_csv(os.path.join(MANIFEST_FOLDER, "S3.csv"))}
    split = {r["image"]: r for r in read_csv(SPLIT_FILE)}

    def time_of_day(image):
        native = s3_manifest[split[image]["rel_path"]]["native_label"]
        return "evening" if native.startswith("evening") else "pre-evening"

    out_rows = []
    figure_predictions = None
    for model in MODELS:
        for train in TRAIN_DATASETS:
            path = os.path.join(options.results, f"within_{train}_{model}_finetuned_seed42", "predictions.csv")
            rows = [r for r in read_csv(path) if r["test_dataset"] == "S3"]
            labels = [int(r["label"]) for r in rows]
            scores = [float(r["fire_probability"]) for r in rows]
            times = [time_of_day(r["image"]) for r in rows]
            result = {"model": model, "train_dataset": train, "images": len(rows),
                      "auc_all": round(auc(labels, scores), 3)}
            for t in ["pre-evening", "evening"]:
                keep = [i for i in range(len(rows)) if times[i] == t]
                result[f"auc_{t}"] = round(auc([labels[i] for i in keep], [scores[i] for i in keep]), 3)
            result["mean_p_fire_images"] = round(np.mean([s for s, y in zip(scores, labels) if y == 1]), 3)
            result["mean_p_nofire_images"] = round(np.mean([s for s, y in zip(scores, labels) if y == 0]), 3)
            out_rows.append(result)
            print(result)
            if model == FIGURE_MODEL and train == "R3":
                figure_predictions = rows

    out_path = os.path.join(options.results, "drone_failure_auc.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"written: {out_path}")

    r3_manifest = read_csv(os.path.join(MANIFEST_FOLDER, "R3.csv"))
    for name, manifest in [("R3", r3_manifest), ("S3", list(s3_manifest.values()))]:
        for label in ["fire_or_smoke", "none"]:
            print(name, label, image_stats(manifest, label))

    pre = [r for r in figure_predictions if time_of_day(r["image"]) == "pre-evening"]
    false_alarms = sorted([r for r in pre if r["label"] == "0"], key=lambda r: -float(r["fire_probability"]))
    missed = sorted([r for r in pre if r["label"] == "1"], key=lambda r: float(r["fire_probability"]))
    rng = random.Random(RANDOM_SEED)
    r3_train = [r for r in split.values() if r["dataset"] == "R3" and r["split"] == "train"]
    r3_examples = (rng.sample([r for r in r3_train if r["l2"] == "fire_or_smoke"], 3)
                   + rng.sample([r for r in r3_train if r["l2"] == "none"], 2))
    make_figure([[r["image"] for r in r3_examples],
                 [r["image"] for r in false_alarms[::2]],
                 [r["image"] for r in missed[::2]]], FIGURE_PATH)
    print(f"written: {FIGURE_PATH}")


if __name__ == "__main__":
    main()
