"""False-alarm rate by type of no-fire image (no retraining).

Groups no-fire test images by FESB MLID pixel classes, UAVs-FFDB time of day,
D-Fire image source and brightness, and writes false_alarms_by_group.csv.

Usage:
    python 04_experiments/src/false_alarms.py --results RESULTS_FOLDER
"""

import argparse
import csv
import os
import re
import statistics

MODELS = ["resnet50", "mobilenet_v2", "efficientnet_b3", "densenet121", "vit_b16"]
TRAIN_DATASETS = ["R1", "R2", "R3", "R4"]
MODE = "finetuned"
SEED = 42
THRESHOLD = 0.5

MANIFEST_FOLDER = "03_datasets/processed/manifest"
SPLIT_FILE = "03_datasets/processed/upload/splits/within_dataset.csv"

DARK_LIMIT = 60
BRIGHT_LIMIT = 120

DATASET_NAMES = {"ALT_WILDFIRE": "S4"}


def read_options():
    parser = argparse.ArgumentParser(description="False alarms by type of no-fire image")
    parser.add_argument("--results", default="04_experiments/results_tf", help="folder with the run folders")
    return parser.parse_args()


def brightness_group(value):
    if value < DARK_LIMIT:
        return "dark"
    if value <= BRIGHT_LIMIT:
        return "medium"
    return "bright"


def fesb_group(native_label):
    classes = native_label.split(";")
    clouds = "clouds_fog" in classes
    sun = "sun" in classes
    if clouds and sun:
        return "clouds/fog and sun"
    if clouds:
        return "clouds/fog"
    if sun:
        return "sun"
    return "neither"


def read_image_info():
    """upload image path -> {dataset, label, groups}"""
    manifest = {}
    for name in os.listdir(MANIFEST_FOLDER):
        if not name.endswith(".csv"):
            continue
        with open(os.path.join(MANIFEST_FOLDER, name), newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                manifest[(row["dataset"], row["rel_path"])] = row

    info = {}
    with open(SPLIT_FILE, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            m = manifest.get((row["dataset"], row["rel_path"]))
            if m is None:
                continue
            dataset = DATASET_NAMES.get(row["dataset"], row["dataset"])
            groups = {}
            if m["mean_lum"]:
                groups["brightness"] = brightness_group(float(m["mean_lum"]))
            if dataset == "S2":
                groups["fesb_classes"] = fesb_group(m["native_label"])
            if dataset == "S3":
                groups["time_of_day"] = "evening" if m["native_label"].startswith("evening") else "pre-evening"
            if dataset == "R4":
                file_name = row["rel_path"].split("/")[-1]
                match = re.match(r"[A-Za-z]+", file_name)
                groups["source"] = match.group(0) if match else "other"
            info[row["image"]] = {"label": 1 if row["l2"] == "fire_or_smoke" else 0, "groups": groups}
    return info


def main():
    options = read_options()
    info = read_image_info()
    print(f"images with inventory information: {len(info)}")

    rates = {}
    for model in MODELS:
        for train in TRAIN_DATASETS:
            path = os.path.join(options.results, f"within_{train}_{model}_{MODE}_seed{SEED}", "predictions.csv")
            if not os.path.exists(path):
                print(f"missing {path}")
                continue
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    test = DATASET_NAMES.get(row["test_dataset"], row["test_dataset"])
                    if test == train:
                        continue
                    item = info.get(row["image"])
                    if item is None:
                        continue
                    called_fire = float(row["fire_probability"]) >= THRESHOLD
                    for grouping, group in item["groups"].items():
                        if item["label"] == 1 and grouping != "time_of_day":
                            continue
                        key = (model, test, grouping, group, item["label"])
                        hits_total = rates.setdefault(key, {}).setdefault(train, [0, 0])
                        hits_total[1] += 1
                        if called_fire:
                            hits_total[0] += 1

    out_rows = []
    for (model, test, grouping, group, label), per_train in sorted(rates.items()):
        values = [hits / total for hits, total in per_train.values() if total > 0]
        images = max(total for _, total in per_train.values())
        out_rows.append({
            "model": model, "test_dataset": test, "grouping": grouping, "group": group,
            "images": "fire images" if label == 1 else "no-fire images",
            "count": images,
            "rate": round(statistics.mean(values), 4),
            "measure": "recall" if label == 1 else "false_alarm_rate",
        })

    out_path = os.path.join(options.results, "false_alarms_by_group.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"written: {out_path}")


if __name__ == "__main__":
    main()
