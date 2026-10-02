"""Step 3c: cross-dataset folds.

Builds the Reference-to-Shift fold and the leave-one-dataset-out folds. Training
images that share a cluster with any test image are removed, and byte-identical
copies across datasets are kept once.

Usage:
    python 04_experiments/src/make_cross_dataset_folds.py
"""

import csv
from collections import Counter, defaultdict
from pathlib import Path


REFERENCE = ["R1", "R2", "R3", "R4"]

SHIFT = ["S1", "S2", "S3", "ALT_WILDFIRE"]

PROJECT = Path(__file__).resolve().parents[2]
SPLITS_FOLDER = PROJECT / "03_datasets" / "processed" / "splits"
INPUT_FILE = SPLITS_FOLDER / "within_dataset.csv"
OUTPUT_FOLDER = SPLITS_FOLDER / "folds"

OUTPUT_COLUMNS = ["dataset", "rel_path", "l2", "l4", "sha256", "cluster_id", "role"]


def read_images():
    with open(INPUT_FILE, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def datasets_per_cluster(images):
    """Return a dict: cluster_id -> set of datasets that have an image in it."""
    found = defaultdict(set)
    for row in images:
        found[row["cluster_id"]].add(row["dataset"])
    return found


def build_fold(images, cluster_datasets, train_datasets, test_rule):
    """Return (fold rows, Counter of removed images per dataset).  train_datasets : datasets whose train/val parts are used for training test_rule      : dict dataset -> which part is the test set ("all" = every image, "test" = only its test part)"""
    test_datasets = set(test_rule)
    fold = []
    removed = Counter()
    seen_sha = set()

    for row in images:
        part = test_rule.get(row["dataset"])
        if part == "all" or (part == "test" and row["split"] == "test"):
            fold.append(dict(row, role="test"))

    size = Counter(row["dataset"] for row in images)
    smallest_first = sorted(images, key=lambda row: size[row["dataset"]])
    for row in smallest_first:
        if row["dataset"] not in train_datasets:
            continue
        if row["split"] not in ("train", "val"):
            continue
        if cluster_datasets[row["cluster_id"]] & test_datasets:
            removed[row["dataset"]] += 1
            continue
        if row["sha256"] in seen_sha:
            removed[row["dataset"] + " (copy in other dataset)"] += 1
            continue
        seen_sha.add(row["sha256"])
        fold.append(dict(row, role=row["split"]))

    return fold, removed


def make_all_folds(images):
    """Return a dict: fold name -> (fold rows, removed counter)."""
    cluster_datasets = datasets_per_cluster(images)
    folds = {}

    shift_rule = {dataset: "all" for dataset in SHIFT}
    folds["ref_to_shift"] = build_fold(images, cluster_datasets, REFERENCE, shift_rule)

    for held_out in REFERENCE:
        others = [d for d in REFERENCE if d != held_out]
        folds[f"lodo_{held_out}"] = build_fold(images, cluster_datasets, others, {held_out: "test"})

    return folds


def leak_check(fold):
    """Return (clusters in both train/val and test, sha256 in both)."""
    train_clusters, test_clusters = set(), set()
    train_sha, test_sha = set(), set()
    for row in fold:
        if row["role"] == "test":
            test_clusters.add(row["cluster_id"])
            test_sha.add(row["sha256"])
        else:
            train_clusters.add(row["cluster_id"])
            train_sha.add(row["sha256"])
    return len(train_clusters & test_clusters), len(train_sha & test_sha)


def write_fold(name, fold):
    path = OUTPUT_FOLDER / f"{name}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(fold)


def write_report(folds):
    lines = []
    lines.append("# Step 3c check: cross-dataset folds")
    lines.append("")
    lines.append("Made by `04_experiments/src/make_cross_dataset_folds.py`. One CSV per fold in this folder.")
    lines.append(f"Reference (training) datasets: {', '.join(REFERENCE)}. "
                 f"Shift (test-only) datasets: {', '.join(SHIFT)}. DeepFire is not used.")
    lines.append("")

    for name, (fold, removed) in folds.items():
        shared_clusters, shared_sha = leak_check(fold)
        lines.append(f"## {name}")
        lines.append("")
        if shared_clusters == 0 and shared_sha == 0:
            lines.append("Leak check passed: no cluster and no identical image is in both train/val and test.")
        else:
            lines.append(f"**LEAK CHECK FAILED: {shared_clusters} clusters and {shared_sha} "
                         "identical images are in both train/val and test.**")
        lines.append("")

        counts = defaultdict(Counter)
        fire = defaultdict(Counter)
        for row in fold:
            counts[row["dataset"]][row["role"]] += 1
            if row["l2"] == "fire_or_smoke":
                fire[row["dataset"]][row["role"]] += 1
        lines.append("| Dataset | Train | Val | Test | Fire share (of its images in this fold) |")
        lines.append("|---|---|---|---|---|")
        for dataset in REFERENCE + SHIFT:
            if dataset not in counts:
                continue
            c = counts[dataset]
            total = sum(c.values())
            share = sum(fire[dataset].values()) / total
            lines.append(f"| {dataset} | {c['train']} | {c['val']} | {c['test']} | {share:.0%} |")
        totals = Counter()
        for c in counts.values():
            totals.update(c)
        lines.append(f"| **Total** | {totals['train']} | {totals['val']} | {totals['test']} | |")
        lines.append("")

        if removed:
            parts = [f"{n} {what}" for what, n in sorted(removed.items())]
            lines.append("Removed from train/val: " + ", ".join(parts) + ".")
        else:
            lines.append("Removed from train/val: nothing.")
        lines.append("")

    path = OUTPUT_FOLDER / "step3c_check.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    print("Reading within-dataset split ...")
    images = read_images()
    print(f"  {len(images)} images")

    print("Building folds ...")
    folds = make_all_folds(images)
    for name, (fold, removed) in folds.items():
        write_fold(name, fold)
        shared_clusters, shared_sha = leak_check(fold)
        print(f"  {name}: {len(fold)} rows, removed {sum(removed.values())}, "
              f"leaks {shared_clusters} clusters / {shared_sha} images (must be 0 / 0)")

    report_path = write_report(folds)
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
