"""Step 3b: cluster-aware train/val/test split (70/15/15).

Assigns whole clusters to splits, largest first, balancing every (dataset, label)
cell against each split's quota. A cluster gets the same split in every dataset.

Usage:
    python 04_experiments/src/make_splits_within.py
"""

import csv
import random
from collections import Counter, defaultdict
from pathlib import Path


TARGET = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLITS = list(TARGET)

RANDOM_SEED = 0

PROJECT = Path(__file__).resolve().parents[2]
SPLITS_FOLDER = PROJECT / "03_datasets" / "processed" / "splits"
INPUT_FILE = SPLITS_FOLDER / "all_images.csv"

OUTPUT_COLUMNS = ["dataset", "rel_path", "l2", "l4", "sha256", "cluster_id", "split"]


def read_usable_images():
    """Return a list of rows (dicts) for the usable images only."""
    images = []
    with open(INPUT_FILE, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["usable"] == "1":
                images.append(row)
    return images


def count_cells_per_cluster(images):
    """A "cell" is one (dataset, label) pair, e.g. ("R4", "none")."""
    cells = defaultdict(Counter)
    for row in images:
        cells[row["cluster_id"]][(row["dataset"], row["l2"])] += 1
    return cells


def choose_split(cluster_cells, filled, totals):
    """Pick the split that most needs the images of this cluster."""
    best_split = None
    best_need = None
    for split in SPLITS:
        need = 0.0
        for cell, n in cluster_cells.items():
            quota = TARGET[split] * totals[cell]
            need += n * (quota - filled[split][cell]) / quota
        if best_need is None or need > best_need:
            best_split = split
            best_need = need
    return best_split


def assign_clusters(cells):
    """Return a dict: cluster_id -> split."""
    totals = Counter()
    for cluster_cells in cells.values():
        totals.update(cluster_cells)

    order = list(cells)
    random.Random(RANDOM_SEED).shuffle(order)
    order.sort(key=lambda c: sum(cells[c].values()), reverse=True)

    filled = {split: Counter() for split in SPLITS}

    split_of_cluster = {}
    for cluster_id in order:
        split = choose_split(cells[cluster_id], filled, totals)
        split_of_cluster[cluster_id] = split
        filled[split].update(cells[cluster_id])
    return split_of_cluster


def check_clusters_not_shared(rows):
    """Return the number of clusters found in more than one split (must be 0)."""
    splits_of_cluster = defaultdict(set)
    for row in rows:
        splits_of_cluster[row["cluster_id"]].add(row["split"])
    return sum(1 for s in splits_of_cluster.values() if len(s) > 1)


def summarise(rows):
    """Return dict dataset -> {split: (images, fire images)}."""
    images = defaultdict(Counter)
    fire = defaultdict(Counter)
    for row in rows:
        images[row["dataset"]][row["split"]] += 1
        if row["l2"] == "fire_or_smoke":
            fire[row["dataset"]][row["split"]] += 1
    summary = {}
    for dataset in images:
        summary[dataset] = {s: (images[dataset][s], fire[dataset][s]) for s in SPLITS}
    return summary


def write_table(rows):
    path = SPLITS_FOLDER / "within_dataset.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_report(summary, n_shared, n_clusters):
    lines = []
    lines.append("# Step 3b check: within-dataset splits")
    lines.append("")
    lines.append("Made by `04_experiments/src/make_splits_within.py`. Output: `within_dataset.csv`.")
    lines.append(f"Target: train {TARGET['train']:.0%} / val {TARGET['val']:.0%} / "
                 f"test {TARGET['test']:.0%}. Random seed {RANDOM_SEED}.")
    lines.append("")

    lines.append("## Leak check")
    lines.append("")
    if n_shared == 0:
        lines.append(f"Passed. All {n_clusters} clusters are in exactly one split.")
    else:
        lines.append(f"**FAILED: {n_shared} clusters are in more than one split.**")
    lines.append("")

    lines.append("## Split sizes per dataset")
    lines.append("")
    lines.append("Each cell: images (share of the dataset), and the fire/smoke share inside that split.")
    lines.append("")
    lines.append("| ID | Images | Train | Val | Test | Fire share: all / train / val / test |")
    lines.append("|---|---|---|---|---|---|")
    for dataset, by_split in summary.items():
        total = sum(n for n, _ in by_split.values())
        total_fire = sum(f for _, f in by_split.values())
        cells = []
        fire_shares = [f"{total_fire / total:.0%}"]
        for split in SPLITS:
            n, f = by_split[split]
            cells.append(f"{n} ({n / total:.1%})")
            fire_shares.append(f"{f / n:.0%}" if n else "-")
        lines.append(f"| {dataset} | {total} | " + " | ".join(cells) + " | "
                     + " / ".join(fire_shares) + " |")
    lines.append("")
    lines.append("Small datasets (BoWFire, FESB, UAVs-FFDB) can be slightly off the target, "
                 "because their clusters are large compared with the dataset.")
    lines.append("")

    path = SPLITS_FOLDER / "step3b_check.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main():
    print("Reading step-3a table ...")
    images = read_usable_images()
    print(f"  {len(images)} usable images")

    print("Assigning clusters to splits ...")
    cells = count_cells_per_cluster(images)
    split_of_cluster = assign_clusters(cells)
    for row in images:
        row["split"] = split_of_cluster[row["cluster_id"]]

    n_shared = check_clusters_not_shared(images)
    summary = summarise(images)

    table_path = write_table(images)
    report_path = write_report(summary, n_shared, len(cells))

    print(f"  clusters in more than one split: {n_shared} (must be 0)")
    print(f"Wrote {table_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
