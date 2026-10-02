"""Step 3a: one table of all images with their labels and clusters.

Joins the step-1 manifests with the step-2 clusters, drops unreadable files,
BoWFire patches and within-dataset exact duplicates, and checks the result.

Usage:
    python 04_experiments/src/build_image_table.py
"""

import csv
from collections import Counter, defaultdict
from pathlib import Path


DATASETS = ["R1", "R2", "R3", "R4", "S1", "S2", "S3", "ALT_DEEPFIRE", "ALT_WILDFIRE"]

PROJECT = Path(__file__).resolve().parents[2]
MANIFEST_FOLDER = PROJECT / "03_datasets" / "processed" / "manifest"
CLUSTERS_FILE = PROJECT / "03_datasets" / "processed" / "near_duplicates" / "clusters.csv"
OUTPUT_FOLDER = PROJECT / "03_datasets" / "processed" / "splits"

OUTPUT_COLUMNS = ["dataset", "rel_path", "orig_split", "l2", "l4", "sha256",
                  "cluster_id", "cluster_size", "usable", "drop_reason"]


def read_manifests():
    """Return a list of manifest rows (dicts), one per image file."""
    rows = []
    for dataset in DATASETS:
        path = MANIFEST_FOLDER / f"{dataset}.csv"
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(row)
    return rows


def read_clusters():
    """Return a dict: (dataset, rel_path) -> clusters.csv row."""
    clusters = {}
    with open(CLUSTERS_FILE, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["dataset"], row["rel_path"])
            if key in clusters:
                raise ValueError(f"Image listed twice in clusters.csv: {key}")
            clusters[key] = row
    return clusters


def build_table(manifest_rows, clusters):
    """Join manifest and clusters."""
    table = []
    problems = []

    next_free_cluster = max(int(r["cluster_id"]) for r in clusters.values()) + 1

    for row in manifest_rows:
        key = (row["dataset"], row["rel_path"])
        cluster_row = clusters.get(key)

        out = {
            "dataset": row["dataset"],
            "rel_path": row["rel_path"],
            "orig_split": row["orig_split"],
            "l2": row["l2"],
            "l4": row["l4"],
            "sha256": row["sha256"],
            "cluster_id": "",
            "cluster_size": "",
            "usable": 1,
            "drop_reason": "",
        }

        if row["readable"] != "1":
            out["usable"] = 0
            out["drop_reason"] = "unreadable"
        elif row["subset"] == "patch_50px":
            out["usable"] = 0
            out["drop_reason"] = "bowfire_patch"
        elif cluster_row is not None:
            out["cluster_id"] = cluster_row["cluster_id"]
            out["cluster_size"] = cluster_row["cluster_size"]
            if cluster_row["l2"] != row["l2"]:
                problems.append(f"L2 differs for {key}: manifest {row['l2']}, clusters {cluster_row['l2']}")
        elif "near_constant" in row["flags"]:
            out["cluster_id"] = str(next_free_cluster)
            out["cluster_size"] = "1"
            next_free_cluster += 1
        else:
            problems.append(f"Usable image has no cluster: {key}")

        if out["usable"] == 0 and cluster_row is not None:
            problems.append(f"Left-out image is in clusters.csv: {key}")

        table.append(out)

    manifest_keys = {(r["dataset"], r["rel_path"]) for r in manifest_rows}
    for key in clusters:
        if key not in manifest_keys:
            problems.append(f"Image in clusters.csv but in no manifest: {key}")

    return table, problems


def mark_exact_duplicates(table):
    """Mark extra copies as not usable."""
    copies = defaultdict(list)
    for row in table:
        if row["usable"] == 1:
            copies[(row["dataset"], row["sha256"])].append(row)

    for group in copies.values():
        if len(group) < 2:
            continue
        labels = {row["l2"] for row in group}
        if len(labels) > 1:
            for row in group:
                row["usable"] = 0
                row["drop_reason"] = "duplicate_label_conflict"
        else:
            group.sort(key=lambda r: r["rel_path"])
            for row in group[1:]:
                row["usable"] = 0
                row["drop_reason"] = "exact_duplicate"


def recount_cluster_sizes(table):
    """Set cluster_size to the number of USABLE images in each cluster."""
    sizes = Counter(row["cluster_id"] for row in table if row["usable"] == 1)
    for row in table:
        if row["usable"] == 1:
            row["cluster_size"] = sizes[row["cluster_id"]]
        else:
            row["cluster_id"] = ""
            row["cluster_size"] = ""


def count_mixed_label_clusters(table):
    """Return (number of mixed clusters, number of images in them, examples)."""
    labels_in_cluster = defaultdict(set)
    members = defaultdict(list)
    for row in table:
        if row["usable"] == 1:
            labels_in_cluster[row["cluster_id"]].add(row["l2"])
            members[row["cluster_id"]].append(row)

    mixed = [c for c, labels in labels_in_cluster.items() if len(labels) > 1]
    images_in_mixed = sum(len(members[c]) for c in mixed)

    mixed.sort(key=lambda c: len(members[c]))
    examples = []
    for c in mixed[:10]:
        parts = [f"{r['dataset']}:{r['rel_path']} ({r['l2']})" for r in members[c][:4]]
        examples.append(f"cluster {c}, {len(members[c])} images: " + "; ".join(parts))
    return len(mixed), images_in_mixed, examples


def count_per_dataset(table):
    """Return dict dataset -> counts used in the report."""
    counts = {}
    for dataset in DATASETS:
        rows = [r for r in table if r["dataset"] == dataset]
        usable = [r for r in rows if r["usable"] == 1]
        counts[dataset] = {
            "files": len(rows),
            "usable": len(usable),
            "dropped": Counter(r["drop_reason"] for r in rows if r["usable"] == 0),
            "pos": sum(1 for r in usable if r["l2"] == "fire_or_smoke"),
            "neg": sum(1 for r in usable if r["l2"] == "none"),
            "clusters": len({r["cluster_id"] for r in usable}),
        }
    return counts


def write_table(table):
    path = OUTPUT_FOLDER / "all_images.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(table)
    return path


def write_report(table, problems, counts, mixed_info):
    n_mixed, n_mixed_images, examples = mixed_info
    lines = []
    lines.append("# Step 3a check: one table of all images")
    lines.append("")
    lines.append("Made by `04_experiments/src/build_image_table.py`. Output: `all_images.csv`.")
    lines.append("")

    lines.append("## Problems found")
    lines.append("")
    if problems:
        lines.append(f"**{len(problems)} problems.** First 20:")
        lines.append("")
        for p in problems[:20]:
            lines.append(f"- {p}")
    else:
        lines.append("None. Every usable image has exactly one cluster, and the labels "
                     "in the manifests and in clusters.csv agree.")
    lines.append("")

    lines.append("## Images per dataset")
    lines.append("")
    lines.append("| ID | Files | Usable | Left out (why) | Fire/smoke | None | Clusters |")
    lines.append("|---|---|---|---|---|---|---|")
    for dataset, c in counts.items():
        dropped = ", ".join(f"{n} {reason}" for reason, n in c["dropped"].items()) or "-"
        lines.append(f"| {dataset} | {c['files']} | {c['usable']} | {dropped} | "
                     f"{c['pos']} | {c['neg']} | {c['clusters']} |")
    lines.append("")
    lines.append("A cluster can hold images of several datasets, so the cluster column "
                 "does not add up to the total number of clusters.")
    lines.append("")

    total_clusters = len({r["cluster_id"] for r in table if r["usable"] == 1})
    total_usable = sum(1 for r in table if r["usable"] == 1)
    lines.append(f"Total: {total_usable} usable images in {total_clusters} clusters.")
    lines.append("")

    lines.append("## Clusters with mixed labels")
    lines.append("")
    lines.append(f"{n_mixed} clusters ({n_mixed_images} images) hold both fire/smoke and "
                 "none images. This is often correct (the same camera before and after a "
                 "fire starts), so it is not an error. It only matters for step 3b: such a "
                 "cluster still goes into one split as a whole.")
    lines.append("")
    lines.append("Smallest examples:")
    lines.append("")
    for e in examples:
        lines.append(f"- {e}")
    lines.append("")

    path = OUTPUT_FOLDER / "step3a_check.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    print("Reading manifests ...")
    manifest_rows = read_manifests()
    print(f"  {len(manifest_rows)} image files")

    print("Reading clusters ...")
    clusters = read_clusters()
    print(f"  {len(clusters)} clustered images")

    print("Joining ...")
    table, problems = build_table(manifest_rows, clusters)

    print("Removing exact copies inside each dataset ...")
    mark_exact_duplicates(table)
    recount_cluster_sizes(table)
    counts = count_per_dataset(table)
    mixed_info = count_mixed_label_clusters(table)

    table_path = write_table(table)
    report_path = write_report(table, problems, counts, mixed_info)

    print(f"  problems found: {len(problems)}")
    for p in problems[:5]:
        print("   ", p)
    print(f"  clusters with mixed labels: {mixed_info[0]} ({mixed_info[1]} images)")
    print(f"Wrote {table_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
