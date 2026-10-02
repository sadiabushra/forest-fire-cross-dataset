"""Step 2b: contact sheets for visually verifying near-duplicate matches.

Draws sampled pairs for every pair of datasets, borderline pairs near the distance
thresholds, and members of the largest clusters.

Usage:
    python 04_experiments/src/check_matches.py
"""

import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw


PAIRS_PER_DATASET_PAIR = 12

BORDERLINE_MIN_PHASH = 5
BORDERLINE_MIN_DHASH = 9
NUMBER_OF_BORDERLINE_PAIRS = 36

NUMBER_OF_LARGE_CLUSTERS = 12
MEMBERS_PER_CLUSTER = 24

THUMB = 200

RANDOM_SEED = 0

PROJECT = Path(__file__).resolve().parents[2]
RAW_FOLDER = PROJECT / "03_datasets" / "raw"
STEP2_FOLDER = PROJECT / "03_datasets" / "processed" / "near_duplicates"
OUTPUT_FOLDER = STEP2_FOLDER / "check_v2"


def add_to_sample(sample, seen_so_far, item, sample_size):
    """Keep a fair random sample of a long list without holding the whole list in memory. ("Reservoir sampling": in the end every item had the same chance to be in the sample.) seen_so_far = how many items have been offered so far, including this one."""
    if len(sample) < sample_size:
        sample.append(item)
    else:
        position = random.randint(0, seen_so_far - 1)
        if position < sample_size:
            sample[position] = item


def sample_pairs():
    """Read pairs.csv once."""
    samples = defaultdict(list)
    all_pairs = Counter()
    not_identical = Counter()
    borderline = []
    borderline_count = 0

    with open(STEP2_FOLDER / "pairs.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            p = int(row["phash_distance"])
            d = int(row["dhash_distance"])

            first = (row["dataset_a"], row["path_a"], row["label_a"])
            second = (row["dataset_b"], row["path_b"], row["label_b"])
            if first[0] > second[0]:
                first, second = second, first
            key = (first[0], second[0])
            pair = (first, second, p, d)

            all_pairs[key] += 1
            if p == 0 and d == 0:
                continue

            not_identical[key] += 1
            add_to_sample(samples[key], not_identical[key], pair, PAIRS_PER_DATASET_PAIR)

            if p >= BORDERLINE_MIN_PHASH and d >= BORDERLINE_MIN_DHASH:
                borderline_count += 1
                add_to_sample(borderline, borderline_count, pair, NUMBER_OF_BORDERLINE_PAIRS)

    return samples, all_pairs, not_identical, borderline, borderline_count


def thumbnail(rel_path):
    """Open one image and shrink it to fit a THUMB x THUMB box."""
    picture = Image.open(RAW_FOLDER / rel_path).convert("RGB")
    picture.thumbnail((THUMB, THUMB))
    return picture


def draw_pair_sheet(pairs, title, file_name):
    """Draw up to 12 pairs: 3 columns x 4 rows, each pair = two pictures side by side."""
    columns = 3
    rows = (len(pairs) + columns - 1) // columns
    cell_width = 2 * THUMB + 10
    cell_height = THUMB + 40
    sheet = Image.new("RGB", (columns * (cell_width + 20), 40 + rows * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 12), title, fill="black")

    for number, (a, b, p, d) in enumerate(pairs):
        x = (number % columns) * (cell_width + 20)
        y = 40 + (number // columns) * cell_height
        sheet.paste(thumbnail(a[1]), (x, y + 20))
        sheet.paste(thumbnail(b[1]), (x + THUMB + 10, y + 20))
        text = f"#{number}  {a[0]} {a[2][:4]} | {b[0]} {b[2][:4]}  p{p} d{d}"
        draw.text((x, y + 4), text, fill="black")

    sheet.save(OUTPUT_FOLDER / file_name, quality=85)


def draw_cluster_sheet(members, title, file_name):
    """Draw random members of one cluster in a grid of 6 columns."""
    columns = 6
    rows = (len(members) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * (THUMB + 10), 40 + rows * (THUMB + 25)), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 12), title, fill="black")

    for number, member in enumerate(members):
        x = (number % columns) * (THUMB + 10)
        y = 40 + (number // columns) * (THUMB + 25)
        sheet.paste(thumbnail(member["rel_path"]), (x, y + 15))
        draw.text((x, y), f"{member['dataset']} {member['l2'][:4]}", fill="black")

    sheet.save(OUTPUT_FOLDER / file_name, quality=85)


def load_clusters():
    """Return cluster id -> list of member rows from clusters.csv."""
    members = defaultdict(list)
    with open(STEP2_FOLDER / "clusters.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            members[row["cluster_id"]].append(row)
    return members


def pick_cluster_members(cluster):
    """Pick random members, but make sure every dataset in the cluster is shown at least twice (if possible)."""
    by_dataset = defaultdict(list)
    for member in cluster:
        by_dataset[member["dataset"]].append(member)

    chosen = []
    for dataset_members in by_dataset.values():
        chosen += random.sample(dataset_members, min(2, len(dataset_members)))
    rest = [m for m in cluster if m not in chosen]
    chosen += random.sample(rest, min(MEMBERS_PER_CLUSTER - len(chosen), len(rest)))
    return chosen


def main():
    random.seed(RANDOM_SEED)
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    for old in OUTPUT_FOLDER.glob("*.jpg"):
        old.unlink()

    print("1/3 Reading pairs.csv and sampling (about 1 minute)...")
    samples, all_pairs, not_identical, borderline, borderline_count = sample_pairs()

    index = [
        "# Step 2b check sheets",
        "",
        "Each `pairs_A_vs_B.jpg` shows random pairs that are NOT byte-identical.",
        "Under each pair: dataset and label (`fire` = fire_or_smoke, `none` = none), then the distances.",
        "",
        "| Sheet | All pairs | Not byte-identical | Shown |",
        "|---|---|---|---|",
    ]

    print("2/3 Drawing pair sheets...")
    for key in sorted(all_pairs):
        a, b = key
        name = f"pairs_{a}_vs_{b}.jpg"
        shown = len(samples[key])
        if shown > 0:
            draw_pair_sheet(samples[key], f"{a} vs {b}: {shown} random of {not_identical[key]} non-identical pairs", name)
        index.append(f"| {name if shown else '(none: all identical)'} | {all_pairs[key]} | {not_identical[key]} | {shown} |")

    for start in range(0, len(borderline), 12):
        number = start // 12
        draw_pair_sheet(borderline[start:start + 12],
                        f"Borderline pairs (pHash >= {BORDERLINE_MIN_PHASH}, dHash >= {BORDERLINE_MIN_DHASH}), sheet {number}",
                        f"borderline_{number}.jpg")
    index += ["", f"Borderline pairs: {borderline_count} exist, {len(borderline)} shown in `borderline_*.jpg`.", ""]

    print("3/3 Drawing the largest clusters...")
    clusters = load_clusters()
    largest = sorted(clusters.values(), key=len, reverse=True)[:NUMBER_OF_LARGE_CLUSTERS]
    index += ["| Sheet | Cluster id | Size | Datasets |", "|---|---|---|---|"]
    for rank, cluster in enumerate(largest):
        name = f"cluster_{rank:02d}_size{len(cluster)}.jpg"
        where = ", ".join(f"{ds} {n}" for ds, n in Counter(m["dataset"] for m in cluster).most_common())
        draw_cluster_sheet(pick_cluster_members(cluster), f"Cluster {cluster[0]['cluster_id']}, size {len(cluster)}: {where}", name)
        index.append(f"| {name} | {cluster[0]['cluster_id']} | {len(cluster)} | {where} |")

    (OUTPUT_FOLDER / "index.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    print(f"Done. Open {OUTPUT_FOLDER / 'index.md'}")


if __name__ == "__main__":
    main()
