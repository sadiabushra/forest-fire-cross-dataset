"""Step 2: near-duplicate search within and across datasets.

Compares the pHash/dHash fingerprints of all images, applies an extra pixel and
colour test to pairs involving dark images, and groups matches into clusters that
must stay in one split. Writes the matching pairs, the clusters, and the images
shared across datasets.

Usage:
    python 04_experiments/src/find_near_duplicates.py
"""

import csv
import random
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


MAX_PHASH_DISTANCE = 6
MAX_DHASH_DISTANCE = 10

DARK_BRIGHTNESS = 60
DARK_MIN_CORRELATION = 0.88
DARK_MIN_CORRELATION_IF_SAME_PIXELS = 0.74
DARK_MAX_PIXEL_DIFFERENCE = 2.0
DARK_MAX_COLOUR_BALANCE_DIFFERENCE = 0.15

BLOCK_SIZE = 128

NUMBER_OF_CHECK_PICTURES = 40

PROJECT = Path(__file__).resolve().parents[2]
RAW_FOLDER = PROJECT / "03_datasets" / "raw"
MANIFEST_FOLDER = PROJECT / "03_datasets" / "processed" / "manifest"
OUTPUT_FOLDER = PROJECT / "03_datasets" / "processed" / "near_duplicates"


def load_images():
    """Return a list of images (one dict per image) that can be compared."""
    images = []
    left_out = Counter()

    for manifest_file in sorted(MANIFEST_FOLDER.glob("*.csv")):
        with open(manifest_file, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["readable"] != "1":
                    left_out["unreadable"] += 1
                    continue
                if row["subset"] == "patch_50px":
                    left_out["BoWFire 50x50 patch"] += 1
                    continue
                if "near_constant" in row["flags"]:
                    left_out["almost blank image"] += 1
                    continue
                images.append(row)

    print(f"Loaded {len(images)} images. Left out: {dict(left_out)}")
    return images, left_out


def fingerprints_as_numbers(images, column):
    """Turn the hex text fingerprints (e.g. 'a3f0...') into 64-bit numbers."""
    return np.array([int(img[column], 16) for img in images], dtype=np.uint64)


def find_similar_pairs(phash, dhash):
    """Return a list of (i, j, phash_distance, dhash_distance) with i < j."""
    number_of_images = len(phash)
    pairs = []

    for start in range(0, number_of_images, BLOCK_SIZE):
        end = min(start + BLOCK_SIZE, number_of_images)

        phash_distance = np.bitwise_count(phash[start:end, None] ^ phash[None, start:])
        dhash_distance = np.bitwise_count(dhash[start:end, None] ^ dhash[None, start:])

        is_copy = (phash_distance <= MAX_PHASH_DISTANCE) & (dhash_distance <= MAX_DHASH_DISTANCE)
        rows, columns = np.nonzero(is_copy)

        for r, c in zip(rows, columns):
            i = start + r
            j = start + c
            if j > i:
                pairs.append((i, j, int(phash_distance[r, c]), int(dhash_distance[r, c])))

        if (start // BLOCK_SIZE) % 100 == 0:
            print(f"  compared {end}/{number_of_images} images, {len(pairs)} pairs found so far")

    return pairs


def small_colour_picture(rel_path):
    """Open an image and shrink it to 32x32 colour pixels (numbers 0-255)."""
    picture = Image.open(RAW_FOLDER / rel_path)
    picture.draft("RGB", (64, 64))
    picture = picture.convert("RGB").resize((32, 32), Image.Resampling.BOX)
    return np.asarray(picture, dtype=np.float32)


def correlation(a, b):
    """How similar the patterns of two pictures are: 1.0 = same pattern, 0 = unrelated."""
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))


def colour_balance_difference(a, b):
    """How different the overall colour of two pictures is, ignoring brightness. 0 = same colour mix; the largest difference in the red, green or blue share."""
    mean_a = a.reshape(-1, 3).mean(axis=0)
    mean_b = b.reshape(-1, 3).mean(axis=0)
    share_a = mean_a / (mean_a.sum() + 1e-9)
    share_b = mean_b / (mean_b.sum() + 1e-9)
    return float(np.abs(share_a - share_b).max())


def check_dark_pairs(images, pairs):
    """Split the pairs into kept and rejected."""
    brightness = [float(img["mean_lum"]) for img in images]
    dark_pairs = [pair for pair in pairs if min(brightness[pair[0]], brightness[pair[1]]) < DARK_BRIGHTNESS]

    needed = sorted({i for pair in dark_pairs for i in pair[:2]})
    print(f"    {len(dark_pairs)} pairs have a dark image; making small pictures of {len(needed)} images...")
    with ProcessPoolExecutor() as pool:
        pictures = list(pool.map(small_colour_picture, [images[i]["rel_path"] for i in needed], chunksize=64))
    picture_of = dict(zip(needed, pictures))

    scores = {}
    rejected = []
    for i, j, p, d in dark_pairs:
        a, b = picture_of[i], picture_of[j]
        corr = correlation(a, b)
        difference = float(np.abs(a - b).mean())
        colour = colour_balance_difference(a, b)
        scores[(i, j)] = (corr, difference, colour)

        strongly_correlated = corr >= DARK_MIN_CORRELATION
        same_pixels = corr >= DARK_MIN_CORRELATION_IF_SAME_PIXELS and difference <= DARK_MAX_PIXEL_DIFFERENCE
        same_colour = colour <= DARK_MAX_COLOUR_BALANCE_DIFFERENCE
        if not ((strongly_correlated or same_pixels) and same_colour):
            rejected.append((i, j, p, d))

    rejected_set = {(i, j) for i, j, _, _ in rejected}
    kept = [pair for pair in pairs if (pair[0], pair[1]) not in rejected_set]
    return kept, rejected, scores


def make_clusters(number_of_images, pairs):
    """If A~B and B~C, then A, B and C are one cluster."""
    parent = list(range(number_of_images))

    def find_group(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j, _, _ in pairs:
        group_i = find_group(i)
        group_j = find_group(j)
        if group_i != group_j:
            parent[group_j] = group_i

    group_to_number = {}
    cluster_ids = []
    for i in range(number_of_images):
        group = find_group(i)
        if group not in group_to_number:
            group_to_number[group] = len(group_to_number)
        cluster_ids.append(group_to_number[group])
    return cluster_ids


def save_pairs(images, pairs, scores, file_name):
    """Write pairs to a CSV."""
    with open(OUTPUT_FOLDER / file_name, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset_a", "path_a", "label_a", "dataset_b", "path_b", "label_b",
                         "phash_distance", "dhash_distance",
                         "pixel_correlation", "pixel_difference", "colour_balance_difference"])
        for i, j, p, d in pairs:
            a, b = images[i], images[j]
            corr, difference, colour = "", "", ""
            if (i, j) in scores:
                corr = round(scores[(i, j)][0], 3)
                difference = round(scores[(i, j)][1], 2)
                colour = round(scores[(i, j)][2], 3)
            writer.writerow([a["dataset"], a["rel_path"], a["l2"], b["dataset"], b["rel_path"], b["l2"],
                             p, d, corr, difference, colour])


def save_clusters(images, cluster_ids):
    cluster_size = Counter(cluster_ids)
    with open(OUTPUT_FOLDER / "clusters.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "rel_path", "orig_split", "l2", "cluster_id", "cluster_size"])
        for img, cid in zip(images, cluster_ids):
            writer.writerow([img["dataset"], img["rel_path"], img["orig_split"], img["l2"], cid, cluster_size[cid]])


def find_shared_across_datasets(images, cluster_ids):
    """For leave-one-dataset-out: when dataset X is the test set, which images of the OTHER datasets sit in the same cluster as an image of X?"""
    members = defaultdict(list)
    for index, cid in enumerate(cluster_ids):
        members[cid].append(index)

    rows = []
    for cid, indexes in members.items():
        datasets_in_cluster = {images[i]["dataset"] for i in indexes}
        if len(datasets_in_cluster) < 2:
            continue
        for held_out in sorted(datasets_in_cluster):
            for i in indexes:
                if images[i]["dataset"] != held_out:
                    rows.append((held_out, images[i], cid))
    return rows


def save_shared_across_datasets(rows):
    with open(OUTPUT_FOLDER / "shared_across_datasets.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["held_out_dataset", "dataset", "rel_path", "l2", "cluster_id"])
        for held_out, img, cid in rows:
            writer.writerow([held_out, img["dataset"], img["rel_path"], img["l2"], cid])


def save_check_pictures(images, pairs):
    """Save side-by-side pictures of random pairs, so a human can confirm the matches."""
    check_folder = OUTPUT_FOLDER / "check"
    check_folder.mkdir(exist_ok=True)
    for old in check_folder.glob("*.jpg"):
        old.unlink()

    uncertain_pairs = [p for p in pairs if p[2] > 0]
    random.seed(0)
    chosen = random.sample(uncertain_pairs, min(NUMBER_OF_CHECK_PICTURES, len(uncertain_pairs)))

    for number, (i, j, p, d) in enumerate(chosen):
        left = Image.open(RAW_FOLDER / images[i]["rel_path"]).convert("RGB")
        right = Image.open(RAW_FOLDER / images[j]["rel_path"]).convert("RGB")
        left.thumbnail((400, 400))
        right.thumbnail((400, 400))

        sheet = Image.new("RGB", (810, 440), "white")
        sheet.paste(left, (0, 40))
        sheet.paste(right, (410, 40))
        text = (f"{images[i]['dataset']} ({images[i]['l2']})   vs   {images[j]['dataset']} ({images[j]['l2']})"
                f"     pHash dist {p}, dHash dist {d}")
        ImageDraw.Draw(sheet).text((10, 12), text, fill="black")
        sheet.save(check_folder / f"pair_{number:02d}_p{p}_d{d}.jpg", quality=85)


def save_summary(images, pairs, rejected, cluster_ids, left_out, shared_rows):
    datasets = sorted({img["dataset"] for img in images})
    lines = [
        "# Near-duplicate summary (step 2)",
        "",
        f"Two images count as a match when pHash distance <= {MAX_PHASH_DISTANCE} "
        f"and dHash distance <= {MAX_DHASH_DISTANCE} (out of 64 bits).",
        f"If one of the two images is dark (brightness < {DARK_BRIGHTNESS}), the pair must also pass a pixel test: "
        f"correlation of the 32x32 colour pictures >= {DARK_MIN_CORRELATION}, or >= {DARK_MIN_CORRELATION_IF_SAME_PIXELS} "
        f"with an average pixel difference <= {DARK_MAX_PIXEL_DIFFERENCE}; and in both cases a colour balance "
        f"difference <= {DARK_MAX_COLOUR_BALANCE_DIFFERENCE}.",
        "A match is either the same photo (re-saved or resized) or the same fixed camera scene at another time.",
        f"Images compared: {len(images)}. Left out: {dict(left_out)}.",
        f"Pairs found: {len(pairs)} (after removing {len(rejected)} dark pairs that failed the pixel test, "
        "listed in `rejected_dark_pairs.csv`).",
        "Check sheets: `check/` (40 random pairs) and `check_v2/` (per dataset pair, borderline, largest clusters; "
        "made by check_matches.py).",
        "",
    ]

    copies_in = defaultdict(set)
    has_copy_same_dataset = set()
    for i, j, _, _ in pairs:
        a, b = images[i]["dataset"], images[j]["dataset"]
        if a == b:
            has_copy_same_dataset.update([i, j])
        else:
            copies_in[i].add(b)
            copies_in[j].add(a)

    size_of = Counter(img["dataset"] for img in images)

    lines += ["## 1. Matches inside each dataset", "",
              "| Dataset | Images | Images with a match in the same dataset |", "|---|---|---|"]
    for ds in datasets:
        n = sum(1 for i in has_copy_same_dataset if images[i]["dataset"] == ds)
        lines.append(f"| {ds} | {size_of[ds]} | {n} ({n / size_of[ds]:.1%}) |")

    lines += ["", "## 2. Matches across datasets", "",
              "Read a row as: this many images of dataset A have a match in dataset B.",
              "", "| Dataset A | Dataset B | Images of A found in B |", "|---|---|---|"]
    found = Counter()
    for i, others in copies_in.items():
        for other in others:
            found[(images[i]["dataset"], other)] += 1
    for (a, b), n in sorted(found.items(), key=lambda item: -item[1] / size_of[item[0][0]]):
        lines.append(f"| {a} | {b} | {n} ({n / size_of[a]:.1%}) |")

    members = defaultdict(list)
    for index, cid in enumerate(cluster_ids):
        members[cid].append(images[index])
    multi = [m for m in members.values() if len(m) > 1]

    def breaks_publisher_split(cluster):
        splits_per_dataset = defaultdict(set)
        for img in cluster:
            splits_per_dataset[img["dataset"]].add(img["orig_split"])
        return any(len(splits) > 1 for splits in splits_per_dataset.values())

    crosses_split = [m for m in multi if breaks_publisher_split(m)]
    mixed_labels = [m for m in multi if len({img["l2"] for img in m}) > 1]

    lines += ["", "## 3. Clusters", "",
              f"- Clusters with 2 or more images: {len(multi)} (holding {sum(len(m) for m in multi)} images)",
              f"- Clusters that sit in both the train and test part of a publisher's split: {len(crosses_split)}",
              f"- Clusters whose images have different L2 labels (fire_or_smoke vs none): {len(mixed_labels)}"
              " (normal for a fixed camera: the same scene before and after smoke appears)",
              "", "Largest clusters (usually one fixed camera or one video; check that they are not false matches):", "",
              "| Size | Datasets | Example image |", "|---|---|---|"]
    for m in sorted(multi, key=len, reverse=True)[:10]:
        where = ", ".join(f"{ds} {n}" for ds, n in Counter(img["dataset"] for img in m).most_common())
        lines.append(f"| {len(m)} | {where} | `{m[0]['rel_path']}` |")

    removed = Counter()
    for held_out, img, _ in shared_rows:
        removed[(held_out, img["dataset"])] += 1
    lines += ["", "## 4. Leave-one-dataset-out: images to remove from training", "",
              "When the dataset in the first column is the test set, these images of the other datasets share a "
              "cluster with a test image and must NOT be used for training. The full list is "
              "`shared_across_datasets.csv`.", "",
              "| Test (held-out) dataset | Remove from this training dataset | Images |", "|---|---|---|"]
    for (held_out, other), n in sorted(removed.items()):
        lines.append(f"| {held_out} | {other} | {n} ({n / size_of[other]:.1%}) |")

    (OUTPUT_FOLDER / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    print("1/5 Loading manifests...")
    images, left_out = load_images()
    phash = fingerprints_as_numbers(images, "phash")
    dhash = fingerprints_as_numbers(images, "dhash")

    print("2/5 Comparing fingerprints (the slow part)...")
    pairs = find_similar_pairs(phash, dhash)
    print(f"    {len(pairs)} similar pairs found")

    print("3/5 Second test for pairs with a dark image...")
    pairs, rejected, scores = check_dark_pairs(images, pairs)
    print(f"    {len(rejected)} dark pairs rejected, {len(pairs)} pairs kept")

    print("4/5 Building clusters...")
    cluster_ids = make_clusters(len(images), pairs)
    shared_rows = find_shared_across_datasets(images, cluster_ids)

    print("5/5 Saving results...")
    save_pairs(images, pairs, scores, "pairs.csv")
    save_pairs(images, rejected, scores, "rejected_dark_pairs.csv")
    save_clusters(images, cluster_ids)
    save_shared_across_datasets(shared_rows)
    save_summary(images, pairs, rejected, cluster_ids, left_out, shared_rows)
    save_check_pictures(images, pairs)
    print(f"Done. Open {OUTPUT_FOLDER / 'summary.md'}")


if __name__ == "__main__":
    main()
