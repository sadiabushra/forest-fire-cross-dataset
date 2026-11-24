"""Synthetic step 1: cut fire and smoke pieces from the boxes of training images.

Uses the training images of R2, R3 and R4 (the datasets with boxes), filters
unusable boxes, samples at most 3,000 pieces per dataset and kind, and writes
the pieces, pieces.csv and contact sheets to 03_datasets/processed/synthetic/.

Usage:
    python 04_experiments/src/cut_fire_pieces.py [--test]
"""

import argparse
import csv
import random
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageOps


THESIS = Path(__file__).resolve().parents[2]
RAW = THESIS / "03_datasets" / "raw"
SPLIT_FILE = THESIS / "03_datasets" / "processed" / "splits" / "within_dataset.csv"
MANIFEST_FOLDER = THESIS / "03_datasets" / "processed" / "manifest"
OUT = THESIS / "03_datasets" / "processed" / "synthetic"

SOURCE_DATASETS = ["R2", "R3", "R4"]

DFIRE_CLASSES = {"0": "smoke", "1": "fire"}

MIN_SIDE = 32
MAX_AREA_SHARE = 0.6
MAX_ASPECT = 6.0

MAX_PER_GROUP = 3000
MAX_PIECE_SIDE = 256
JPEG_QUALITY = 92

RANDOM_SEED = 0
SHEET_PIECES = 48
SHEET_TILE = 128


def read_options():
    parser = argparse.ArgumentParser(description="Cut fire and smoke pieces from training images")
    parser.add_argument("--test", action="store_true", help="only 50 pieces per group, written to synthetic_test/")
    return parser.parse_args()


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_voc_boxes(path):
    """FASDD: list of (kind, xmin, ymin, xmax, ymax) in pixels."""
    boxes = []
    root = ET.parse(path).getroot()
    for obj in root.iter("object"):
        name = obj.findtext("name", "").strip().lower()
        if name not in ("fire", "smoke"):
            continue
        b = obj.find("bndbox")
        boxes.append((name, float(b.findtext("xmin")), float(b.findtext("ymin")),
                      float(b.findtext("xmax")), float(b.findtext("ymax"))))
    return boxes


def read_yolo_boxes(path, width, height):
    """D-Fire: YOLO lines 'class x_centre y_centre w h' (0-1), turned into pixels."""
    boxes = []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) != 5 or parts[0] not in DFIRE_CLASSES:
            continue
        xc, yc, w, h = (float(p) for p in parts[1:])
        boxes.append((DFIRE_CLASSES[parts[0]],
                      (xc - w / 2) * width, (yc - h / 2) * height,
                      (xc + w / 2) * width, (yc + h / 2) * height))
    return boxes


def usable(box, width, height):
    """Is this box big enough, not too big, and not too thin?"""
    _, x0, y0, x1, y1 = box
    w = x1 - x0
    h = y1 - y0
    if w < MIN_SIDE or h < MIN_SIDE:
        return False
    if w * h > MAX_AREA_SHARE * width * height:
        return False
    if max(w, h) / min(w, h) > MAX_ASPECT:
        return False
    return True


def collect_candidates():
    manifest = {}
    for dataset in SOURCE_DATASETS:
        for row in read_csv(MANIFEST_FOLDER / f"{dataset}.csv"):
            manifest[row["rel_path"]] = row

    candidates = []
    for row in read_csv(SPLIT_FILE):
        if row["dataset"] not in SOURCE_DATASETS:
            continue
        if row["split"] != "train" or row["l2"] != "fire_or_smoke":
            continue
        m = manifest.get(row["rel_path"])
        if m is None or m["ann_status"] != "ok" or not m["width"]:
            continue
        width = int(m["width"])
        height = int(m["height"])
        ann = RAW / m["ann_path"]
        if row["dataset"] == "R4":
            boxes = read_yolo_boxes(ann, width, height)
        else:
            boxes = read_voc_boxes(ann)
        for box in boxes:
            if usable(box, width, height):
                candidates.append({"dataset": row["dataset"], "rel_path": row["rel_path"],
                                   "cluster_id": row["cluster_id"], "kind": box[0],
                                   "box": [round(v) for v in box[1:]]})
    return candidates


def cut_pieces(candidates, per_group, out_folder):
    rng = random.Random(RANDOM_SEED)
    groups = {}
    for c in candidates:
        groups.setdefault((c["dataset"], c["kind"]), []).append(c)
    chosen = []
    for key in sorted(groups):
        items = groups[key]
        rng.shuffle(items)
        chosen.extend(items[:per_group])

    rows = []
    for number, c in enumerate(chosen, 1):
        try:
            with Image.open(RAW / c["rel_path"]) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                piece = image.crop(tuple(c["box"]))
        except OSError as error:
            print(f"skipped {c['rel_path']}: {error}")
            continue
        piece.thumbnail((MAX_PIECE_SIDE, MAX_PIECE_SIDE))
        name = f"{c['kind']}_{c['dataset']}_{number:05d}.jpg"
        path = out_folder / "pieces" / c["kind"] / name
        path.parent.mkdir(parents=True, exist_ok=True)
        piece.save(path, quality=JPEG_QUALITY)
        rows.append({"piece": f"pieces/{c['kind']}/{name}", "kind": c["kind"], "dataset": c["dataset"],
                     "source_image": c["rel_path"], "cluster_id": c["cluster_id"],
                     "box": " ".join(map(str, c["box"])), "width": piece.width, "height": piece.height})
        if number % 500 == 0:
            print(f"  {number}/{len(chosen)} pieces")
    return rows


def contact_sheet(rows, kind, path):
    rng = random.Random(RANDOM_SEED)
    pick = [r for r in rows if r["kind"] == kind]
    rng.shuffle(pick)
    pick = pick[:SHEET_PIECES]
    per_row = 8
    sheet = Image.new("RGB", (per_row * SHEET_TILE, ((len(pick) + per_row - 1) // per_row) * SHEET_TILE), "white")
    for k, r in enumerate(pick):
        with Image.open(path.parent / r["piece"]) as piece:
            piece = piece.copy()
        piece.thumbnail((SHEET_TILE - 4, SHEET_TILE - 4))
        sheet.paste(piece, ((k % per_row) * SHEET_TILE + 2, (k // per_row) * SHEET_TILE + 2))
    sheet.save(path, quality=85)


def main():
    options = read_options()
    per_group = 50 if options.test else MAX_PER_GROUP
    out_folder = OUT.parent / "synthetic_test" if options.test else OUT

    candidates = collect_candidates()
    print(f"usable boxes in training images: {len(candidates)}")
    rows = cut_pieces(candidates, per_group, out_folder)

    out_folder.mkdir(parents=True, exist_ok=True)
    with open(out_folder / "pieces.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    for kind in ("fire", "smoke"):
        contact_sheet(rows, kind, out_folder / f"check_{kind}.jpg")

    lines = ["# Synthetic step 1: fire and smoke pieces", "",
             f"Usable boxes in training images of {', '.join(SOURCE_DATASETS)}: {len(candidates)}",
             f"Filters: each side >= {MIN_SIDE} px, area <= {MAX_AREA_SHARE:.0%} of the image, "
             f"aspect ratio <= {MAX_ASPECT}. At most {per_group} pieces per dataset and kind.", "",
             "| Dataset | Kind | Usable boxes | Pieces saved |", "|---|---|---|---|"]
    for dataset in SOURCE_DATASETS:
        for kind in ("fire", "smoke"):
            usable_count = sum(1 for c in candidates if c["dataset"] == dataset and c["kind"] == kind)
            saved = sum(1 for r in rows if r["dataset"] == dataset and r["kind"] == kind)
            lines.append(f"| {dataset} | {kind} | {usable_count} | {saved} |")
    (out_folder / "step1_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"written: {out_folder}")


if __name__ == "__main__":
    main()
