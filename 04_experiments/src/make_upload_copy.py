"""Step 4: resized copy of the dataset for GPU training.

Resizes every usable image to a short side of 256 px (JPEG), copies the split
files with an image-path column, and packs everything into one zip.

Usage:
    python 04_experiments/src/make_upload_copy.py [--test]
"""

import csv
import sys
import time
import zipfile
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

from PIL import Image, ImageOps


SHORT_SIDE = 256

JPEG_QUALITY = 90

SKIP_DATASETS = {"ALT_DEEPFIRE"}

TEST_IMAGES = 200

WORKERS = 15

PROJECT = Path(__file__).resolve().parents[2]
RAW_FOLDER = PROJECT / "03_datasets" / "raw"
SPLITS_FOLDER = PROJECT / "03_datasets" / "processed" / "splits"
UPLOAD_FOLDER = PROJECT / "03_datasets" / "processed" / "upload"
TEST_FOLDER = PROJECT / "03_datasets" / "processed" / "upload_test"


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def small_image_path(rel_path):
    """Path of the small copy inside the upload folder, e.g. images/R4_D-Fire/.../x.jpg"""
    return "images/" + str(Path(rel_path).with_suffix(".jpg").as_posix())


def check_no_name_clashes(rows):
    """Two raw files like a.png and a.jpg would both become a.jpg."""
    seen = {}
    for row in rows:
        new = small_image_path(row["rel_path"])
        if new in seen:
            raise ValueError(f"Two images would get the same name {new}: "
                             f"{seen[new]} and {row['rel_path']}")
        seen[new] = row["rel_path"]


def resize_one(job):
    """job = (raw path, output path)."""
    raw_path, out_path = job
    out_path = Path(out_path)
    if out_path.exists():
        return "skipped", raw_path, ""
    try:
        with Image.open(raw_path) as img:
            target_w, target_h = target_size(img.size)
            img.draft("RGB", (target_w, target_h))
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
            img = img.resize(target_size(img.size), Image.Resampling.LANCZOS)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = out_path.with_suffix(".tmp")
            img.save(temp_path, "JPEG", quality=JPEG_QUALITY)
            temp_path.replace(out_path)
        return "done", raw_path, ""
    except Exception as error:
        return "failed", raw_path, f"{type(error).__name__}: {error}"


def target_size(size):
    """New (width, height) with the short side at SHORT_SIDE."""
    width, height = size
    short = min(width, height)
    if short <= SHORT_SIDE:
        return width, height
    scale = SHORT_SIDE / short
    return max(1, round(width * scale)), max(1, round(height * scale))


def resize_all(rows, out_folder):
    """Resize every image with several worker processes."""
    jobs = []
    for row in rows:
        raw = RAW_FOLDER / row["rel_path"]
        out = out_folder / small_image_path(row["rel_path"])
        jobs.append((str(raw), str(out)))

    status = Counter()
    failures = []
    start = time.time()
    with Pool(WORKERS) as pool:
        for i, (result, raw_path, message) in enumerate(pool.imap_unordered(resize_one, jobs, chunksize=32), 1):
            status[result] += 1
            if result == "failed":
                failures.append((raw_path, message))
            if i % 5000 == 0 or i == len(jobs):
                minutes = (time.time() - start) / 60
                print(f"  {i}/{len(jobs)} images, {minutes:.1f} min", flush=True)
    return status, failures


def write_split_files(out_folder, copied_paths):
    """Copy within_dataset.csv and the fold CSVs, adding the 'image' column."""
    split_files = [SPLITS_FOLDER / "within_dataset.csv"] + sorted((SPLITS_FOLDER / "folds").glob("*.csv"))
    (out_folder / "splits").mkdir(parents=True, exist_ok=True)
    written = []
    for path in split_files:
        rows = read_csv(path)
        rows = [r for r in rows if r["rel_path"] in copied_paths]
        if not rows:
            continue
        for row in rows:
            row["image"] = small_image_path(row["rel_path"])
        columns = ["image"] + [c for c in rows[0] if c != "image"]
        out_path = out_folder / "splits" / path.name
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        written.append((path.name, len(rows)))
    return written


def write_readme(out_folder):
    text = f"""Forest fire thesis: small copy of the images, for training on a GPU machine.
Made by 04_experiments/src/make_upload_copy.py.

images/   every usable image, short side {SHORT_SIDE} px, JPEG quality {JPEG_QUALITY},
          same folder structure as the raw datasets.
splits/   within_dataset.csv  -> column "split" = train / val / test  (within-dataset baselines)
          ref_to_shift.csv, lodo_R1..R4.csv -> column "role" = train / val / test  (cross-dataset)
          The column "image" is the path of the image, relative to this folder.
          Labels: l2 = fire_or_smoke / none (all datasets); l4 = fire / smoke / fire_and_smoke / none
          (empty where the dataset has no 4-class labels).

Exact duplicates, unreadable files and BoWFire patches are already removed.
Leakage between train/val and test is already removed (see step 3 check files).
"""
    (out_folder / "README.txt").write_text(text, encoding="utf-8")


def make_zip(out_folder):
    zip_path = out_folder.parent / (out_folder.name + ".zip")
    print(f"Packing {zip_path} ...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:
        for path in sorted(out_folder.rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(out_folder).as_posix())
    return zip_path


def folder_size_gb(folder):
    return sum(p.stat().st_size for p in folder.rglob("*") if p.is_file()) / 1e9


def write_report(out_folder, rows, status, failures, written, minutes):
    per_dataset = Counter(r["dataset"] for r in rows)
    lines = ["# Step 4 check: small copy for upload", ""]
    lines.append(f"Short side {SHORT_SIDE} px, JPEG quality {JPEG_QUALITY}. Took {minutes:.1f} minutes.")
    lines.append("")
    lines.append(f"Images: {status['done']} made now, {status['skipped']} already there, "
                 f"**{status['failed']} failed**.")
    lines.append(f"Folder size: {folder_size_gb(out_folder):.2f} GB.")
    lines.append("")
    lines.append("| Dataset | Images |")
    lines.append("|---|---|")
    for dataset, n in sorted(per_dataset.items()):
        lines.append(f"| {dataset} | {n} |")
    lines.append("")
    lines.append("Split files written:")
    lines.append("")
    for name, n in written:
        lines.append(f"- `splits/{name}`: {n} rows")
    lines.append("")
    if failures:
        lines.append("## Failures")
        lines.append("")
        for raw_path, message in failures[:50]:
            lines.append(f"- `{raw_path}`: {message}")
        lines.append("")
    (out_folder / "step4_check.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    test_run = "--test" in sys.argv
    out_folder = TEST_FOLDER if test_run else UPLOAD_FOLDER

    rows = read_csv(SPLITS_FOLDER / "within_dataset.csv")
    rows = [r for r in rows if r["dataset"] not in SKIP_DATASETS]
    check_no_name_clashes(rows)

    if test_run:
        rows = rows[::max(1, len(rows) // TEST_IMAGES)][:TEST_IMAGES]
        print(f"TEST RUN: {len(rows)} images into {out_folder}")
    else:
        print(f"{len(rows)} images into {out_folder}")

    start = time.time()
    status, failures = resize_all(rows, out_folder)
    copied_paths = {r["rel_path"] for r in rows}
    written = write_split_files(out_folder, copied_paths)
    write_readme(out_folder)
    minutes = (time.time() - start) / 60
    write_report(out_folder, rows, status, failures, written, minutes)

    print(f"  made {status['done']}, already there {status['skipped']}, failed {status['failed']}")
    print(f"  folder size {folder_size_gb(out_folder):.2f} GB")
    if not test_run:
        zip_path = make_zip(out_folder)
        print(f"Wrote {zip_path} ({zip_path.stat().st_size / 1e9:.2f} GB)")
    print(f"Wrote {out_folder / 'step4_check.md'}")


if __name__ == "__main__":
    main()
