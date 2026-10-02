"""Step 1: inventory of the raw datasets.

Opens every image, maps native labels to the shared L2/L4 labels, checks folder
labels against annotations, and records size, SHA-256, pHash/dHash and image
statistics. Writes one manifest CSV per dataset and an inventory summary.

Usage:
    python 04_experiments/src/inventory.py [DATASET_ID ...] [--force] [--summary-only]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import re
import statistics
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "03_datasets" / "raw"
OUT = ROOT / "03_datasets" / "processed" / "manifest"
SUMMARY = ROOT / "03_datasets" / "processed" / "inventory_summary.md"

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"}
POS, NEG = "fire_or_smoke", "none"

FIELDS = [
    "dataset", "rel_path", "orig_split", "subset", "native_label", "l2", "l4",
    "has_fire", "has_smoke", "ann_path", "ann_status", "boxes", "mask_frac",
    "label_conflict", "readable", "error", "bytes", "sha256", "format", "mode",
    "width", "height", "mean_lum", "std_lum", "lap_var", "phash", "dhash", "flags",
]

DATASETS = {
    "R1": "Baseline Forest Fire Images",
    "R2": "FASDD_CV",
    "R3": "FASDD_UAV",
    "R4": "D-Fire",
    "S1": "BoWFire",
    "S2": "FESB MLID",
    "S3": "UAVs-FFDB (raw images only)",
    "ALT_DEEPFIRE": "DeepFire",
    "ALT_WILDFIRE": "The Wildfire Dataset (v2)",
}

PUBLISHED = {
    "R1": 5050, "R2": 95314, "R3": 25097, "R4": 21527, "S1": 226, "S2": 400,
    "S3": 1653, "ALT_DEEPFIRE": 1900, "ALT_WILDFIRE": 2700,
}

skipped_files: dict[str, Counter] = defaultdict(Counter)


def l4_of(fire: int, smoke: int) -> str:
    return {(1, 1): "fire_and_smoke", (1, 0): "fire", (0, 1): "smoke", (0, 0): "none"}[(fire, smoke)]


def rel(p: Path) -> str:
    return p.relative_to(RAW).as_posix()


def images(ds: str, folder: Path):
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            yield p
        elif p.is_file():
            skipped_files[ds][p.suffix.lower() or p.name] += 1


def job(ds, p, split="", subset="", native="", l2="", l4="", has_fire="", has_smoke="",
        ann=None, kind="", labels_from_ann=False):
    return {
        "dataset": ds, "rel_path": rel(p), "orig_split": split, "subset": subset,
        "native_label": native, "l2": l2, "l4": l4,
        "has_fire": has_fire, "has_smoke": has_smoke,
        "ann_path": rel(ann) if ann is not None else "", "_kind": kind,
        "_labels_from_ann": labels_from_ann,
    }


def adapt_r1():
    base = RAW / "R1_baseline_forest_fire" / "Data"
    for split_dir, split in (("Train_Data", "train"), ("Test_Data", "test")):
        for cls, l2 in (("Fire", POS), ("Non_Fire", NEG)):
            for p in images("R1", base / split_dir / cls):
                yield job("R1", p, split, native=cls, l2=l2,
                          has_fire="" if l2 == POS else 0, has_smoke="" if l2 == POS else 0)


FASDD_PREFIX = {
    "fire": (1, 0), "smoke": (0, 1), "bothFireAndSmoke": (1, 1), "neitherFireNorSmoke": (0, 0),
}


def adapt_fasdd(ds, sub):
    base = RAW / "R2_R3_FASDD" / f"FASDD_{sub}"
    split_of = {}
    for s in ("train", "val", "test"):
        for line in (base / "annotations" / f"YOLO_{sub}" / f"{s}.txt").read_text().splitlines():
            name = line.strip().replace("\\", "/").split("/")[-1]
            if name:
                split_of[name] = s
    for p in images(ds, base / "images"):
        native = re.match(rf"(.+)_{sub}\d+$", p.stem).group(1)
        f, s = FASDD_PREFIX[native]
        ann = base / "annotations" / f"VOC_{sub}" / "Annotations" / f"{p.stem}.xml"
        yield job(ds, p, split_of.get(p.name, ""), native=native, l2=POS if f or s else NEG,
                  l4=l4_of(f, s), has_fire=f, has_smoke=s, ann=ann, kind="voc")


def adapt_r4():
    base = RAW / "R4_D-Fire"
    for split in ("train", "test"):
        for p in images("R4", base / split / "images"):
            yield job("R4", p, split, ann=base / split / "labels" / f"{p.stem}.txt",
                      kind="yolo_dfire", labels_from_ann=True)


def adapt_s1():
    base = RAW / "S1_BoWFire" / "BoWFireDataset"
    for p in images("S1", base / "dataset" / "img"):
        fire = 0 if p.stem.startswith("not_fire") else 1
        yield job("S1", p, "test", "full_image", native="fire" if fire else "not_fire",
                  l2=POS if fire else NEG, has_fire=fire,
                  ann=base / "dataset" / "gt" / f"{p.stem}_gt.png", kind="bowfire_gt")
    for p in images("S1", base / "train"):
        native = re.match(r"[a-z]+", p.stem).group(0)
        yield job("S1", p, "train", "patch_50px", native=native,
                  l2=POS if native in ("fire", "smoke") else NEG,
                  has_fire=int(native == "fire"), has_smoke=int(native == "smoke"))


def adapt_s2():
    base = RAW / "S2_FESB_MLID" / "FESB MLID"
    for split_dir, split in (("train_images", "train"), ("test_images", "test")):
        for p in images("S2", base / split_dir / "src"):
            gt = base / split_dir / "gt" / f"{p.stem.replace('src_', 'gt_', 1)}.bmp"
            yield job("S2", p, split, ann=gt, kind="fesb_gt", labels_from_ann=True)


UAV_CLASSES = {
    "Pre-Evening Fire Incident_raw_img": ("pre_evening_fire_incident", POS, "Pre-evening fire incident_raw_annot"),
    "Evening Fire Incident_raw_img": ("evening_fire_incident", POS, "Evening fire incident_raw_annot"),
    "Pre-evening Forest condition_raw_img": ("pre_evening_forest_condition", NEG, None),
    "Evening Forest condition_raw_img": ("evening_forest_condition", NEG, None),
}


def adapt_s3():
    base = RAW / "S3_UAVs-FFDB" / "UAVS-FDDB UAVs-based Forest Fire Detection Database"
    img_root = base / "Original Image Dataset (Raw Images)"
    ann_root = base / "Annotation Files" / "Raw image annotation"
    for folder, (native, l2, ann_folder) in UAV_CLASSES.items():
        for p in images("S3", img_root / folder):
            if ann_folder:
                yield job("S3", p, "", "raw", native=native, l2=l2,
                          ann=ann_root / ann_folder / f"{p.stem}.xml", kind="voc", labels_from_ann=True)
            else:
                yield job("S3", p, "", "raw", native=native, l2=l2, l4="none", has_fire=0, has_smoke=0)


def adapt_deepfire():
    base = RAW / "alt_DeepFire" / "Dataset"
    for split_dir, split in (("Training and Validation", "trainval"), ("Testing", "test")):
        for cls, l2 in (("fire", POS), ("nofire", NEG)):
            for p in images("ALT_DEEPFIRE", base / split_dir / cls):
                yield job("ALT_DEEPFIRE", p, split, native=cls, l2=l2,
                          has_fire="" if l2 == POS else 0, has_smoke="" if l2 == POS else 0)


def adapt_wildfire():
    base = RAW / "alt_The_Wildfire_Dataset" / "the_wildfire_dataset_2n_version"
    for split in ("train", "val", "test"):
        for cls, l2 in (("fire", POS), ("nofire", NEG)):
            for p in images("ALT_WILDFIRE", base / split / cls):
                yield job("ALT_WILDFIRE", p, split, native=cls, l2=l2,
                          has_fire="" if l2 == POS else 0)


ADAPTERS = {
    "R1": adapt_r1,
    "R2": lambda: adapt_fasdd("R2", "CV"),
    "R3": lambda: adapt_fasdd("R3", "UAV"),
    "R4": adapt_r4,
    "S1": adapt_s1,
    "S2": adapt_s2,
    "S3": adapt_s3,
    "ALT_DEEPFIRE": adapt_deepfire,
    "ALT_WILDFIRE": adapt_wildfire,
}


def _dct_matrix(n: int) -> np.ndarray:
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    m = np.cos(np.pi * (2 * i + 1) * k / (2 * n)) * np.sqrt(2 / n)
    m[0] /= np.sqrt(2)
    return m


DCT32 = _dct_matrix(32)
FESB_CLASSES = {
    0: "unknown", 1: "smoke", 2: "clouds_fog", 3: "sun", 4: "sky", 5: "sea",
    6: "distant_landscape", 7: "rocks", 8: "distant_vegetation", 9: "close_vegetation",
    10: "low_vegetation", 11: "buildings",
}


def _bits_hex(bits: np.ndarray) -> str:
    return np.packbits(bits.flatten()).tobytes().hex()


def phash(gray: Image.Image) -> str:
    a = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS, reducing_gap=3.0), dtype=np.float64)
    low = (DCT32 @ a @ DCT32.T)[:8, :8]
    return _bits_hex(low > np.median(low))


def dhash(gray: Image.Image) -> str:
    a = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS, reducing_gap=3.0), dtype=np.int16)
    return _bits_hex(a[:, 1:] > a[:, :-1])


def _read_annotation(rec: dict, kind: str, labels_from_ann: bool) -> None:
    path = RAW / rec["ann_path"]
    if not path.is_file():
        rec["ann_status"] = "missing"
        if not labels_from_ann and rec["l4"] not in ("", "none"):
            rec["label_conflict"] = "positive label but annotation file missing"
        return
    rec["ann_status"] = "ok"

    if kind in ("voc", "yolo_dfire"):
        if kind == "voc":
            names = [o.findtext("name", "").strip().lower() for o in ET.parse(path).getroot().iter("object")]
        else:
            ids = [ln.split()[0] for ln in path.read_text().splitlines() if ln.strip()]
            names = ["smoke" if i == "0" else "fire" if i == "1" else f"class{i}" for i in ids]
        counts = Counter(names)
        rec["boxes"] = ";".join(f"{k}:{v}" for k, v in sorted(counts.items()))
        f = int(any("fire" in n for n in names))
        s = int(any("smoke" in n for n in names))
        if labels_from_ann:
            rec["has_fire"], rec["has_smoke"], rec["l4"] = f, s, l4_of(f, s)
            if not rec["l2"]:
                rec["l2"] = POS if f or s else NEG
            if not rec["native_label"]:
                rec["native_label"] = rec["l4"]
            if rec["l2"] == POS and not (f or s):
                rec["label_conflict"] = "positive class but no fire/smoke boxes"
        elif (f, s) != (rec["has_fire"], rec["has_smoke"]):
            rec["label_conflict"] = f"label says fire={rec['has_fire']} smoke={rec['has_smoke']}; boxes say fire={f} smoke={s}"

    elif kind == "bowfire_gt":
        with Image.open(path) as m:
            frac = float((np.asarray(m.convert("L")) > 127).mean())
        rec["mask_frac"] = round(frac, 5)
        if rec["has_fire"] == 1 and frac == 0:
            rec["label_conflict"] = "fire image with empty fire mask"
        elif rec["has_fire"] == 0 and frac > 0:
            rec["label_conflict"] = "non-fire image with non-empty fire mask"

    elif kind == "fesb_gt":
        with Image.open(path) as m:
            a = np.asarray(m.convert("L")).astype(np.int16)
        idx = np.rint(a / 23).astype(np.int16)
        off_grid = float((np.abs(a - idx * 23) > 3).mean())
        present = Counter(idx[np.abs(a - idx * 23) <= 3].ravel().tolist())
        rec["native_label"] = ";".join(FESB_CLASSES.get(k, f"code{k}") for k in sorted(present))
        smoke_frac = present.get(1, 0) / a.size
        rec["mask_frac"] = round(smoke_frac, 5)
        rec["has_smoke"] = int(smoke_frac > 0)
        rec["l2"] = POS if smoke_frac > 0 else NEG
        if off_grid > 0.05:
            rec["label_conflict"] = f"{off_grid:.1%} of mask pixels match no class code"


def process(job: dict) -> dict:
    kind, labels_from_ann = job.pop("_kind"), job.pop("_labels_from_ann")
    rec = {k: "" for k in FIELDS} | job
    path = RAW / rec["rel_path"]
    flags = []
    try:
        data = path.read_bytes()
    except OSError as e:
        rec.update(readable=0, error=f"read: {e}")
        return rec
    rec["bytes"] = len(data)
    rec["sha256"] = hashlib.sha256(data).hexdigest()
    try:
        with Image.open(io.BytesIO(data)) as im:
            rec["format"], rec["mode"] = im.format, im.mode
            rec["width"], rec["height"] = im.size
            im.load()
            gray = im.convert("L")
        rec["phash"], rec["dhash"] = phash(gray), dhash(gray)
        gray.thumbnail((512, 512), Image.Resampling.LANCZOS, reducing_gap=3.0)
        a = np.asarray(gray, dtype=np.float32)
        rec["mean_lum"], rec["std_lum"] = round(float(a.mean()), 2), round(float(a.std()), 2)
        if min(a.shape) >= 3:
            lap = a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:] - 4 * a[1:-1, 1:-1]
            rec["lap_var"] = round(float(lap.var()), 2)
        rec["readable"] = 1
    except Exception as e:
        rec.update(readable=0, error=f"decode: {type(e).__name__}: {e}")
    if rec["readable"] == 1:
        if min(rec["width"], rec["height"]) < 64 and rec["subset"] != "patch_50px":
            flags.append("tiny")
        if rec["std_lum"] < 5:
            flags.append("near_constant")
        if rec["mean_lum"] < 20:
            flags.append("very_dark")
    rec["flags"] = ";".join(flags)
    if kind:
        try:
            _read_annotation(rec, kind, labels_from_ann)
        except Exception as e:
            rec["ann_status"] = f"unparseable: {type(e).__name__}"
    return rec


def build_manifest(ds: str, workers: int) -> None:
    jobs = list(ADAPTERS[ds]())
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / f"{ds}.csv.partial"
    t0 = time.time()
    print(f"[{ds}] {len(jobs)} images, {workers} workers", flush=True)
    with open(tmp, "w", newline="", encoding="utf-8") as fh, ProcessPoolExecutor(workers) as ex:
        w = csv.DictWriter(fh, FIELDS)
        w.writeheader()
        for i, rec in enumerate(ex.map(process, jobs, chunksize=32), 1):
            w.writerow(rec)
            if i % 5000 == 0:
                rate = i / (time.time() - t0)
                print(f"[{ds}] {i}/{len(jobs)}  {rate:.0f} img/s  eta {(len(jobs) - i) / rate / 60:.1f} min", flush=True)
    os.replace(tmp, OUT / f"{ds}.csv")
    if skipped_files[ds]:
        (OUT / f"{ds}.skipped.txt").write_text(
            "\n".join(f"{k}\t{v}" for k, v in skipped_files[ds].items()) + "\n", encoding="utf-8")
    print(f"[{ds}] done in {(time.time() - t0) / 60:.1f} min", flush=True)


def _table(header: list[str], rows: list[list]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def write_summary() -> None:
    manifests = {ds: list(csv.DictReader(open(OUT / f"{ds}.csv", encoding="utf-8")))
                 for ds in DATASETS if (OUT / f"{ds}.csv").is_file()}
    L = [
        "# Dataset inventory summary",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `04_experiments/src/inventory.py` from `03_datasets/raw/`.",
        "Every number here was counted locally, not taken from a publication.",
        "",
        "## Overview",
        "",
    ]
    rows = []
    for ds, recs in manifests.items():
        main = [r for r in recs if r["subset"] != "patch_50px"]
        n_bad = sum(r["readable"] != "1" for r in recs)
        dup_extra = len(main) - len({r["sha256"] for r in main if r["sha256"]})
        rows.append([ds, DATASETS[ds], len(main), PUBLISHED[ds], len(main) - PUBLISHED[ds], n_bad,
                     dup_extra, sum(r["l2"] == POS for r in main), sum(r["l2"] == NEG for r in main),
                     sum(bool(r["label_conflict"]) for r in recs)])
    L += _table(["ID", "Dataset", "Images", "Reported", "Diff", "Unreadable", "Exact dups (extra copies)",
                 "L2 pos", "L2 neg", "Label conflicts"], rows)
    L += ["", "BoWFire's 50x50 training patches are excluded from the overview row and listed in its own section.", ""]

    for ds, recs in manifests.items():
        L += [f"## {ds} — {DATASETS[ds]}", ""]
        ok = [r for r in recs if r["readable"] == "1"]
        multi = len({r["native_label"] for r in recs}) > 15
        by = Counter((r["subset"] or "-", r["orig_split"] or "-", "(see below)" if multi else r["native_label"],
                      r["l2"], r["l4"] or "?") for r in recs)
        L += ["**Labels** (L4 `?` = not recoverable from this dataset's labels):", ""]
        L += _table(["Subset", "Orig. split", "Native", "L2", "L4", "Images"],
                    [[*k, n] for k, n in sorted(by.items())]) + [""]
        if multi:
            present = Counter(c for r in recs for c in r["native_label"].split(";") if c)
            L += ["- Images containing each mask class: " + ", ".join(f"{k} {v}" for k, v in present.most_common())]
        if ok:
            ws = [int(r["width"]) for r in ok]
            hs = [int(r["height"]) for r in ok]
            L += [f"- Size (W x H): min {min(ws)}x{min(hs)}, median {statistics.median(ws):.0f}x{statistics.median(hs):.0f}, "
                  f"max {max(ws)}x{max(hs)}",
                  f"- Formats: {dict(Counter(r['format'] for r in ok))}; modes: {dict(Counter(r['mode'] for r in ok))}"]
        bad = [r for r in recs if r["readable"] != "1"]
        L.append(f"- Unreadable: {len(bad)}" + ("".join(f"\n  - `{r['rel_path']}` — {r['error']}" for r in bad[:15])))
        flags = Counter(f for r in recs for f in r["flags"].split(";") if f)
        L.append(f"- Flags: {dict(flags) or 'none'}")
        groups = defaultdict(list)
        for r in recs:
            if r["sha256"]:
                groups[r["sha256"]].append(r)
        dups = [g for g in groups.values() if len(g) > 1]
        cross_split = [g for g in dups if len({r["orig_split"] for r in g}) > 1]
        cross_label = [g for g in dups if len({r["l2"] for r in g}) > 1]
        L.append(f"- Exact-duplicate groups: {len(dups)} ({sum(len(g) for g in dups)} files); "
                 f"spanning the publisher's splits: {len(cross_split)}; with contradictory L2 labels: {len(cross_label)}")
        for g in (cross_label + cross_split)[:5]:
            L.append("  - " + " = ".join(f"`{r['rel_path']}` ({r['orig_split']}, {r['l2']})" for r in g[:4]))
        ann = Counter(r["ann_status"] for r in recs if r["ann_path"])
        if ann:
            L.append(f"- Annotation files: {dict(ann)}")
        conflicts = [r for r in recs if r["label_conflict"]]
        L.append(f"- Label conflicts: {len(conflicts)}")
        for msg, n in Counter(r["label_conflict"] for r in conflicts).most_common(8):
            L.append(f"  - {n} x {msg}")
        if skipped_files.get(ds) or (OUT / f"{ds}.skipped.txt").is_file():
            L.append(f"- Non-image files skipped: see `manifest/{ds}.skipped.txt`")
        L.append("")

    L += ["## Shared images across datasets", "",
          "Images found in more than one dataset, matched by identical hashes only. Resized, cropped or "
          "edited copies need a Hamming-distance search on the pHash/dHash columns (the overlap audit), so "
          "every figure here is a lower bound.", ""]
    main = {ds: [r for r in recs if r["readable"] == "1" and r["subset"] != "patch_50px"]
            for ds, recs in manifests.items()}
    for title, key in (("Same SHA-256 (byte-identical)", lambda r: r["sha256"]),
                       ("Same pHash and dHash (identical or re-encoded; a lower bound on overlap)",
                        lambda r: (r["phash"], r["dhash"]))):
        index = defaultdict(lambda: defaultdict(list))
        for ds, recs in main.items():
            for r in recs:
                index[key(r)][ds].append(r)
        rows = []
        ids = list(main)
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                shared = [k for k, v in index.items() if a in v and b in v]
                if not shared:
                    continue
                na = sum(len(index[k][a]) for k in shared)
                nb = sum(len(index[k][b]) for k in shared)
                agree = sum(index[k][a][0]["l2"] == index[k][b][0]["l2"] for k in shared)
                rows.append([f"{a} / {b}", f"{na} ({na / len(main[a]):.1%})", f"{nb} ({nb / len(main[b]):.1%})",
                             f"{agree}/{len(shared)}"])
        rows.sort(key=lambda r: -max(float(r[1].split("(")[1][:-2]), float(r[2].split("(")[1][:-2])))
        L += [f"**{title}**:", ""]
        L += _table(["Pair A / B", "Images of A matched", "Images of B matched", "L2 labels agree"], rows) if rows else ["None found."]
        L.append("")

    SUMMARY.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"summary -> {SUMMARY}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="*", help=f"dataset IDs, any of {', '.join(DATASETS)} (default: all)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--force", action="store_true", help="rebuild manifests that already exist")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    unknown = set(args.datasets) - set(DATASETS)
    if unknown:
        ap.error(f"unknown dataset IDs: {', '.join(sorted(unknown))}")
    if not args.summary_only:
        for ds in args.datasets or DATASETS:
            if (OUT / f"{ds}.csv").is_file() and not args.force:
                print(f"[{ds}] manifest exists, skipping (use --force to rebuild)")
                continue
            try:
                build_manifest(ds, args.workers)
            except FileNotFoundError as e:
                print(f"[{ds}] SKIPPED, folder not found: {e}", file=sys.stderr)
    write_summary()


if __name__ == "__main__":
    main()
