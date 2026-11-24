"""Synthetic step 2: paste fire and smoke pieces onto no-fire backgrounds.

Fire is pasted through a flame-colour mask, smoke half-transparent with soft
edges, onto no-fire training images of another dataset. Image types: fire,
smoke, fire+smoke and none (a pasted non-fire patch). Writes the images and
synthetic_images.csv with the source of every background and piece.

Usage:
    python 04_experiments/src/make_synthetic_images.py [--count N] [--test]
"""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps


THESIS = Path(__file__).resolve().parents[2]
UPLOAD = THESIS / "03_datasets" / "processed" / "upload"
SPLIT_FILE = UPLOAD / "splits" / "within_dataset.csv"
SYNTHETIC = THESIS / "03_datasets" / "processed" / "synthetic"

BACKGROUND_DATASETS = ["R1", "R2", "R3", "R4"]

TYPE_SHARES = {"fire": 0.35, "smoke": 0.30, "fire+smoke": 0.15, "none": 0.20}

FIRE_AREA = (0.03, 0.20)
SMALL_FIRE_AREA = (0.01, 0.04)
SMALL_FIRE_CHANCE = 0.3
SMOKE_AREA = (0.08, 0.35)
SMOKE_OPACITY = (0.55, 0.9)

FLAME_MIN_RED = 160
FLAME_MIN_RED_MINUS_BLUE = 60
MIN_FLAME_SHARE = 0.08
MAX_FLAME_SHARE = 0.7
WHITE_HOT_MIN = 200
WHITE_HOT_REACH = 7

MAX_FLAME_SHARE_IN_SMOKE = 0.01
GREY_MAX_SATURATION = 0.25
MIN_GREY_SHARE = 0.5

OUTPUT_SHORT_SIDE = 256
JPEG_QUALITY = 90
RANDOM_SEED = 0


def read_options():
    parser = argparse.ArgumentParser(description="Make synthetic fire/smoke images")
    parser.add_argument("--count", type=int, default=5000, help="number of synthetic images")
    parser.add_argument("--test", action="store_true", help="48 images in synthetic_test/ to check by eye")
    return parser.parse_args()


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def flame_mask(rgb):
    """True where a pixel has a flame colour (array of shape h x w x 3, 0-255)."""
    r = rgb[..., 0].astype(int)
    g = rgb[..., 1].astype(int)
    b = rgb[..., 2].astype(int)
    return (r >= FLAME_MIN_RED) & (r >= g) & (g >= b) & (r - b >= FLAME_MIN_RED_MINUS_BLUE)


def grey_share(rgb):
    """Share of pixels that are grey or white (low saturation, not dark)."""
    hsv = np.asarray(Image.fromarray(rgb).convert("HSV")).astype(float) / 255
    return float(((hsv[..., 1] < GREY_MAX_SATURATION) & (hsv[..., 2] > 0.25)).mean())


def soft_window(width, height):
    """Alpha mask that is 1 in the middle and fades to 0 at the edges."""
    y = np.linspace(-1, 1, height)[:, None]
    x = np.linspace(-1, 1, width)[None, :]
    distance = np.sqrt(x ** 2 + y ** 2)
    return np.clip(1.3 - distance, 0, 1) ** 1.5


def prepare_piece(path, kind):
    """Return (RGB image, alpha mask 0-1) or None if the piece is not suitable."""
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
    if kind == "fire":
        mask = flame_mask(rgb) & (rgb.max(axis=2) >= 200)
        near_flame = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(WHITE_HOT_REACH))) > 0
        mask = mask | ((rgb.min(axis=2) >= WHITE_HOT_MIN) & near_flame)
        if mask.mean() < MIN_FLAME_SHARE or mask.mean() > MAX_FLAME_SHARE:
            return None
        closed = Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(5))
        alpha = closed.filter(ImageFilter.MaxFilter(3))
        alpha = np.asarray(alpha.filter(ImageFilter.GaussianBlur(2))).astype(float) / 255
        return rgb, alpha
    if flame_mask(rgb).mean() > MAX_FLAME_SHARE_IN_SMOKE or grey_share(rgb) < MIN_GREY_SHARE:
        return None
    return rgb, soft_window(rgb.shape[1], rgb.shape[0])


def paste(background, rgb, alpha, area_range, opacity, rng, lower_part=False):
    """Paste one piece onto the background (both arrays), in place."""
    bh, bw = background.shape[:2]
    ph, pw = rgb.shape[:2]
    area = rng.uniform(*area_range) * bh * bw
    scale = (area / (ph * pw)) ** 0.5
    new_w = max(8, min(bw, int(pw * scale)))
    new_h = max(8, min(bh, int(ph * scale)))
    piece = np.asarray(Image.fromarray(rgb).resize((new_w, new_h), Image.BILINEAR)).astype(float)
    mask = np.asarray(Image.fromarray((alpha * 255).astype(np.uint8)).resize((new_w, new_h), Image.BILINEAR))
    mask = mask.astype(float)[..., None] / 255 * opacity
    if rng.random() < 0.5:
        piece = piece[:, ::-1]
        mask = mask[:, ::-1]
    x = rng.randint(0, bw - new_w)
    y_min = (bh - new_h) // 3 if lower_part else 0
    y = rng.randint(min(y_min, bh - new_h), bh - new_h)
    region = background[y:y + new_h, x:x + new_w]
    background[y:y + new_h, x:x + new_w] = region * (1 - mask) + piece * mask


def load_background(path):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
    scale = OUTPUT_SHORT_SIDE / min(image.size)
    if scale < 1:
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.BILINEAR)
    return np.asarray(image).astype(float)


def main():
    options = read_options()
    rng = random.Random(RANDOM_SEED)
    count = 48 if options.test else options.count
    out = SYNTHETIC.parent / "synthetic_test" if options.test else SYNTHETIC

    pieces = {"fire": [], "smoke": []}
    rejected = {"fire": 0, "smoke": 0}
    for row in read_csv(SYNTHETIC / "pieces.csv"):
        prepared = prepare_piece(SYNTHETIC / row["piece"], row["kind"])
        if prepared is None:
            rejected[row["kind"]] += 1
            continue
        pieces[row["kind"]].append((row, prepared))
    print(f"usable pieces: fire {len(pieces['fire'])} (rejected {rejected['fire']}), "
          f"smoke {len(pieces['smoke'])} (rejected {rejected['smoke']})")

    backgrounds = [r for r in read_csv(SPLIT_FILE)
                   if r["dataset"] in BACKGROUND_DATASETS and r["split"] == "train" and r["l2"] == "none"]
    print(f"background images: {len(backgrounds)}")
    backgrounds_by_dataset = {d: [r for r in backgrounds if r["dataset"] == d] for d in BACKGROUND_DATASETS}

    def pick_piece(kind, not_from):
        while True:
            row, prepared = rng.choice(pieces[kind])
            if row["dataset"] != not_from:
                return row, prepared

    types = list(TYPE_SHARES)
    weights = list(TYPE_SHARES.values())
    (out / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for number in range(1, count + 1):
        image_type = rng.choices(types, weights)[0]
        bg_row = rng.choice(backgrounds_by_dataset[rng.choice(BACKGROUND_DATASETS)])
        image = load_background(UPLOAD / bg_row["image"])
        used = []

        if image_type in ("fire", "fire+smoke"):
            for _ in range(rng.choice([1, 1, 2])):
                row, (rgb, alpha) = pick_piece("fire", bg_row["dataset"])
                area = SMALL_FIRE_AREA if rng.random() < SMALL_FIRE_CHANCE else FIRE_AREA
                paste(image, rgb, alpha, area, 1.0, rng, lower_part=True)
                used.append(row)
        if image_type in ("smoke", "fire+smoke"):
            row, (rgb, alpha) = pick_piece("smoke", bg_row["dataset"])
            paste(image, rgb, alpha, SMOKE_AREA, rng.uniform(*SMOKE_OPACITY), rng)
            used.append(row)
        if image_type == "none":
            other = rng.choice([b for b in rng.sample(backgrounds, 20) if b["dataset"] != bg_row["dataset"]] or backgrounds)
            patch = load_background(UPLOAD / other["image"]).astype(np.uint8)
            h, w = patch.shape[:2]
            ph, pw = rng.randint(h // 4, h // 2), rng.randint(w // 4, w // 2)
            y, x = rng.randint(0, h - ph), rng.randint(0, w - pw)
            patch = patch[y:y + ph, x:x + pw]
            paste(image, patch, soft_window(pw, ph), SMOKE_AREA, rng.uniform(*SMOKE_OPACITY), rng)
            used.append({"dataset": other["dataset"], "piece": other["image"], "cluster_id": other["cluster_id"]})

        name = f"syn_{number:05d}.jpg"
        Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).save(out / "images" / name, quality=JPEG_QUALITY)
        l4 = {"fire": "fire", "smoke": "smoke", "fire+smoke": "fire_and_smoke", "none": "none"}[image_type]
        rows.append({
            "image": f"images/{name}", "l2": "none" if image_type == "none" else "fire_or_smoke", "l4": l4,
            "type": image_type, "background_dataset": bg_row["dataset"], "background_image": bg_row["image"],
            "background_cluster": bg_row["cluster_id"],
            "piece_datasets": " ".join(u["dataset"] for u in used),
            "pieces": " ".join(u["piece"] for u in used),
            "piece_clusters": " ".join(u["cluster_id"] for u in used),
        })
        if number % 500 == 0:
            print(f"  {number}/{count} images")

    with open(out / "synthetic_images.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    sample = rng.sample(rows, min(48, len(rows)))
    tile = 160
    sheet = Image.new("RGB", (8 * tile, ((len(sample) + 7) // 8) * tile), "white")
    for k, r in enumerate(sample):
        with Image.open(out / r["image"]) as im:
            im = im.copy()
        im.thumbnail((tile - 4, tile - 4))
        sheet.paste(im, ((k % 8) * tile + 2, (k // 8) * tile + 2))
    sheet.save(out / "check_synthetic.jpg", quality=85)

    lines = ["# Synthetic step 2: synthetic images", "",
             f"Usable pieces: fire {len(pieces['fire'])} (rejected {rejected['fire']}), "
             f"smoke {len(pieces['smoke'])} (rejected {rejected['smoke']}).",
             f"Background images (no-fire training images of R1-R4): {len(backgrounds)}.", "",
             "| Type | Images |", "|---|---|"]
    for t in types:
        lines.append(f"| {t} | {sum(1 for r in rows if r['type'] == t)} |")
    lines += ["", "| Background dataset | Images |", "|---|---|"]
    for d in BACKGROUND_DATASETS:
        lines.append(f"| {d} | {sum(1 for r in rows if r['background_dataset'] == d)} |")
    (out / "step2_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"written: {out}")


if __name__ == "__main__":
    main()
