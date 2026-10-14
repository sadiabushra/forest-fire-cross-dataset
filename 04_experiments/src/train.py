"""Step 5: train and test one fire/no-fire classifier.

Fine-tunes an ImageNet-pretrained model (ResNet-50, MobileNetV2 or
EfficientNet-B3; frozen or full) on one split file, keeps the epoch with the best
validation F1, and reports test metrics overall and per dataset.

Usage:
    python 04_experiments/src/train.py --data UPLOAD_FOLDER [--split-file FILE]
        [--dataset ID] [--model NAME] [--freeze] [--epochs N] [--quick]
"""

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


LABELS = {"none": 0, "fire_or_smoke": 1}

MODEL_NAMES = ["resnet50", "mobilenet_v2", "efficientnet_b3"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

QUICK_TRAIN, QUICK_VAL, QUICK_TEST = 300, 100, 100
QUICK_EPOCHS = 2
QUICK_IMAGE_SIZE = 128


def read_options():
    parser = argparse.ArgumentParser(description="Train and test one fire/no-fire classifier.")
    parser.add_argument("--data", required=True, help="the unzipped upload folder (has images/ and splits/)")
    parser.add_argument("--split-file", default="within_dataset.csv",
                        help="file in <data>/splits/: within_dataset.csv, ref_to_shift.csv, lodo_R1.csv ...")
    parser.add_argument("--dataset", default=None,
                        help="keep only this dataset (e.g. R1). Needed for within_dataset.csv")
    parser.add_argument("--model", default="resnet50", choices=MODEL_NAMES)
    parser.add_argument("--freeze", action="store_true", help="train only the new last layer")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=None,
                        help="learning rate (default 1e-3 with --freeze, 1e-4 without)")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4, help="processes that load images")
    parser.add_argument("--out", default="results", help="folder where run folders are made")
    parser.add_argument("--no-save", action="store_true", help="do not save best_model.pt")
    parser.add_argument("--quick", action="store_true",
                        help=f"CPU test: {QUICK_TRAIN}/{QUICK_VAL}/{QUICK_TEST} images, "
                             f"{QUICK_EPOCHS} epochs, {QUICK_IMAGE_SIZE} px")
    options = parser.parse_args()

    if options.lr is None:
        options.lr = 1e-3 if options.freeze else 1e-4
    if options.quick:
        options.epochs = QUICK_EPOCHS
        options.image_size = QUICK_IMAGE_SIZE
    return options


def read_split(options):
    """Return a dict {"train": rows, "val": rows, "test": rows}."""
    path = Path(options.data) / "splits" / options.split_file
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    split_column = "split" if "split" in rows[0] else "role"

    if options.dataset:
        rows = [r for r in rows if r["dataset"] == options.dataset]
        if not rows:
            sys.exit(f"No images of dataset {options.dataset} in {path}")

    parts = {"train": [], "val": [], "test": []}
    for row in rows:
        part = row[split_column]
        if part in parts:
            parts[part].append({"image": row["image"], "dataset": row["dataset"], "label": LABELS[row["l2"]]})

    if options.quick:
        rng = random.Random(options.seed)
        sizes = {"train": QUICK_TRAIN, "val": QUICK_VAL, "test": QUICK_TEST}
        for part in parts:
            rng.shuffle(parts[part])
            parts[part] = parts[part][:sizes[part]]

    for part in parts:
        if not parts[part]:
            sys.exit(f"The {part} part is empty. Check --split-file and --dataset.")
    return parts


class ImageList(Dataset):
    """Gives PyTorch one (image tensor, label) pair at a time."""

    def __init__(self, rows, data_folder, transform):
        self.rows = rows
        self.data_folder = Path(data_folder)
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.data_folder / row["image"]) as img:
            img = img.convert("RGB")
        return self.transform(img), row["label"]


def make_transforms(image_size):
    """Image changes for training (random, for augmentation) and for val/test (fixed)."""
    train = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    evaluate = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train, evaluate


def make_model(name, freeze):
    """Load a pretrained model and give it a new 2-class last layer."""
    if name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, 2)
        new_layer = model.fc
    elif name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
        new_layer = model.classifier[1]
    elif name == "efficientnet_b3":
        model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
        new_layer = model.classifier[1]
    else:
        raise ValueError(f"Unknown model {name}")

    if freeze:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in new_layer.parameters():
            parameter.requires_grad = True
    return model


def class_weights(train_rows):
    """Weights so both classes count equally in the loss, even if one has more images."""
    counts = Counter(r["label"] for r in train_rows)
    total = len(train_rows)
    weights = [total / (2 * max(1, counts[label])) for label in (0, 1)]
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(model, loader, loss_function, optimizer, device, use_amp, scaler):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            loss = loss_function(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


def predict(model, loader, device, use_amp):
    """Return (true labels, predicted fire probabilities) for every image in the loader."""
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for images, labels in loader:
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images.to(device))
            probs = torch.softmax(outputs.float(), dim=1)[:, 1]
            all_labels.extend(labels.tolist())
            all_probs.extend(probs.cpu().tolist())
    return np.array(all_labels), np.array(all_probs)


def scores(labels, probs):
    """Accuracy, precision, recall, F1 (fire = positive), AUC and confusion matrix."""
    predicted = (probs >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted, average="binary", pos_label=1, zero_division=0)
    if len(set(labels.tolist())) == 2:
        auc = float(roc_auc_score(labels, probs))
    else:
        auc = None
    matrix = confusion_matrix(labels, predicted, labels=[0, 1]).tolist()
    return {
        "images": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predicted)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": auc,
        "confusion_matrix_[none,fire]": matrix,
    }


def set_seed(seed):
    """Same seed = same random choices, so a run can be repeated."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_name(options):
    """Folder name that says what the run was, e.g. within_R1_resnet50_frozen_seed42."""
    split = Path(options.split_file).stem.replace("_dataset", "")
    parts = [split]
    if options.dataset:
        parts.append(options.dataset)
    parts.append(options.model)
    parts.append("frozen" if options.freeze else "finetuned")
    parts.append(f"seed{options.seed}")
    if options.quick:
        parts.append("QUICK")
    return "_".join(parts)


class Log:
    """Prints a line on screen and also writes it to log.txt."""

    def __init__(self, path):
        self.file = open(path, "w", encoding="utf-8")

    def __call__(self, text):
        print(text, flush=True)
        self.file.write(text + "\n")
        self.file.flush()


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main():
    options = read_options()
    set_seed(options.seed)

    out_folder = Path(options.out) / run_name(options)
    out_folder.mkdir(parents=True, exist_ok=True)
    log = Log(out_folder / "log.txt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    log(f"Run {out_folder.name} on {device}" + (f" ({torch.cuda.get_device_name(0)})" if use_amp else ""))

    parts = read_split(options)
    for part, rows in parts.items():
        fire = sum(r["label"] for r in rows)
        log(f"  {part}: {len(rows)} images, {fire} fire_or_smoke, {len(rows) - fire} none")
    train_transform, eval_transform = make_transforms(options.image_size)
    loaders = {}
    for part, rows in parts.items():
        transform = train_transform if part == "train" else eval_transform
        loaders[part] = DataLoader(ImageList(rows, options.data, transform),
                                   batch_size=options.batch_size, shuffle=(part == "train"),
                                   num_workers=options.workers, pin_memory=use_amp)

    settings = vars(options).copy()
    settings["device"] = str(device)
    settings["images"] = {part: len(rows) for part, rows in parts.items()}
    write_json(out_folder / "settings.json", settings)

    model = make_model(options.model, options.freeze).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    log(f"  model {options.model}, {sum(p.numel() for p in trainable):,} trainable weights")
    loss_function = nn.CrossEntropyLoss(weight=class_weights(parts["train"]).to(device))
    optimizer = torch.optim.AdamW(trainable, lr=options.lr)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_f1 = -1.0
    best_weights = None
    best_epoch = 0
    history = []
    for epoch in range(1, options.epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, loaders["train"], loss_function, optimizer, device, use_amp, scaler)
        val = scores(*predict(model, loaders["val"], device, use_amp))
        minutes = (time.time() - start) / 60
        history.append({"epoch": epoch, "train_loss": round(train_loss, 5), "val_accuracy": round(val["accuracy"], 5),
                        "val_f1": round(val["f1"], 5), "val_auc": val["auc"], "minutes": round(minutes, 2)})
        log(f"  epoch {epoch}/{options.epochs}: loss {train_loss:.4f}, val acc {val['accuracy']:.4f}, "
            f"val F1 {val['f1']:.4f}, {minutes:.1f} min")
        if val["f1"] > best_f1:
            best_f1 = val["f1"]
            best_epoch = epoch
            best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    with open(out_folder / "history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    log(f"  best epoch {best_epoch} (val F1 {best_f1:.4f}); testing it")
    model.load_state_dict(best_weights)
    if not options.no_save:
        torch.save(best_weights, out_folder / "best_model.pt")
    labels, probs = predict(model, loaders["test"], device, use_amp)
    results = {"best_epoch": best_epoch, "best_val_f1": best_f1, "overall": scores(labels, probs), "per_dataset": {}}
    test_datasets = np.array([r["dataset"] for r in parts["test"]])
    for dataset in sorted(set(test_datasets.tolist())):
        keep = test_datasets == dataset
        results["per_dataset"][dataset] = scores(labels[keep], probs[keep])
    write_json(out_folder / "test_metrics.json", results)

    with open(out_folder / "test_predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "dataset", "label", "fire_probability"])
        for row, prob in zip(parts["test"], probs):
            writer.writerow([row["image"], row["dataset"], row["label"], round(float(prob), 5)])

    overall = results["overall"]
    auc_text = f"{overall['auc']:.4f}" if overall["auc"] is not None else "n/a"
    log(f"TEST: acc {overall['accuracy']:.4f}, precision {overall['precision']:.4f}, "
        f"recall {overall['recall']:.4f}, F1 {overall['f1']:.4f}, AUC {auc_text}")
    for dataset, s in results["per_dataset"].items():
        log(f"  {dataset}: {s['images']} images, acc {s['accuracy']:.4f}, F1 {s['f1']:.4f}")
    log(f"Results in {out_folder}")


if __name__ == "__main__":
    main()
