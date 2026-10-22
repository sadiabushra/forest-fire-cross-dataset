"""Step 5 (Keras + TensorFlow): train one fire/no-fire classifier and test it on every dataset.

Fine-tunes an ImageNet-pretrained model (ResNet-50, MobileNetV2, EfficientNet-B3,
DenseNet-121 or ViT-B/16; frozen or full) on one training dataset (R1-R4), keeps
the epoch with the best validation macro-F1, and tests it on its own test split
and, without retraining, on every other dataset (leak-free by cluster). Reports
accuracy, balanced accuracy, macro-F1, precision, recall, AUC, confusion
matrices, speed and size, and saves every prediction.

Usage:
    python 04_experiments/src/train_tf.py --data UPLOAD_FOLDER --dataset ID
        [--model NAME] [--freeze] [--epochs N] [--batch-size N] [--quick]
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
import keras
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)


LABELS = {"none": 0, "fire_or_smoke": 1}

MODEL_NAMES = ["resnet50", "mobilenet_v2", "efficientnet_b3", "densenet121", "vit_b16"]

VIT_PRESET = "vit_base_patch16_224_imagenet"
VIT_IMAGE_SIZE = 224

TRAIN_DATASETS = ["R1", "R2", "R3", "R4"]
TEST_ONLY_DATASETS = ["S1", "S2", "S3", "ALT_WILDFIRE"]

LR_FROZEN = 1e-3
LR_FINETUNED = {"vit_b16": 3e-5}
LR_FINETUNED_DEFAULT = 1e-4

QUICK_TRAIN, QUICK_VAL, QUICK_TEST = 300, 100, 100
QUICK_EPOCHS = 2
QUICK_IMAGE_SIZE = 128

SPEED_BATCH = 32
SPEED_REPEATS = 10


def read_options():
    parser = argparse.ArgumentParser(description="Train one fire/no-fire classifier and test it on every dataset.")
    parser.add_argument("--data", required=True, help="the unzipped upload folder (has images/ and splits/)")
    parser.add_argument("--dataset", required=True, choices=TRAIN_DATASETS, help="the dataset to train on")
    parser.add_argument("--model", default="resnet50", choices=MODEL_NAMES)
    parser.add_argument("--freeze", action="store_true", help="train only the new last layer")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=None, help="learning rate (default: see LR_ settings)")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="results_tf", help="folder where run folders are made")
    parser.add_argument("--no-save", action="store_true", help="do not save best.weights.h5")
    parser.add_argument("--no-mixed-precision", action="store_true",
                        help="use float32 on the GPU (slower; only if mixed precision causes problems)")
    parser.add_argument("--quick", action="store_true",
                        help=f"CPU test: {QUICK_TRAIN}/{QUICK_VAL}/{QUICK_TEST} images, "
                             f"{QUICK_EPOCHS} epochs, {QUICK_IMAGE_SIZE} px")
    options = parser.parse_args()

    if options.quick:
        options.epochs = QUICK_EPOCHS
        options.image_size = QUICK_IMAGE_SIZE
    if options.model == "vit_b16":
        options.image_size = VIT_IMAGE_SIZE
    if options.lr is None:
        if options.freeze:
            options.lr = LR_FROZEN
        else:
            options.lr = LR_FINETUNED.get(options.model, LR_FINETUNED_DEFAULT)
    return options


def read_rows(data_folder):
    """All rows of within_dataset.csv, each as a small dict."""
    path = Path(data_folder) / "splits" / "within_dataset.csv"
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({"image": row["image"], "dataset": row["dataset"], "cluster": row["cluster_id"],
                         "split": row["split"], "label": LABELS[row["l2"]]})
    return rows


def choose_images(rows, options):
    """Return (parts, test_sets).  parts     = {"train": rows, "val": rows} of the training dataset test_sets = {dataset name: rows}, the own test part first, then every other dataset"""
    source = options.dataset
    parts = {"train": [], "val": []}
    for row in rows:
        if row["dataset"] == source and row["split"] in parts:
            parts[row["split"]].append(row)

    used_clusters = set()
    for row in parts["train"] + parts["val"]:
        used_clusters.add(row["cluster"])

    test_sets = {}
    for target in [source] + [d for d in TRAIN_DATASETS + TEST_ONLY_DATASETS if d != source]:
        chosen = []
        for row in rows:
            if row["dataset"] != target:
                continue
            if target in TRAIN_DATASETS and row["split"] != "test":
                continue
            if row["cluster"] in used_clusters:
                continue
            chosen.append(row)
        if chosen:
            test_sets[target] = chosen

    if options.quick:
        rng = random.Random(options.seed)
        for name, size in (("train", QUICK_TRAIN), ("val", QUICK_VAL)):
            rng.shuffle(parts[name])
            parts[name] = parts[name][:size]
        for target in test_sets:
            rng.shuffle(test_sets[target])
            test_sets[target] = test_sets[target][:QUICK_TEST]

    for name in parts:
        if not parts[name]:
            sys.exit(f"The {name} part of {source} is empty. Check the split file.")
    return parts, test_sets


def scale_for_model(model_name):
    """Return a function that turns RGB pixels 0-255 into what the pretrained model expects."""
    if model_name == "resnet50":
        return keras.applications.resnet50.preprocess_input
    if model_name == "densenet121":
        return keras.applications.densenet.preprocess_input
    if model_name == "efficientnet_b3":
        return lambda images: images
    return lambda images: images / 127.5 - 1.0


def load_image(path):
    """Read one image file into a float tensor of RGB pixels 0-255."""
    data = tf.io.read_file(path)
    image = tf.io.decode_image(data, channels=3, expand_animations=False)
    return tf.cast(image, tf.float32)


def train_view(image, size):
    """Random crop, flip and small brightness/contrast change (augmentation for training)."""
    begin, crop_size, _ = tf.image.sample_distorted_bounding_box(
        tf.shape(image), bounding_boxes=tf.zeros([1, 0, 4]), area_range=(0.6, 1.0),
        aspect_ratio_range=(3 / 4, 4 / 3), min_object_covered=0.0, use_image_if_no_bounding_boxes=True)
    image = tf.slice(image, begin, crop_size)
    image = tf.image.resize(image, (size, size))
    image = tf.image.random_flip_left_right(image)
    image = image * tf.random.uniform([], 0.8, 1.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    return tf.clip_by_value(image, 0.0, 255.0)


def eval_view(image, size):
    """Always the same: shortest side resized to `size`, then the centre square."""
    height = tf.cast(tf.shape(image)[0], tf.float32)
    width = tf.cast(tf.shape(image)[1], tf.float32)
    factor = size / tf.minimum(height, width)
    new_height = tf.cast(tf.math.ceil(height * factor), tf.int32)
    new_width = tf.cast(tf.math.ceil(width * factor), tf.int32)
    image = tf.image.resize(image, (new_height, new_width))
    return tf.image.resize_with_crop_or_pad(image, size, size)


def make_dataset(rows, options, training):
    """A tf.data pipeline that gives batches of (images, labels)."""
    paths = [str(Path(options.data) / row["image"]) for row in rows]
    labels = [row["label"] for row in rows]
    scale = scale_for_model(options.model)
    size = options.image_size

    def load(path, label):
        image = load_image(path)
        image = train_view(image, size) if training else eval_view(image, size)
        return scale(image), label

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(len(paths), seed=options.seed, reshuffle_each_iteration=True)
    dataset = dataset.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(options.batch_size)
    return dataset.prefetch(tf.data.AUTOTUNE)


def make_backbone(name, size):
    """The pretrained model without its ImageNet output layer."""
    shape = (size, size, 3)
    if name == "resnet50":
        return keras.applications.ResNet50(include_top=False, weights="imagenet", input_shape=shape, pooling="avg")
    if name == "mobilenet_v2":
        return keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=shape, pooling="avg")
    if name == "efficientnet_b3":
        return keras.applications.EfficientNetB3(include_top=False, weights="imagenet", input_shape=shape, pooling="avg")
    if name == "densenet121":
        return keras.applications.DenseNet121(include_top=False, weights="imagenet", input_shape=shape, pooling="avg")
    if name == "vit_b16":
        import keras_hub
        return keras_hub.models.Backbone.from_preset(VIT_PRESET)
    raise ValueError(f"Unknown model {name}")


def make_model(name, size, freeze):
    """Pretrained backbone + a new 2-class output layer."""
    backbone = make_backbone(name, size)
    backbone.trainable = not freeze

    inputs = keras.Input(shape=(size, size, 3))
    features = backbone(inputs, training=False)
    if name == "vit_b16":
        features = features[:, 0, :]
    outputs = keras.layers.Dense(2, activation="softmax", dtype="float32", name="fire_output")(features)
    return keras.Model(inputs, outputs, name=name)


def count_weights(model):
    total = int(sum(np.prod(w.shape) for w in model.weights))
    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    return total, trainable


def class_weights(train_rows):
    """Weights so both classes count equally in the loss, even if one has more images."""
    counts = Counter(r["label"] for r in train_rows)
    total = len(train_rows)
    return {label: total / (2 * max(1, counts[label])) for label in (0, 1)}


def predict(model, rows, options):
    """Return (true labels, predicted fire probabilities) for the given rows."""
    probs = model.predict(make_dataset(rows, options, training=False), verbose=0)[:, 1]
    labels = np.array([r["label"] for r in rows])
    return labels, probs.astype(float)


def scores(labels, probs):
    """All the scores the supervisor asked for."""
    predicted = (probs >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted, average="binary", pos_label=1, zero_division=0)
    macro_f1 = precision_recall_fscore_support(
        labels, predicted, average="macro", labels=[0, 1], zero_division=0)[2]
    both_classes = len(set(labels.tolist())) == 2
    matrix = confusion_matrix(labels, predicted, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel().tolist()
    return {
        "images": int(len(labels)),
        "fire_images": int(labels.sum()),
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)) if both_classes else None,
        "macro_f1": float(macro_f1),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": tn / (tn + fp) if (tn + fp) else None,
        "auc": float(roc_auc_score(labels, probs)) if both_classes else None,
        "confusion_matrix_[none,fire]": matrix.tolist(),
        "true_negatives": tn, "false_positives": fp, "false_negatives": fn, "true_positives": tp,
    }


def measure_speed(model, size):
    """Milliseconds per image on random input: in batches of SPEED_BATCH, and one image at a time."""
    results = {}
    run_model = tf.function(lambda images: model(images, training=False))
    for batch in (SPEED_BATCH, 1):
        images = tf.random.uniform((batch, size, size, 3))
        for _ in range(3):
            run_model(images)
        start = time.perf_counter()
        for _ in range(SPEED_REPEATS):
            output = run_model(images)
        _ = output.numpy()
        seconds = time.perf_counter() - start
        results[f"ms_per_image_batch{batch}"] = round(1000 * seconds / (SPEED_REPEATS * batch), 3)
    return results


def run_name(options):
    """Folder name that says what the run was, e.g. within_R1_resnet50_frozen_seed42."""
    mode = "frozen" if options.freeze else "finetuned"
    name = f"within_{options.dataset}_{options.model}_{mode}_seed{options.seed}"
    if options.quick:
        name += "_QUICK"
    return name


def set_up_gpu(options):
    """Take GPU memory only when needed (good on a shared server), and use mixed precision."""
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    use_mixed = bool(gpus) and not options.no_mixed_precision
    if use_mixed:
        keras.mixed_precision.set_global_policy("mixed_float16")
    if gpus:
        details = tf.config.experimental.get_device_details(gpus[0])
        device = f"GPU ({details.get('device_name', gpus[0].name)})"
    else:
        device = "CPU"
    return device, use_mixed


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


def short(value):
    """A score as text with 4 decimals, or n/a."""
    return "n/a" if value is None else f"{value:.4f}"


def main():
    options = read_options()
    keras.utils.set_random_seed(options.seed)
    device, use_mixed = set_up_gpu(options)

    out_folder = Path(options.out) / run_name(options)
    out_folder.mkdir(parents=True, exist_ok=True)
    log = Log(out_folder / "log.txt")
    log(f"Run {out_folder.name} on {device}, TensorFlow {tf.__version__}, Keras {keras.__version__}")

    parts, test_sets = choose_images(read_rows(options.data), options)
    for name, rows in parts.items():
        fire = sum(r["label"] for r in rows)
        log(f"  {name}: {len(rows)} images, {fire} fire_or_smoke, {len(rows) - fire} none")
    log("  test sets: " + ", ".join(f"{d} {len(rows)}" for d, rows in test_sets.items()))

    settings = vars(options).copy()
    settings.update({"device": device, "mixed_precision": use_mixed, "tensorflow": tf.__version__,
                     "keras": keras.__version__,
                     "images": {name: len(rows) for name, rows in parts.items()},
                     "test_images": {d: len(rows) for d, rows in test_sets.items()}})
    write_json(out_folder / "settings.json", settings)

    model = make_model(options.model, options.image_size, options.freeze)
    total_weights, trainable_weights = count_weights(model)
    log(f"  model {options.model}: {total_weights:,} weights, {trainable_weights:,} of them learn")
    model.compile(optimizer=keras.optimizers.AdamW(learning_rate=options.lr),
                  loss="sparse_categorical_crossentropy")
    train_data = make_dataset(parts["train"], options, training=True)
    weights = class_weights(parts["train"])

    best_score = -1.0
    best_weights = None
    best_epoch = 0
    history = []
    train_start = time.time()
    for epoch in range(1, options.epochs + 1):
        start = time.time()
        fit = model.fit(train_data, epochs=1, class_weight=weights, shuffle=False, verbose=0)
        train_loss = float(fit.history["loss"][0])
        val = scores(*predict(model, parts["val"], options))
        minutes = (time.time() - start) / 60
        history.append({"epoch": epoch, "train_loss": round(train_loss, 5),
                        "val_accuracy": round(val["accuracy"], 5), "val_macro_f1": round(val["macro_f1"], 5),
                        "val_balanced_accuracy": val["balanced_accuracy"], "val_auc": val["auc"],
                        "minutes": round(minutes, 2)})
        log(f"  epoch {epoch}/{options.epochs}: loss {train_loss:.4f}, val acc {val['accuracy']:.4f}, "
            f"val macro-F1 {val['macro_f1']:.4f}, {minutes:.1f} min")
        if val["macro_f1"] > best_score:
            best_score = val["macro_f1"]
            best_epoch = epoch
            best_weights = model.get_weights()
    train_minutes = (time.time() - train_start) / 60

    with open(out_folder / "history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    log(f"  best epoch {best_epoch} (val macro-F1 {best_score:.4f}); testing it")
    model.set_weights(best_weights)
    if not options.no_save:
        clean_copy = make_model(options.model, options.image_size, options.freeze)
        clean_copy.set_weights(best_weights)
        clean_copy.save_weights(out_folder / "best.weights.h5")
        del clean_copy

    results = {"train_dataset": options.dataset, "model": options.model,
               "mode": "frozen" if options.freeze else "finetuned",
               "best_epoch": best_epoch, "best_val_macro_f1": best_score,
               "train_minutes": round(train_minutes, 2), "test": {}}
    prediction_rows = []
    for target, rows in test_sets.items():
        labels, probs = predict(model, rows, options)
        results["test"][target] = scores(labels, probs)
        for row, prob in zip(rows, probs):
            prediction_rows.append([target, row["image"], row["label"], round(float(prob), 5)])

    results["weights_total"] = total_weights
    results["weights_trainable"] = trainable_weights
    results.update(measure_speed(model, options.image_size))
    write_json(out_folder / "test_metrics.json", results)

    with open(out_folder / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["test_dataset", "image", "label", "fire_probability"])
        writer.writerows(prediction_rows)

    log("TEST (first line = own test set, the others = cross-dataset):")
    for target, s in results["test"].items():
        log(f"  {target:13s} {s['images']:6d} images: acc {s['accuracy']:.4f}, "
            f"macro-F1 {s['macro_f1']:.4f}, bal. acc {short(s['balanced_accuracy'])}, "
            f"precision {s['precision']:.4f}, recall {s['recall']:.4f}")
    log(f"  speed: {results['ms_per_image_batch32']} ms/image in batches of {SPEED_BATCH}, "
        f"{results['ms_per_image_batch1']} ms for one image; training {train_minutes:.1f} min")
    log(f"Results in {out_folder}")


if __name__ == "__main__":
    main()
