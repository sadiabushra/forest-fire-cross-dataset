"""Step 6 (Keras + TensorFlow): train the hybrid CNN + ViT model and test it on every dataset.

CNN branch (DenseNet-121, or MobileNet-V2 with --cnn) + ViT-B/16 branch, joined by
concatenation (--fusion concat) or by one transformer block over both feature sets
(--fusion transformer).
Optional MixStyle in the early CNN layers (--mixstyle) and copy-paste synthetic
training images (--synthetic). Uses the data, test sets and scores of train_tf.py
and writes the same result files.

Usage:
    python 04_experiments/src/train_hybrid_tf.py --data UPLOAD_FOLDER --dataset ID
        [--cnn densenet121|mobilenet_v2] [--fusion concat|transformer] [--mixstyle] [--synthetic FOLDER]
        [--synthetic-share S] [--freeze] [--epochs N] [--batch-size N] [--seed N] [--quick]
"""

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import keras

import train_tf as base


FUSIONS = ["concat", "transformer"]

CNNS = ["densenet121", "mobilenet_v2"]

IMAGE_SIZE = 224

LR_FROZEN = 1e-3
LR_FINETUNED = 3e-5

DROPOUT = 0.2

TOKEN_SIZE = 256
ATTENTION_HEADS = 4

MIXSTYLE_AFTER = {"densenet121": ["conv2_block6_concat", "conv3_block12_concat"],
                  "mobilenet_v2": ["block_2_add", "block_5_add"]}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
MIXSTYLE_PROBABILITY = 0.5
MIXSTYLE_ALPHA = 0.1

SYNTHETIC_SHARE = 0.25


def read_options():
    parser = argparse.ArgumentParser(description="Train the hybrid CNN + ViT model and test it on every dataset.")
    parser.add_argument("--data", required=True, help="the unzipped upload folder (has images/ and splits/)")
    parser.add_argument("--dataset", required=True, choices=base.TRAIN_DATASETS, help="the dataset to train on")
    parser.add_argument("--fusion", default="concat", choices=FUSIONS)
    parser.add_argument("--cnn", default="densenet121", choices=CNNS, help="the CNN branch")
    parser.add_argument("--mixstyle", action="store_true", help="MixStyle in the early CNN layers")
    parser.add_argument("--synthetic", default="", help="folder with synthetic_images.csv and images/")
    parser.add_argument("--synthetic-share", type=float, default=SYNTHETIC_SHARE)
    parser.add_argument("--freeze", action="store_true", help="both backbones fixed, only the new layers learn")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment", default="standard", choices=base.AUGMENT_LEVELS)
    parser.add_argument("--tag", default="", help="text added to the run folder name")
    parser.add_argument("--out", default="results_tf", help="folder where run folders are made")
    parser.add_argument("--no-save", action="store_true", help="do not save best.weights.h5")
    parser.add_argument("--no-mixed-precision", action="store_true")
    parser.add_argument("--quick", action="store_true", help="CPU test: tiny run, 2 epochs")
    options = parser.parse_args()

    options.model = f"hybrid-{options.cnn}-{options.fusion}"
    options.image_size = IMAGE_SIZE
    if options.quick:
        options.epochs = base.QUICK_EPOCHS
    if options.lr is None:
        options.lr = LR_FROZEN if options.freeze else LR_FINETUNED
    return options


class MixStyle(keras.layers.Layer):
    """Mixes the feature statistics (style) of images in a batch, during training only."""

    def __init__(self, probability=MIXSTYLE_PROBABILITY, alpha=MIXSTYLE_ALPHA, **kwargs):
        super().__init__(**kwargs)
        self.probability = probability
        self.alpha = alpha

    def mix(self, x):
        dtype = x.dtype
        x = tf.cast(x, tf.float32)
        batch = tf.shape(x)[0]
        mean = tf.stop_gradient(tf.reduce_mean(x, axis=[1, 2], keepdims=True))
        std = tf.stop_gradient(tf.math.reduce_std(x, axis=[1, 2], keepdims=True) + 1e-6)
        normed = (x - mean) / std
        g1 = tf.random.gamma([batch, 1, 1, 1], self.alpha)
        g2 = tf.random.gamma([batch, 1, 1, 1], self.alpha)
        lam = g1 / (g1 + g2 + 1e-8)
        order = tf.random.shuffle(tf.range(batch))
        mixed_mean = lam * mean + (1 - lam) * tf.gather(mean, order)
        mixed_std = lam * std + (1 - lam) * tf.gather(std, order)
        return tf.cast(normed * mixed_std + mixed_mean, dtype)

    def call(self, x, training=None):
        if not training:
            return x
        return tf.cond(tf.random.uniform([]) < self.probability, lambda: self.mix(x), lambda: x)

    def get_config(self):
        config = super().get_config()
        config.update({"probability": self.probability, "alpha": self.alpha})
        return config


def insert_mixstyle(cnn, after_layers):
    """Rebuild the CNN with a MixStyle layer after each named layer (same layers, same weights)."""
    new_input = keras.Input(shape=cnn.input.shape[1:])
    new_tensors = {id(cnn.input): new_input}
    for layer in cnn.layers[1:]:
        node = layer._inbound_nodes[0]
        inputs = [new_tensors[id(t)] for t in node.input_tensors]
        output = layer(inputs[0] if len(inputs) == 1 else inputs)
        if layer.name in after_layers:
            output = MixStyle(name=f"mixstyle_{layer.name}")(output)
        new_tensors[id(node.output_tensors[0])] = output
    return keras.Model(new_input, new_tensors[id(cnn.output)], name=cnn.name)


def make_hybrid(options):
    size = options.image_size
    inputs = keras.Input(shape=(size, size, 3))

    if options.cnn == "densenet121":
        cnn = keras.applications.DenseNet121(include_top=False, weights="imagenet", input_shape=(size, size, 3))
        cnn_input = keras.layers.Rescaling(0.5, offset=0.5)(inputs)
        cnn_input = keras.layers.Normalization(mean=IMAGENET_MEAN, variance=[s ** 2 for s in IMAGENET_STD])(cnn_input)
    else:
        cnn = keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=(size, size, 3))
        cnn_input = inputs
    if options.mixstyle:
        cnn = insert_mixstyle(cnn, MIXSTYLE_AFTER[options.cnn])
    cnn.trainable = not options.freeze
    for layer in cnn.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
    cnn_map = cnn(cnn_input)

    import keras_hub
    vit = keras_hub.models.Backbone.from_preset(base.VIT_PRESET)
    vit.trainable = not options.freeze
    vit_tokens = vit(inputs, training=False)

    if options.fusion == "concat":
        cnn_vector = keras.layers.GlobalAveragePooling2D()(cnn_map)
        vit_vector = vit_tokens[:, 0, :]
        features = keras.layers.Concatenate()([cnn_vector, vit_vector])
    else:
        cnn_tokens = keras.layers.Reshape((-1, cnn_map.shape[-1]))(cnn_map)
        cnn_tokens = keras.layers.Dense(TOKEN_SIZE)(cnn_tokens)
        vit_projected = keras.layers.Dense(TOKEN_SIZE)(vit_tokens)
        tokens = keras.layers.Concatenate(axis=1)([cnn_tokens, vit_projected])
        x = keras.layers.LayerNormalization()(tokens)
        attention = keras.layers.MultiHeadAttention(ATTENTION_HEADS, TOKEN_SIZE // ATTENTION_HEADS)(x, x)
        tokens = tokens + attention
        x = keras.layers.LayerNormalization()(tokens)
        x = keras.layers.Dense(2 * TOKEN_SIZE, activation="gelu")(x)
        tokens = tokens + keras.layers.Dense(TOKEN_SIZE)(x)
        features = keras.layers.GlobalAveragePooling1D()(keras.layers.LayerNormalization()(tokens))

    features = keras.layers.Dropout(DROPOUT)(features)
    outputs = keras.layers.Dense(2, activation="softmax", dtype="float32", name="fire_output")(features)
    return keras.Model(inputs, outputs, name=options.model)


def add_synthetic(parts, test_sets, options, log):
    """Add copy-paste images to the training part and remove test images that share a group with them."""
    folder = Path(options.synthetic).resolve()
    with open(folder / "synthetic_images.csv", newline="", encoding="utf-8") as f:
        synthetic = list(csv.DictReader(f))

    rng = random.Random(options.seed)
    rng.shuffle(synthetic)
    wanted = min(len(synthetic), int(options.synthetic_share * len(parts["train"])))
    if options.quick:
        wanted = min(wanted, 50)
    chosen = synthetic[:wanted]

    used_clusters = set()
    for row in chosen:
        parts["train"].append({"image": str(folder / row["image"]), "dataset": "SYN", "cluster": "syn",
                               "split": "train", "label": base.LABELS[row["l2"]]})
        used_clusters.add(row["background_cluster"])
        used_clusters.update(row["piece_clusters"].split())

    removed = 0
    for target in list(test_sets):
        kept = [r for r in test_sets[target] if r["cluster"] not in used_clusters]
        removed += len(test_sets[target]) - len(kept)
        test_sets[target] = kept
    log(f"  synthetic: {len(chosen)} images added to training; "
        f"{removed} test images removed (same near-duplicate group as a synthetic image's source)")
    return len(chosen), removed


def run_name(options):
    mode = "frozen" if options.freeze else "finetuned"
    name = f"within_{options.dataset}_{options.model}_{mode}_seed{options.seed}"
    if options.mixstyle:
        name += "_mixstyle"
    if options.synthetic:
        name += "_syn"
    name += options.tag
    if options.quick:
        name += "_QUICK"
    return name


def main():
    options = read_options()
    keras.utils.set_random_seed(options.seed)
    device, use_mixed = base.set_up_gpu(options)

    out_folder = Path(options.out) / run_name(options)
    out_folder.mkdir(parents=True, exist_ok=True)
    log = base.Log(out_folder / "log.txt")
    log(f"Run {out_folder.name} on {device}, TensorFlow {tf.__version__}, Keras {keras.__version__}")

    parts, test_sets = base.choose_images(base.read_rows(options.data), options)
    synthetic_added, synthetic_removed = 0, 0
    if options.synthetic:
        synthetic_added, synthetic_removed = add_synthetic(parts, test_sets, options, log)
    for name, rows in parts.items():
        fire = sum(r["label"] for r in rows)
        log(f"  {name}: {len(rows)} images, {fire} fire_or_smoke, {len(rows) - fire} none")
    log("  test sets: " + ", ".join(f"{d} {len(rows)}" for d, rows in test_sets.items()))

    settings = vars(options).copy()
    settings.update({"device": device, "mixed_precision": use_mixed, "tensorflow": tf.__version__,
                     "keras": keras.__version__, "synthetic_added": synthetic_added,
                     "synthetic_test_images_removed": synthetic_removed,
                     "images": {name: len(rows) for name, rows in parts.items()},
                     "test_images": {d: len(rows) for d, rows in test_sets.items()}})
    base.write_json(out_folder / "settings.json", settings)

    model = make_hybrid(options)
    total_weights, trainable_weights = base.count_weights(model)
    log(f"  model {options.model}{' + MixStyle' if options.mixstyle else ''}: "
        f"{total_weights:,} weights, {trainable_weights:,} of them learn")
    model.compile(optimizer=keras.optimizers.AdamW(learning_rate=options.lr),
                  loss="sparse_categorical_crossentropy")
    train_data = base.make_dataset(parts["train"], options, training=True)
    weights = base.class_weights(parts["train"])

    best_score, best_weights, best_epoch = -1.0, None, 0
    history = []
    train_start = time.time()
    for epoch in range(1, options.epochs + 1):
        start = time.time()
        fit = model.fit(train_data, epochs=1, class_weight=weights, shuffle=False, verbose=0)
        train_loss = float(fit.history["loss"][0])
        val = base.scores(*base.predict(model, parts["val"], options))
        minutes = (time.time() - start) / 60
        history.append({"epoch": epoch, "train_loss": round(train_loss, 5),
                        "val_accuracy": round(val["accuracy"], 5), "val_macro_f1": round(val["macro_f1"], 5),
                        "val_balanced_accuracy": val["balanced_accuracy"], "val_auc": val["auc"],
                        "minutes": round(minutes, 2)})
        log(f"  epoch {epoch}/{options.epochs}: loss {train_loss:.4f}, val acc {val['accuracy']:.4f}, "
            f"val macro-F1 {val['macro_f1']:.4f}, {minutes:.1f} min")
        if val["macro_f1"] > best_score:
            best_score, best_epoch = val["macro_f1"], epoch
            best_weights = model.get_weights()
    train_minutes = (time.time() - train_start) / 60

    with open(out_folder / "history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    log(f"  best epoch {best_epoch} (val macro-F1 {best_score:.4f}); testing it")
    model.set_weights(best_weights)
    if not options.no_save:
        model.save_weights(out_folder / "best.weights.h5")

    results = {"train_dataset": options.dataset, "model": options.model, "cnn": options.cnn,
               "fusion": options.fusion,
               "mixstyle": options.mixstyle, "synthetic": bool(options.synthetic),
               "mode": "frozen" if options.freeze else "finetuned",
               "best_epoch": best_epoch, "best_val_macro_f1": best_score,
               "train_minutes": round(train_minutes, 2), "test": {}}
    prediction_rows = []
    for target, rows in test_sets.items():
        labels, probs = base.predict(model, rows, options)
        results["test"][target] = base.scores(labels, probs)
        for row, prob in zip(rows, probs):
            prediction_rows.append([target, row["image"], row["label"], round(float(prob), 5)])

    results["weights_total"] = total_weights
    results["weights_trainable"] = trainable_weights
    results.update(base.measure_speed(model, options.image_size))
    base.write_json(out_folder / "test_metrics.json", results)

    with open(out_folder / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["test_dataset", "image", "label", "fire_probability"])
        writer.writerows(prediction_rows)

    log("TEST (first line = own test set, the others = cross-dataset):")
    for target, s in results["test"].items():
        log(f"  {target:13s} {s['images']:6d} images: acc {s['accuracy']:.4f}, "
            f"macro-F1 {s['macro_f1']:.4f}, bal. acc {base.short(s['balanced_accuracy'])}, "
            f"precision {s['precision']:.4f}, recall {s['recall']:.4f}")
    log(f"  speed: {results['ms_per_image_batch32']} ms/image in batches of {base.SPEED_BATCH}; "
        f"training {train_minutes:.1f} min")
    log(f"Results in {out_folder}")


if __name__ == "__main__":
    main()
