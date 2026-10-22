"""Task 1 + 2 (Keras + TensorFlow): 40 baseline runs with cross-dataset tests.

Runs train_tf.py for 5 models x 4 training datasets x frozen/fine-tuned, skipping
finished runs, and writes all_scores.csv with one row per (run, test dataset).

Usage:
    python 04_experiments/src/run_task1_tf.py --data UPLOAD_FOLDER [--out results_tf]
        [--models NAME ...] [--quick] [--dry-run]
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


DATASETS = ["R1", "R4", "R3", "R2"]

MODELS = ["resnet50", "mobilenet_v2", "efficientnet_b3", "densenet121", "vit_b16"]

TRAIN_SCRIPT = Path(__file__).resolve().parent / "train_tf.py"

SCORE_COLUMNS = ["images", "fire_images", "accuracy", "balanced_accuracy", "macro_f1", "precision",
                 "recall", "f1", "specificity", "auc",
                 "true_negatives", "false_positives", "false_negatives", "true_positives"]


def read_options():
    parser = argparse.ArgumentParser(description="Task 1 + 2: 40 baseline runs with cross-dataset tests.")
    parser.add_argument("--data", required=True, help="the unzipped upload folder")
    parser.add_argument("--out", default="results_tf", help="folder for the run folders and all_scores.csv")
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS, help="only these models")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true",
                        help="check only: every model once, frozen and fine-tuned, on R1, with train_tf.py --quick")
    parser.add_argument("--dry-run", action="store_true", help="only print the list of runs")
    return parser.parse_args()


def make_run_list(models, quick):
    """Each run is (model, dataset, frozen yes/no)."""
    datasets = ["R1"] if quick else DATASETS
    runs = []
    for frozen in (True, False):
        for dataset in datasets:
            for model in models:
                runs.append((model, dataset, frozen))
    return runs


def folder_name(model, dataset, frozen, seed, quick):
    """The folder train_tf.py writes this run to, e.g. within_R1_resnet50_frozen_seed42."""
    mode = "frozen" if frozen else "finetuned"
    name = f"within_{dataset}_{model}_{mode}_seed{seed}"
    if quick:
        name += "_QUICK"
    return name


def run_all(runs, options, log):
    for number, (model, dataset, frozen) in enumerate(runs, 1):
        name = folder_name(model, dataset, frozen, options.seed, options.quick)
        if (Path(options.out) / name / "test_metrics.json").exists():
            log(f"[{number}/{len(runs)}] {name}: already done, skipped")
            continue
        command = [sys.executable, str(TRAIN_SCRIPT), "--data", options.data, "--out", options.out,
                   "--dataset", dataset, "--model", model, "--epochs", str(options.epochs),
                   "--batch-size", str(options.batch_size), "--seed", str(options.seed)]
        if frozen:
            command.append("--freeze")
        if options.quick:
            command.append("--quick")
        log(f"[{number}/{len(runs)}] {name}: started")
        start = time.time()
        result = subprocess.run(command)
        minutes = (time.time() - start) / 60
        status = "finished" if result.returncode == 0 else f"FAILED (exit code {result.returncode})"
        log(f"[{number}/{len(runs)}] {name}: {status} after {minutes:.0f} min")


def write_scores(out_folder):
    """One row per (finished run, test dataset)."""
    rows = []
    for metrics_path in sorted(Path(out_folder).glob("within_*/test_metrics.json")):
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)
        for target, s in metrics["test"].items():
            row = {"run": metrics_path.parent.name, "model": metrics["model"], "mode": metrics["mode"],
                   "train_dataset": metrics["train_dataset"], "test_dataset": target,
                   "kind": "within" if target == metrics["train_dataset"] else "cross",
                   "best_epoch": metrics["best_epoch"], "train_minutes": metrics["train_minutes"],
                   "ms_per_image": metrics["ms_per_image_batch32"], "weights": metrics["weights_total"]}
            for column in SCORE_COLUMNS:
                value = s[column]
                row[column] = round(value, 4) if isinstance(value, float) else value
            rows.append(row)
    path = Path(out_folder) / "all_scores.csv"
    if rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return path, len(rows)


def main():
    options = read_options()
    runs = make_run_list(options.models, options.quick)
    if options.dry_run:
        for number, (model, dataset, frozen) in enumerate(runs, 1):
            print(f"{number}. {folder_name(model, dataset, frozen, options.seed, options.quick)}")
        return

    Path(options.out).mkdir(parents=True, exist_ok=True)
    log_file = open(Path(options.out) / "run_log.txt", "a", encoding="utf-8")

    def log(text):
        line = f"{time.strftime('%Y-%m-%d %H:%M')}  {text}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    log(f"Task 1: starting {len(runs)} runs")
    run_all(runs, options, log)
    path, n = write_scores(options.out)
    log(f"Wrote {path} ({n} rows)")


if __name__ == "__main__":
    main()
