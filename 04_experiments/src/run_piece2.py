"""Piece 2: ResNet-50 on R2, R3 and R4, frozen and fine-tuned.

Runs train_tf.py six times, skipping finished runs, and writes
resnet50_scores.csv with every finished ResNet-50 run (pieces 1 and 2),
one row per (run, test dataset).

Usage:
    python 04_experiments/src/run_piece2.py --data UPLOAD_FOLDER [--out results_tf] [--quick]
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


MODEL = "resnet50"

DATASETS = ["R4", "R3", "R2"]

TRAIN_SCRIPT = Path(__file__).resolve().parent / "train_tf.py"

SCORE_COLUMNS = ["images", "fire_images", "accuracy", "balanced_accuracy", "macro_f1", "precision",
                 "recall", "f1", "specificity", "auc",
                 "true_negatives", "false_positives", "false_negatives", "true_positives"]


def read_options():
    parser = argparse.ArgumentParser(description="Piece 2: ResNet-50 on R2, R3, R4, frozen and fine-tuned.")
    parser.add_argument("--data", required=True, help="the unzipped upload folder")
    parser.add_argument("--out", default="results_tf", help="folder for the run folders and resnet50_scores.csv")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--quick", action="store_true", help="check only: tiny runs (train_tf.py --quick)")
    return parser.parse_args()


def make_run_list():
    """Each run is (dataset, frozen yes/no)."""
    runs = []
    for frozen in (True, False):
        for dataset in DATASETS:
            runs.append((dataset, frozen))
    return runs


def folder_name(dataset, frozen, quick):
    """The folder train_tf.py writes this run to, e.g. within_R4_resnet50_frozen_seed42."""
    mode = "frozen" if frozen else "finetuned"
    name = f"within_{dataset}_{MODEL}_{mode}_seed42"
    if quick:
        name += "_QUICK"
    return name


def run_all(runs, options, log):
    for number, (dataset, frozen) in enumerate(runs, 1):
        name = folder_name(dataset, frozen, options.quick)
        if (Path(options.out) / name / "test_metrics.json").exists():
            log(f"[{number}/{len(runs)}] {name}: already done, skipped")
            continue
        command = [sys.executable, str(TRAIN_SCRIPT), "--data", options.data, "--out", options.out,
                   "--dataset", dataset, "--model", MODEL, "--batch-size", str(options.batch_size)]
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


def write_scores(options):
    """One row per (finished ResNet-50 run, test dataset)."""
    pattern = f"within_*_{MODEL}_*/test_metrics.json"
    rows = []
    for metrics_path in sorted(Path(options.out).glob(pattern)):
        if metrics_path.parent.name.endswith("_QUICK") != options.quick:
            continue
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
    path = Path(options.out) / f"{MODEL}_scores.csv"
    if rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return path, len(rows)


def main():
    options = read_options()
    runs = make_run_list()
    Path(options.out).mkdir(parents=True, exist_ok=True)
    log_file = open(Path(options.out) / "run_log.txt", "a", encoding="utf-8")

    def log(text):
        line = f"{time.strftime('%Y-%m-%d %H:%M')}  {text}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    log(f"Piece 2: ResNet-50 on {', '.join(DATASETS)}, frozen and fine-tuned ({len(runs)} runs)")
    run_all(runs, options, log)
    path, n = write_scores(options)
    log(f"Wrote {path} ({n} rows)")


if __name__ == "__main__":
    main()
