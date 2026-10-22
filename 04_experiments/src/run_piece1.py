"""Piece 1: ResNet-50 on R1, frozen and fine-tuned (first GPU check).

Runs train_tf.py twice, skipping finished runs, and writes piece1_scores.csv
with one row per (run, test dataset).

Usage:
    python 04_experiments/src/run_piece1.py --data UPLOAD_FOLDER [--out results_tf] [--quick]
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


MODEL = "resnet50"
DATASET = "R1"

TRAIN_SCRIPT = Path(__file__).resolve().parent / "train_tf.py"

SCORE_COLUMNS = ["images", "fire_images", "accuracy", "balanced_accuracy", "macro_f1", "precision",
                 "recall", "f1", "specificity", "auc",
                 "true_negatives", "false_positives", "false_negatives", "true_positives"]


def read_options():
    parser = argparse.ArgumentParser(description="Piece 1: ResNet-50 on R1, frozen and fine-tuned.")
    parser.add_argument("--data", required=True, help="the unzipped upload folder")
    parser.add_argument("--out", default="results_tf", help="folder for the run folders and piece1_scores.csv")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--quick", action="store_true", help="check only: tiny runs (train_tf.py --quick)")
    return parser.parse_args()


def folder_name(frozen, quick):
    """The folder train_tf.py writes this run to, e.g. within_R1_resnet50_frozen_seed42."""
    mode = "frozen" if frozen else "finetuned"
    name = f"within_{DATASET}_{MODEL}_{mode}_seed42"
    if quick:
        name += "_QUICK"
    return name


def run_both(options, log):
    for number, frozen in enumerate((True, False), 1):
        name = folder_name(frozen, options.quick)
        if (Path(options.out) / name / "test_metrics.json").exists():
            log(f"[{number}/2] {name}: already done, skipped")
            continue
        command = [sys.executable, str(TRAIN_SCRIPT), "--data", options.data, "--out", options.out,
                   "--dataset", DATASET, "--model", MODEL, "--batch-size", str(options.batch_size)]
        if frozen:
            command.append("--freeze")
        if options.quick:
            command.append("--quick")
        log(f"[{number}/2] {name}: started")
        start = time.time()
        result = subprocess.run(command)
        minutes = (time.time() - start) / 60
        status = "finished" if result.returncode == 0 else f"FAILED (exit code {result.returncode})"
        log(f"[{number}/2] {name}: {status} after {minutes:.0f} min")


def write_scores(options):
    """One row per (finished run, test dataset)."""
    rows = []
    for frozen in (True, False):
        metrics_path = Path(options.out) / folder_name(frozen, options.quick) / "test_metrics.json"
        if not metrics_path.exists():
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
    path = Path(options.out) / "piece1_scores.csv"
    if rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return path, len(rows)


def main():
    options = read_options()
    Path(options.out).mkdir(parents=True, exist_ok=True)
    log_file = open(Path(options.out) / "run_log.txt", "a", encoding="utf-8")

    def log(text):
        line = f"{time.strftime('%Y-%m-%d %H:%M')}  {text}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    log("Piece 1: ResNet-50 on R1, frozen and fine-tuned")
    run_both(options, log)
    path, n = write_scores(options)
    log(f"Wrote {path} ({n} rows)")


if __name__ == "__main__":
    main()
