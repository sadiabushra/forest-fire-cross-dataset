"""Part 1: within-dataset baselines.

Trains and tests ResNet-50 (frozen and fine-tuned) on each training dataset
(R1-R4) with train.py, skipping finished runs, and writes summary.csv.

Usage:
    python 04_experiments/src/run_part1_within.py --data UPLOAD_FOLDER [--out results] [--dry-run]
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


DATASETS = ["R1", "R2", "R3", "R4"]

MODEL = "resnet50"

TRAIN_SCRIPT = Path(__file__).resolve().parent / "train.py"


def read_options():
    parser = argparse.ArgumentParser(description="Part 1: within-dataset baselines.")
    parser.add_argument("--data", required=True, help="the unzipped upload folder")
    parser.add_argument("--out", default="results", help="folder for the run folders and summary.csv")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="only print the list of runs")
    return parser.parse_args()


def make_run_list():
    """Each run is (dataset, frozen yes/no)."""
    runs = []
    for frozen in (True, False):
        for dataset in DATASETS:
            runs.append((dataset, frozen))
    return runs


def folder_name(dataset, frozen, seed):
    """The folder train.py writes this run to, e.g. within_R1_resnet50_frozen_seed42."""
    mode = "frozen" if frozen else "finetuned"
    return f"within_{dataset}_{MODEL}_{mode}_seed{seed}"


def run_all(runs, options, log):
    for number, (dataset, frozen) in enumerate(runs, 1):
        name = folder_name(dataset, frozen, options.seed)
        if (Path(options.out) / name / "test_metrics.json").exists():
            log(f"[{number}/{len(runs)}] {name}: already done, skipped")
            continue
        command = [sys.executable, str(TRAIN_SCRIPT), "--data", options.data, "--out", options.out,
                   "--dataset", dataset, "--model", MODEL,
                   "--epochs", str(options.epochs), "--batch-size", str(options.batch_size),
                   "--workers", str(options.workers), "--seed", str(options.seed)]
        if frozen:
            command.append("--freeze")
        log(f"[{number}/{len(runs)}] {name}: started")
        start = time.time()
        result = subprocess.run(command)
        minutes = (time.time() - start) / 60
        status = "finished" if result.returncode == 0 else f"FAILED (exit code {result.returncode})"
        log(f"[{number}/{len(runs)}] {name}: {status} after {minutes:.0f} min")


def write_summary(out_folder):
    """One row per finished run, with its test scores."""
    rows = []
    for metrics_path in sorted(Path(out_folder).glob("within_*/test_metrics.json")):
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)
        with open(metrics_path.parent / "settings.json", encoding="utf-8") as f:
            settings = json.load(f)
        s = metrics["overall"]
        rows.append({
            "run": metrics_path.parent.name,
            "dataset": settings["dataset"],
            "model": settings["model"],
            "mode": "frozen" if settings["freeze"] else "finetuned",
            "best_epoch": metrics["best_epoch"],
            "test_images": s["images"],
            "accuracy": round(s["accuracy"], 4),
            "precision": round(s["precision"], 4),
            "recall": round(s["recall"], 4),
            "f1": round(s["f1"], 4),
            "auc": round(s["auc"], 4) if s["auc"] is not None else "",
        })
    summary_path = Path(out_folder) / "summary.csv"
    if rows:
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return summary_path, len(rows)


def main():
    options = read_options()
    runs = make_run_list()
    if options.dry_run:
        for number, (dataset, frozen) in enumerate(runs, 1):
            print(f"{number}. {folder_name(dataset, frozen, options.seed)}")
        return

    Path(options.out).mkdir(parents=True, exist_ok=True)
    log_file = open(Path(options.out) / "run_log.txt", "a", encoding="utf-8")

    def log(text):
        line = f"{time.strftime('%Y-%m-%d %H:%M')}  {text}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    log(f"Part 1: starting {len(runs)} runs")
    run_all(runs, options, log)
    summary_path, n = write_summary(options.out)
    log(f"Wrote {summary_path} ({n} runs)")


if __name__ == "__main__":
    main()
