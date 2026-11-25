"""Piece 7: ViT-B/16 and DenseNet-121 on R1-R4, fine-tuned, extra seeds (1 and 2).

Runs train_tf.py for every seed x model x dataset, skipping finished runs, and
writes <model>_scores.csv (all seeds) and <model>_seed_summary.csv for both models.

Usage:
    python 04_experiments/src/run_piece7.py --data UPLOAD_FOLDER [--out results_tf]
        [--models vit_b16 densenet121] [--seeds 1 2] [--datasets R1 R4 R3 R2]
        [--batch-size N] [--quick]
"""

import argparse
import csv
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------

MODELS = ["vit_b16", "densenet121"]

# Seed 42 was done in pieces 5 and 6; these are the two extra seeds.
SEEDS = [1, 2]

# Smallest first, so results arrive early.
DATASETS = ["R1", "R4", "R3", "R2"]

TRAIN_SCRIPT = Path(__file__).resolve().parent / "train_tf.py"

# Columns of the score tables taken from each test result.
SCORE_COLUMNS = ["images", "fire_images", "accuracy", "balanced_accuracy", "macro_f1", "precision",
                 "recall", "f1", "specificity", "auc",
                 "true_negatives", "false_positives", "false_negatives", "true_positives"]

# Scores that get a mean and spread over the seeds.
SUMMARY_SCORES = ["macro_f1", "recall", "specificity", "auc"]


def read_options():
    parser = argparse.ArgumentParser(description="Piece 7: ViT-B/16 and DenseNet-121 with extra seeds.")
    parser.add_argument("--data", required=True, help="the unzipped upload folder")
    parser.add_argument("--out", default="results_tf", help="folder for the run folders and the score tables")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--quick", action="store_true", help="check only: tiny runs (train_tf.py --quick)")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS, help="seeds to run (default: 1 2)")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS,
                        help="only these training datasets (default: R1 R4 R3 R2)")
    return parser.parse_args()


def make_run_list(seeds, models, datasets):
    """Each run is (seed, model, dataset)."""
    runs = []
    for seed in seeds:
        for model in models:
            for dataset in datasets:
                runs.append((seed, model, dataset))
    return runs


def folder_name(seed, model, dataset, quick):
    """The folder train_tf.py writes this run to, e.g. within_R4_vit_b16_finetuned_seed1."""
    name = f"within_{dataset}_{model}_finetuned_seed{seed}"
    if quick:
        name += "_QUICK"
    return name


# ---------------------------------------------------------------------------
# 1. The runs
# ---------------------------------------------------------------------------

def run_all(runs, options, log):
    for number, (seed, model, dataset) in enumerate(runs, 1):
        name = folder_name(seed, model, dataset, options.quick)
        if (Path(options.out) / name / "test_metrics.json").exists():
            log(f"[{number}/{len(runs)}] {name}: already done, skipped")
            continue
        # No --freeze: all runs are fine-tuned
        command = [sys.executable, str(TRAIN_SCRIPT), "--data", options.data, "--out", options.out,
                   "--dataset", dataset, "--model", model, "--batch-size", str(options.batch_size),
                   "--seed", str(seed)]
        if options.quick:
            command.append("--quick")
        log(f"[{number}/{len(runs)}] {name}: started")
        start = time.time()
        # Each run is its own process, so the GPU memory is freed after it,
        # and one failed run does not stop the others.
        result = subprocess.run(command)
        minutes = (time.time() - start) / 60
        status = "finished" if result.returncode == 0 else f"FAILED (exit code {result.returncode})"
        log(f"[{number}/{len(runs)}] {name}: {status} after {minutes:.0f} min")


# ---------------------------------------------------------------------------
# 2. One table per model with all its scores (every seed and mode)
# ---------------------------------------------------------------------------

def read_scores(model, options):
    """One row per (finished run of this model, test dataset)."""
    pattern = f"within_*_{model}_*/test_metrics.json"
    rows = []
    for metrics_path in sorted(Path(options.out).glob(pattern)):
        run = metrics_path.parent.name
        # A quick check and a real run never end up in the same table.
        if run.endswith("_QUICK") != options.quick:
            continue
        # Search results have a tag after the seed (e.g. _aug-none); they are not baselines.
        if not re.search(r"_seed\d+(_QUICK)?$", run):
            continue
        seed = int(re.search(r"_seed(\d+)", run).group(1))
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)
        for target, s in metrics["test"].items():
            row = {"run": run, "model": metrics["model"], "mode": metrics["mode"], "seed": seed,
                   "train_dataset": metrics["train_dataset"], "test_dataset": target,
                   "kind": "within" if target == metrics["train_dataset"] else "cross",
                   "best_epoch": metrics["best_epoch"], "train_minutes": metrics["train_minutes"],
                   "ms_per_image": metrics["ms_per_image_batch32"], "weights": metrics["weights_total"]}
            for column in SCORE_COLUMNS:
                value = s[column]
                row[column] = round(value, 4) if isinstance(value, float) else value
            rows.append(row)
    return rows


def summarize_seeds(model, rows):
    """Mean, standard deviation, min and max over the seeds, per (mode, train dataset, test dataset)."""
    groups = {}
    for row in rows:
        key = (row["mode"], row["train_dataset"], row["test_dataset"], row["kind"])
        groups.setdefault(key, []).append(row)

    summary = []
    for (mode, train, test, kind), group in sorted(groups.items()):
        line = {"model": model, "mode": mode, "train_dataset": train, "test_dataset": test, "kind": kind,
                "seeds": " ".join(str(r["seed"]) for r in sorted(group, key=lambda r: r["seed"]))}
        for score in SUMMARY_SCORES:
            # Scores can be empty (None) when a test set has only one class.
            values = [r[score] for r in group if r[score] is not None]
            if not values:
                continue
            line[f"{score}_mean"] = round(statistics.mean(values), 4)
            # The standard deviation needs at least two seeds.
            line[f"{score}_std"] = round(statistics.stdev(values), 4) if len(values) > 1 else ""
            line[f"{score}_min"] = min(values)
            line[f"{score}_max"] = max(values)
        summary.append(line)
    return summary


def write_csv(path, rows):
    # Columns from all rows, in first-seen order (a summary row can miss a score).
    columns = []
    for row in rows:
        for column in row:
            if column not in columns:
                columns.append(column)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_tables(options, log):
    for model in options.models:
        rows = read_scores(model, options)
        if not rows:
            continue
        write_csv(Path(options.out) / f"{model}_scores.csv", rows)
        write_csv(Path(options.out) / f"{model}_seed_summary.csv", summarize_seeds(model, rows))
        log(f"Wrote {model}_scores.csv ({len(rows)} rows) and {model}_seed_summary.csv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    options = read_options()
    runs = make_run_list(options.seeds, options.models, options.datasets)
    Path(options.out).mkdir(parents=True, exist_ok=True)
    log_file = open(Path(options.out) / "run_log.txt", "a", encoding="utf-8")

    def log(text):
        line = f"{time.strftime('%Y-%m-%d %H:%M')}  {text}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    log(f"Piece 7: {', '.join(options.models)} on {', '.join(options.datasets)}, seeds "
        f"{', '.join(map(str, options.seeds))}, fine-tuned ({len(runs)} runs)")
    run_all(runs, options, log)
    write_tables(options, log)


if __name__ == "__main__":
    main()
