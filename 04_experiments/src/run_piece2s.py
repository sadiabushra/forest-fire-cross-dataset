"""Piece 2S: ResNet-50 on R1-R4, frozen and fine-tuned, with extra seeds (1 and 2).

Runs train_tf.py for every seed x dataset x mode, skipping finished runs, and
writes resnet50_scores.csv (all seeds, one row per run and test dataset) and
resnet50_seed_summary.csv (mean, std, min, max over the seeds).

Usage:
    python 04_experiments/src/run_piece2s.py --data UPLOAD_FOLDER [--out results_tf]
        [--seeds 1 2] [--datasets R1 R4 R3 R2] [--quick]
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


MODEL = "resnet50"

SEEDS = [1, 2]

DATASETS = ["R1", "R4", "R3", "R2"]

TRAIN_SCRIPT = Path(__file__).resolve().parent / "train_tf.py"

SCORE_COLUMNS = ["images", "fire_images", "accuracy", "balanced_accuracy", "macro_f1", "precision",
                 "recall", "f1", "specificity", "auc",
                 "true_negatives", "false_positives", "false_negatives", "true_positives"]

SUMMARY_SCORES = ["macro_f1", "recall", "specificity", "auc"]


def read_options():
    parser = argparse.ArgumentParser(description="Piece 2S: ResNet-50 on R1-R4 with extra seeds.")
    parser.add_argument("--data", required=True, help="the unzipped upload folder")
    parser.add_argument("--out", default="results_tf", help="folder for the run folders and the score tables")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--quick", action="store_true", help="check only: tiny runs (train_tf.py --quick)")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS, help="seeds to run (default: 1 2)")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS,
                        help="only these training datasets (default: R1 R4 R3 R2)")
    return parser.parse_args()


def make_run_list(seeds, datasets):
    """Each run is (seed, dataset, frozen yes/no)."""
    runs = []
    for seed in seeds:
        for frozen in (True, False):
            for dataset in datasets:
                runs.append((seed, dataset, frozen))
    return runs


def folder_name(seed, dataset, frozen, quick):
    """The folder train_tf.py writes this run to, e.g. within_R4_resnet50_frozen_seed1."""
    mode = "frozen" if frozen else "finetuned"
    name = f"within_{dataset}_{MODEL}_{mode}_seed{seed}"
    if quick:
        name += "_QUICK"
    return name


def run_all(runs, options, log):
    for number, (seed, dataset, frozen) in enumerate(runs, 1):
        name = folder_name(seed, dataset, frozen, options.quick)
        if (Path(options.out) / name / "test_metrics.json").exists():
            log(f"[{number}/{len(runs)}] {name}: already done, skipped")
            continue
        command = [sys.executable, str(TRAIN_SCRIPT), "--data", options.data, "--out", options.out,
                   "--dataset", dataset, "--model", MODEL, "--batch-size", str(options.batch_size),
                   "--seed", str(seed)]
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


def read_scores(options):
    """One row per (finished ResNet-50 run, test dataset)."""
    pattern = f"within_*_{MODEL}_*/test_metrics.json"
    rows = []
    for metrics_path in sorted(Path(options.out).glob(pattern)):
        run = metrics_path.parent.name
        if run.endswith("_QUICK") != options.quick:
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


def summarize_seeds(rows):
    """Mean, standard deviation, min and max over the seeds, per (mode, train dataset, test dataset)."""
    groups = {}
    for row in rows:
        key = (row["mode"], row["train_dataset"], row["test_dataset"], row["kind"])
        groups.setdefault(key, []).append(row)

    summary = []
    for (mode, train, test, kind), group in sorted(groups.items()):
        line = {"model": MODEL, "mode": mode, "train_dataset": train, "test_dataset": test, "kind": kind,
                "seeds": " ".join(str(r["seed"]) for r in sorted(group, key=lambda r: r["seed"]))}
        for score in SUMMARY_SCORES:
            values = [r[score] for r in group if r[score] is not None]
            if not values:
                continue
            line[f"{score}_mean"] = round(statistics.mean(values), 4)
            line[f"{score}_std"] = round(statistics.stdev(values), 4) if len(values) > 1 else ""
            line[f"{score}_min"] = min(values)
            line[f"{score}_max"] = max(values)
        summary.append(line)
    return summary


def write_csv(path, rows):
    columns = []
    for row in rows:
        for column in row:
            if column not in columns:
                columns.append(column)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_tables(options):
    rows = read_scores(options)
    if not rows:
        return 0
    write_csv(Path(options.out) / f"{MODEL}_scores.csv", rows)
    write_csv(Path(options.out) / f"{MODEL}_seed_summary.csv", summarize_seeds(rows))
    return len(rows)


def main():
    options = read_options()
    runs = make_run_list(options.seeds, options.datasets)
    Path(options.out).mkdir(parents=True, exist_ok=True)
    log_file = open(Path(options.out) / "run_log.txt", "a", encoding="utf-8")

    def log(text):
        line = f"{time.strftime('%Y-%m-%d %H:%M')}  {text}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    log(f"Piece 2S: ResNet-50 on {', '.join(options.datasets)}, seeds "
        f"{', '.join(map(str, options.seeds))}, frozen and fine-tuned ({len(runs)} runs)")
    run_all(runs, options, log)
    n = write_tables(options)
    log(f"Wrote {MODEL}_scores.csv ({n} rows) and {MODEL}_seed_summary.csv in {options.out}")


if __name__ == "__main__":
    main()
