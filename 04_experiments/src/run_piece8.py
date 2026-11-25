"""Piece 8: the hybrid model, its parts added one at a time.

Runs train_hybrid_tf.py for every seed x dataset x variant (e.g. concat,
concat+mixstyle, concat+mixstyle+syn), skipping finished runs, and writes
hybrid_scores.csv and hybrid_seed_summary.csv.

Usage:
    python 04_experiments/src/run_piece8.py --data UPLOAD_FOLDER --variants V [V ...]
        [--cnn densenet121|mobilenet_v2] [--datasets R4 R2] [--seeds 42 1 2]
        [--synthetic FOLDER] [--out results_tf] [--quick]
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


TRAIN_SCRIPT = Path(__file__).resolve().parent / "train_hybrid_tf.py"
DATASETS = ["R1", "R2", "R3", "R4"]

SCORE_COLUMNS = ["images", "fire_images", "accuracy", "balanced_accuracy", "macro_f1", "precision",
                 "recall", "f1", "specificity", "auc",
                 "true_negatives", "false_positives", "false_negatives", "true_positives"]
SUMMARY_SCORES = ["macro_f1", "recall", "specificity", "auc"]


def read_options():
    parser = argparse.ArgumentParser(description="Piece 8: hybrid model, parts added one at a time.")
    parser.add_argument("--data", required=True, help="the unzipped upload folder")
    parser.add_argument("--out", default="results_tf")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=["R4"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--variants", nargs="+", required=True,
                        help="e.g. concat transformer concat+mixstyle concat+mixstyle+syn")
    parser.add_argument("--cnn", default="densenet121", choices=["densenet121", "mobilenet_v2"],
                        help="the CNN branch (default DenseNet-121)")
    parser.add_argument("--synthetic", default="", help="folder with synthetic_images.csv (needed for +syn)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def parse_variant(variant):
    """'concat+mixstyle+syn' -> ('concat', True, True)."""
    parts = variant.split("+")
    if parts[0] not in ("concat", "transformer") or not set(parts[1:]) <= {"mixstyle", "syn"}:
        sys.exit(f"Unknown variant {variant}")
    return parts[0], "mixstyle" in parts, "syn" in parts


def folder_name(dataset, cnn, fusion, mixstyle, syn, seed, quick):
    """Same name as train_hybrid_tf.run_name."""
    name = f"within_{dataset}_hybrid-{cnn}-{fusion}_finetuned_seed{seed}"
    if mixstyle:
        name += "_mixstyle"
    if syn:
        name += "_syn"
    if quick:
        name += "_QUICK"
    return name


def run_all(options, log):
    runs = [(seed, dataset, variant) for seed in options.seeds
            for dataset in options.datasets for variant in options.variants]
    for number, (seed, dataset, variant) in enumerate(runs, 1):
        fusion, mixstyle, syn = parse_variant(variant)
        name = folder_name(dataset, options.cnn, fusion, mixstyle, syn, seed, options.quick)
        if (Path(options.out) / name / "test_metrics.json").exists():
            log(f"[{number}/{len(runs)}] {name}: already done, skipped")
            continue
        if syn and not options.synthetic:
            log(f"[{number}/{len(runs)}] {name}: needs --synthetic, skipped")
            continue
        command = [sys.executable, str(TRAIN_SCRIPT), "--data", options.data, "--out", options.out,
                   "--dataset", dataset, "--cnn", options.cnn, "--fusion", fusion, "--seed", str(seed),
                   "--batch-size", str(options.batch_size)]
        if mixstyle:
            command.append("--mixstyle")
        if syn:
            command += ["--synthetic", options.synthetic]
        if options.quick:
            command.append("--quick")
        log(f"[{number}/{len(runs)}] {name}: started")
        start = time.time()
        result = subprocess.run(command)
        minutes = (time.time() - start) / 60
        status = "finished" if result.returncode == 0 else f"FAILED (exit code {result.returncode})"
        log(f"[{number}/{len(runs)}] {name}: {status} after {minutes:.0f} min")


def read_scores(options):
    rows = []
    for metrics_path in sorted(Path(options.out).glob("within_*_hybrid-*/test_metrics.json")):
        run = metrics_path.parent.name
        if run.endswith("_QUICK") != options.quick:
            continue
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)
        seed = int(re.search(r"_seed(\d+)", run).group(1))
        cnn = metrics.get("cnn", "mobilenet_v2")
        variant = cnn + "-" + metrics["fusion"] + ("+mixstyle" if metrics["mixstyle"] else "") + \
            ("+syn" if metrics["synthetic"] else "")
        for target, s in metrics["test"].items():
            row = {"run": run, "variant": variant, "cnn": cnn, "fusion": metrics["fusion"],
                   "mixstyle": metrics["mixstyle"], "synthetic": metrics["synthetic"],
                   "mode": metrics["mode"], "seed": seed, "train_dataset": metrics["train_dataset"],
                   "test_dataset": target, "kind": "within" if target == metrics["train_dataset"] else "cross",
                   "best_epoch": metrics["best_epoch"], "best_val_macro_f1": round(metrics["best_val_macro_f1"], 4),
                   "train_minutes": metrics["train_minutes"],
                   "ms_per_image": metrics["ms_per_image_batch32"], "weights": metrics["weights_total"]}
            for column in SCORE_COLUMNS:
                value = s[column]
                row[column] = round(value, 4) if isinstance(value, float) else value
            rows.append(row)
    return rows


def summarize_seeds(rows):
    groups = {}
    for row in rows:
        key = (row["variant"], row["train_dataset"], row["test_dataset"], row["kind"])
        groups.setdefault(key, []).append(row)
    summary = []
    for (variant, train, test, kind), group in sorted(groups.items()):
        line = {"variant": variant, "train_dataset": train, "test_dataset": test, "kind": kind,
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


def main():
    options = read_options()
    for variant in options.variants:
        parse_variant(variant)
    Path(options.out).mkdir(parents=True, exist_ok=True)
    log_file = open(Path(options.out) / "run_log.txt", "a", encoding="utf-8")

    def log(text):
        line = f"{time.strftime('%Y-%m-%d %H:%M')}  {text}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    log(f"Piece 8: hybrid {', '.join(options.variants)} on {', '.join(options.datasets)}, "
        f"seeds {', '.join(map(str, options.seeds))}")
    run_all(options, log)
    rows = read_scores(options)
    if rows:
        write_csv(Path(options.out) / "hybrid_scores.csv", rows)
        write_csv(Path(options.out) / "hybrid_seed_summary.csv", summarize_seeds(rows))
        log(f"Wrote hybrid_scores.csv ({len(rows)} rows) and hybrid_seed_summary.csv")


if __name__ == "__main__":
    main()
