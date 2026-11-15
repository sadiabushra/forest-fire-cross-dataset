"""Hyperparameter search: grid search, random search and augmentation comparison.

Runs train_tf.py for one model on one or more training datasets:
  grid     learning rate {1e-3, 3e-4, 1e-4, 3e-5} x batch size {32, 64}
  random   N trials, learning rate log-uniform in [1e-5, 1e-3], batch size {32, 64}
  augment  training augmentation none / standard / strong
The best setting is chosen on the validation macro-F1 only. Writes search_scores.csv
(one row per run) and search_best.csv (best run per search, model, dataset and mode).

Usage:
    python 04_experiments/src/run_search.py --data UPLOAD_FOLDER --model NAME
        --search grid|random|augment [...] [--datasets R4 ...] [--trials N]
        [--freeze] [--epochs N] [--out results_search] [--quick]
"""

import argparse
import csv
import json
import math
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------

MODEL_NAMES = ["resnet50", "mobilenet_v2", "efficientnet_b3", "densenet121", "vit_b16"]
DATASETS = ["R1", "R2", "R3", "R4"]

# Grid search values.
LR_GRID = [1e-3, 3e-4, 1e-4, 3e-5]
BATCH_GRID = [32, 64]

# Random search: learning rate range (log-uniform) and the seed for drawing the trials.
RANDOM_LR = (1e-5, 1e-3)
RANDOM_SEARCH_SEED = 0

AUGMENT_LEVELS = ["none", "standard", "strong"]
DEFAULT_BATCH = 64

SEED = 42
TRAIN_SCRIPT = Path(__file__).resolve().parent / "train_tf.py"


def read_options():
    parser = argparse.ArgumentParser(description="Grid search, random search and augmentation comparison.")
    parser.add_argument("--data", required=True, help="the unzipped upload folder")
    parser.add_argument("--out", default="results_search", help="folder for the run folders and tables")
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=["R4"])
    parser.add_argument("--search", nargs="+", choices=["grid", "random", "augment"], required=True)
    parser.add_argument("--trials", type=int, default=8, help="number of random search trials")
    parser.add_argument("--freeze", action="store_true", help="frozen backbone (default: fine-tuned)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--quick", action="store_true", help="check only: tiny runs (train_tf.py --quick)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 1. The list of runs
# ---------------------------------------------------------------------------

def make_trials(options):
    """Each trial is a dict: search, tag, lr (None = train_tf.py default), batch, augment."""
    trials = []
    if "grid" in options.search:
        for lr in LR_GRID:
            for batch in BATCH_GRID:
                trials.append({"search": "grid", "tag": f"_grid_lr{lr:.0e}_bs{batch}",
                               "lr": lr, "batch": batch, "augment": "standard"})
    if "random" in options.search:
        rng = random.Random(RANDOM_SEARCH_SEED)
        low, high = RANDOM_LR
        for number in range(1, options.trials + 1):
            # log-uniform: every power of ten is equally likely
            lr = 10 ** rng.uniform(math.log10(low), math.log10(high))
            batch = rng.choice(BATCH_GRID)
            trials.append({"search": "random", "tag": f"_rand{number:02d}_lr{lr:.1e}_bs{batch}",
                           "lr": lr, "batch": batch, "augment": "standard"})
    if "augment" in options.search:
        for level in AUGMENT_LEVELS:
            trials.append({"search": "augment", "tag": f"_aug-{level}",
                           "lr": None, "batch": DEFAULT_BATCH, "augment": level})
    return trials


def folder_name(model, dataset, freeze, tag, quick):
    """The folder train_tf.py writes the run to."""
    mode = "frozen" if freeze else "finetuned"
    name = f"within_{dataset}_{model}_{mode}_seed{SEED}{tag}"
    if quick:
        name += "_QUICK"
    return name


# ---------------------------------------------------------------------------
# 2. Run them
# ---------------------------------------------------------------------------

def run_all(options, trials, log):
    runs = [(dataset, trial) for dataset in options.datasets for trial in trials]
    for number, (dataset, trial) in enumerate(runs, 1):
        name = folder_name(options.model, dataset, options.freeze, trial["tag"], options.quick)
        if (Path(options.out) / name / "test_metrics.json").exists():
            log(f"[{number}/{len(runs)}] {name}: already done, skipped")
            continue
        command = [sys.executable, str(TRAIN_SCRIPT), "--data", options.data, "--out", options.out,
                   "--dataset", dataset, "--model", options.model, "--seed", str(SEED),
                   "--batch-size", str(trial["batch"]), "--augment", trial["augment"],
                   "--epochs", str(options.epochs), "--tag", trial["tag"], "--no-save"]
        if trial["lr"] is not None:
            command += ["--lr", str(trial["lr"])]
        if options.freeze:
            command.append("--freeze")
        if options.quick:
            command.append("--quick")
        log(f"[{number}/{len(runs)}] {name}: started")
        start = time.time()
        # Each run is its own process, so the GPU memory is freed after it.
        result = subprocess.run(command)
        minutes = (time.time() - start) / 60
        status = "finished" if result.returncode == 0 else f"FAILED (exit code {result.returncode})"
        log(f"[{number}/{len(runs)}] {name}: {status} after {minutes:.0f} min")


# ---------------------------------------------------------------------------
# 3. Collect the results
# ---------------------------------------------------------------------------

def search_type(tag):
    for prefix, kind in (("_grid", "grid"), ("_rand", "random"), ("_aug-", "augment")):
        if tag.startswith(prefix):
            return kind
    return ""


def mean(values):
    values = [v for v in values if v is not None]
    return round(statistics.mean(values), 4) if values else None


def read_runs(options):
    """One row per finished run in the --out folder."""
    rows = []
    for metrics_path in sorted(Path(options.out).glob("within_*/test_metrics.json")):
        run = metrics_path.parent.name
        if run.endswith("_QUICK") != options.quick:
            continue
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)
        with open(metrics_path.parent / "settings.json", encoding="utf-8") as f:
            settings = json.load(f)
        own = metrics["train_dataset"]
        cross = [s for target, s in metrics["test"].items() if target != own]
        rows.append({
            "run": run, "search": search_type(settings.get("tag", "")), "model": metrics["model"],
            "mode": metrics["mode"], "train_dataset": own, "lr": settings["lr"],
            "batch_size": settings["batch_size"], "augment": settings.get("augment", "standard"),
            "best_epoch": metrics["best_epoch"], "val_macro_f1": round(metrics["best_val_macro_f1"], 4),
            "test_macro_f1": round(metrics["test"][own]["macro_f1"], 4),
            "cross_macro_f1": mean(s["macro_f1"] for s in cross),
            "cross_auc": mean(s["auc"] for s in cross),
            "train_minutes": metrics["train_minutes"],
        })
    return rows


def best_per_group(rows):
    """Per (search, model, dataset, mode): the run with the highest VALIDATION macro-F1."""
    groups = {}
    for row in rows:
        key = (row["search"], row["model"], row["train_dataset"], row["mode"])
        if key not in groups or row["val_macro_f1"] > groups[key]["val_macro_f1"]:
            groups[key] = row
    return [groups[key] for key in sorted(groups)]


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    options = read_options()
    trials = make_trials(options)
    Path(options.out).mkdir(parents=True, exist_ok=True)
    log_file = open(Path(options.out) / "run_log.txt", "a", encoding="utf-8")

    def log(text):
        line = f"{time.strftime('%Y-%m-%d %H:%M')}  {text}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    mode = "frozen" if options.freeze else "fine-tuned"
    log(f"Search {' + '.join(options.search)}: {options.model} {mode} on {', '.join(options.datasets)} "
        f"({len(trials) * len(options.datasets)} runs)")
    run_all(options, trials, log)

    rows = read_runs(options)
    if not rows:
        log("No finished runs to collect.")
        return
    best = best_per_group(rows)
    write_csv(Path(options.out) / "search_scores.csv", rows)
    write_csv(Path(options.out) / "search_best.csv", best)
    log(f"Wrote search_scores.csv ({len(rows)} runs) and search_best.csv")
    for row in best:
        log(f"  best {row['search']:8s} {row['model']} {row['mode']} {row['train_dataset']}: "
            f"lr {row['lr']:.1e}, batch {row['batch_size']}, augment {row['augment']} -> "
            f"val {row['val_macro_f1']:.4f}, test {row['test_macro_f1']:.4f}, cross {row['cross_macro_f1']:.4f}")


if __name__ == "__main__":
    main()
