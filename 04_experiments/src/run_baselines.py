"""Step 5b: run all baseline trainings and summarise them.

Runs train.py for every model, frozen and fine-tuned, on the within-dataset
splits (R1-R4) and the cross-dataset folds, skipping finished runs, then
collects all test scores into summary.csv.

Usage:
    python 04_experiments/src/run_baselines.py --data UPLOAD_FOLDER [--group 1 2]
        [--setups within ref_to_shift lodo]
        [--dry-run] [--summary-only]
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train


WITHIN_DATASETS = ["R1", "R2", "R3", "R4"]

FOLD_FILES = ["ref_to_shift.csv", "lodo_R1.csv", "lodo_R2.csv", "lodo_R3.csv", "lodo_R4.csv"]

GROUP_MODELS = {
    1: ["resnet50"],
    2: ["mobilenet_v2", "efficientnet_b3"],
}

TRAIN_SCRIPT = Path(__file__).resolve().parent / "train.py"


def read_options():
    parser = argparse.ArgumentParser(description="Run all baseline trainings and summarise them.")
    parser.add_argument("--data", required=True, help="the unzipped upload folder")
    parser.add_argument("--out", default="results", help="folder for the run folders and summary.csv")
    parser.add_argument("--group", type=int, nargs="+", default=[1], choices=sorted(GROUP_MODELS))
    parser.add_argument("--setups", nargs="+", default=["within", "ref_to_shift", "lodo"],
                        choices=["within", "ref_to_shift", "lodo"],
                        help="which kinds of setup to run (to split the work into parts)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="only print the list of runs")
    parser.add_argument("--summary-only", action="store_true", help="only remake summary.csv")
    return parser.parse_args()


def make_setups(kinds):
    """The train.py options for each setup of the chosen kinds (within / ref_to_shift / lodo)."""
    setups = []
    if "within" in kinds:
        setups += [["--dataset", d] for d in WITHIN_DATASETS]
    for fold_file in FOLD_FILES:
        kind = "ref_to_shift" if fold_file.startswith("ref_to_shift") else "lodo"
        if kind in kinds:
            setups.append(["--split-file", fold_file])
    return setups


def make_run_list(options):
    """Each run is a list of command-line options for train.py."""
    runs = []
    for group in options.group:
        for model in GROUP_MODELS[group]:
            for freeze in (True, False):
                setups = make_setups(options.setups)
                for setup in setups:
                    args = setup + ["--model", model]
                    if freeze:
                        args.append("--freeze")
                    runs.append(args)
    return runs


def folder_of(run_args, options):
    """The folder train.py will write this run to (same name rule as train.py)."""
    settings = argparse.Namespace(split_file="within_dataset.csv", dataset=None, freeze=False,
                                quick=False, seed=options.seed)
    for i, word in enumerate(run_args):
        if word == "--dataset":
            settings.dataset = run_args[i + 1]
        elif word == "--split-file":
            settings.split_file = run_args[i + 1]
        elif word == "--model":
            settings.model = run_args[i + 1]
        elif word == "--freeze":
            settings.freeze = True
    return Path(options.out) / train.run_name(settings)


def run_all(runs, options, log):
    for number, run_args in enumerate(runs, 1):
        folder = folder_of(run_args, options)
        if (folder / "test_metrics.json").exists():
            log(f"[{number}/{len(runs)}] {folder.name}: already done, skipped")
            continue
        command = [sys.executable, str(TRAIN_SCRIPT), "--data", options.data, "--out", options.out,
                   "--epochs", str(options.epochs), "--batch-size", str(options.batch_size),
                   "--workers", str(options.workers), "--seed", str(options.seed)] + run_args
        log(f"[{number}/{len(runs)}] {folder.name}: started")
        start = time.time()
        result = subprocess.run(command)
        minutes = (time.time() - start) / 60
        status = "finished" if result.returncode == 0 else f"FAILED (exit code {result.returncode})"
        log(f"[{number}/{len(runs)}] {folder.name}: {status} after {minutes:.0f} min")


def write_summary(out_folder):
    """One row per (run, test dataset) from every finished run folder."""
    rows = []
    for metrics_path in sorted(Path(out_folder).glob("*/test_metrics.json")):
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)
        with open(metrics_path.parent / "settings.json", encoding="utf-8") as f:
            settings = json.load(f)
        parts = [("overall", metrics["overall"])] + list(metrics["per_dataset"].items())
        for test_dataset, s in parts:
            rows.append({
                "run": metrics_path.parent.name,
                "split_file": settings["split_file"],
                "train_dataset": settings["dataset"] or "(fold)",
                "model": settings["model"],
                "mode": "frozen" if settings["freeze"] else "finetuned",
                "best_epoch": metrics["best_epoch"],
                "test_dataset": test_dataset,
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
    Path(options.out).mkdir(parents=True, exist_ok=True)
    log_file = open(Path(options.out) / "run_baselines_log.txt", "a", encoding="utf-8")

    def log(text):
        line = f"{time.strftime('%Y-%m-%d %H:%M')}  {text}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    if not options.summary_only:
        runs = make_run_list(options)
        if options.dry_run:
            for number, run_args in enumerate(runs, 1):
                print(f"{number:2d}. {folder_of(run_args, options).name}")
            return
        log(f"Starting {len(runs)} runs (groups {options.group}, setups {options.setups})")
        run_all(runs, options, log)

    summary_path, n = write_summary(options.out)
    log(f"Wrote {summary_path} ({n} rows)")


if __name__ == "__main__":
    main()
