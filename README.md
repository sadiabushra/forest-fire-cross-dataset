# Cross-dataset generalization of fire and smoke image classification

Code for a master's thesis on how well deep-learning models for fire and smoke
classification generalize across independently collected image datasets.

This repository contains the **data audit and split pipeline** (steps 1-4, CPU)
and the **baseline training** scripts (step 5, GPU).

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1 | `inventory.py` | Opens every image, maps labels to shared L2/L4 labels, records hashes and statistics |
| 2 | `find_near_duplicates.py` | Finds near-duplicates within and across datasets and groups them into clusters |
| 2b | `check_matches.py` | Draws contact sheets to verify the matches by eye |
| 3a | `build_image_table.py` | Joins labels and clusters; drops unreadable files and exact duplicates |
| 3b | `make_splits_within.py` | Cluster-aware 70/15/15 train/val/test split per dataset |
| 3c | `make_cross_dataset_folds.py` | Reference-to-Shift and leave-one-dataset-out folds, leakage removed |
| 4 | `make_upload_copy.py` | Resized copy (short side 256 px) and split files for GPU training |
| 5 | `train.py` | Trains and tests one ImageNet-pretrained classifier (ResNet-50, MobileNetV2, EfficientNet-B3; frozen or fine-tuned) |
| 5b | `run_part1_within.py` | Part 1: within-dataset baselines (ResNet-50, frozen and fine-tuned, on R1-R4) and `summary.csv` |

All scripts are in `04_experiments/src/`. Steps 1-4 run on a CPU; step 5 needs a GPU
(`train.py --quick` is a small CPU test).

## Data

The datasets are **not included**; download them from their original sources.
The scripts expect the following layout next to `04_experiments/`:

```
03_datasets/
  raw/          the unzipped datasets (never modified)
  processed/    everything the scripts write
```

Labels used across datasets:

- **L2**: `fire_or_smoke` / `none` (all datasets)
- **L4**: `fire` / `smoke` / `fire_and_smoke` / `none` (datasets with box annotations)

## How to run

```
python -m venv .venv
.venv/Scripts/python -m pip install -r 04_experiments/requirements.txt

python 04_experiments/src/inventory.py
python 04_experiments/src/find_near_duplicates.py
python 04_experiments/src/check_matches.py
python 04_experiments/src/build_image_table.py
python 04_experiments/src/make_splits_within.py
python 04_experiments/src/make_cross_dataset_folds.py
python 04_experiments/src/make_upload_copy.py
```

Training (GPU), on the unzipped upload folder made by step 4:

```
python 04_experiments/src/train.py --data UPLOAD_FOLDER --dataset R1 --quick   # CPU smoke test
python 04_experiments/src/run_part1_within.py --data UPLOAD_FOLDER --out results   # part 1
```

The prepared data (`upload.zip`, 3.9 GB) is too large for GitHub and is shared separately.
