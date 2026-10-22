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
| 5 | `train_tf.py` | Keras/TensorFlow: trains one ImageNet-pretrained classifier (ResNet-50, MobileNetV2, EfficientNet-B3, DenseNet-121, ViT-B/16; frozen or fine-tuned) and tests it within-dataset and on every other dataset |
| 5b | `run_piece1.py` | Piece 1 of the GPU runs: ResNet-50 on R1, frozen and fine-tuned, and `piece1_scores.csv` |
| 5c | `run_piece2.py` | Piece 2: ResNet-50 on R2, R3, R4, frozen and fine-tuned, and `resnet50_scores.csv` |

All scripts are in `04_experiments/src/`. Steps 1-4 run on a CPU; step 5 needs a GPU
(`train_tf.py --quick` is a small CPU test).

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

Training (GPU, Keras + TensorFlow), on the unzipped upload folder made by step 4:

```
pip install -r 04_experiments/requirements_tf.txt
python 04_experiments/src/run_piece1.py --data UPLOAD_FOLDER --quick --out results_check   # quick check
python 04_experiments/src/run_piece1.py --data UPLOAD_FOLDER --out results_tf              # piece 1
python 04_experiments/src/run_piece2.py --data UPLOAD_FOLDER --out results_tf              # piece 2
```

The prepared data (`upload.zip`, 3.9 GB) is too large for GitHub and is shared separately.
