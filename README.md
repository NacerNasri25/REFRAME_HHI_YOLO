# REFRAME – YOLO11 Inference (Fraunhofer HHI dataset)

This repository contains scripts and documentation to run a provided YOLO11 detection model
on the Fraunhofer HHI Surgical Instrument dataset and export predictions in a reproducible way.

⚠️ Dataset files, model weights, and outputs are NOT included due to confidentiality.

---

## Structure

- `scripts/test_yolo_inference.py`  
  Sanity check: loads the model and runs inference on one image.

- `scripts/run_yolo_on_dataset.py`  
  Runs inference on a dataset split and exports results into:
  `outputs/yolo_<split>/<image_name>/info.json`

- `docs/ENVIRONMENT.md`  
  Setup instructions and environment notes.

- `docs/DATASET_SUMMARY.md`  
  Dataset overview: structure, splits, GT availability, and classes.

- `requirements.txt`  
  Exact python dependencies (generated via `pip freeze`).

---

## Setup (example)

```bash
conda create -n reframe_yolo python=3.10 -y
conda activate reframe_yolo
pip install -r requirements.txt
