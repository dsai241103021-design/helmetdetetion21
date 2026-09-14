# Helmet Detection App

A computer vision project for detecting helmet usage in motorcycle images. The Streamlit app compares predictions from YOLO and SSD models side by side.

## Features

- Upload one image, multiple images, or use a webcam.
- YOLO object detection output with annotated bounding boxes.
- SSD object detection output with class-colored boxes:
  - Green: With Helmet
  - Yellow: Without Helmet
- English and Urdu interface options.
- XML annotation loader and standalone SSD training script.

## Project Structure

```text
app.py                 Streamlit application
ssd.py                 SSD training script
data.yaml              YOLO dataset configuration
prepare_yolo_dataset.py  Converts the source annotations for YOLO
cleanup_dataset.py     Dataset cleanup helper
requirements.txt       Python dependencies
Untitled.ipynb         Training and experimentation notebook
```

The dataset, generated runs, virtual environment, and model weights are intentionally excluded from GitHub by `.gitignore`.

## Setup

Use Python 3.10 or newer.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Required Local Files

Place the dataset in this structure:

```text
images/
  train/
  val/
  test/
annotations/
labels/
```

Place the trained YOLO model at:

```text
runs/detect/train-3/weights/best.pt
```

The Streamlit app also looks for the SSD checkpoint at:

```text
ssd_model.pth
```

These large files should be stored outside GitHub or in Git LFS / an external model registry.

## Run the App

```bash
streamlit run app.py
```

Open the local URL shown by Streamlit, normally `http://localhost:8501`.

## Train SSD

SSD training uses the Pascal VOC XML files in `annotations/`:

```bash
python ssd.py --epochs 20 --batch-size 8
```

The final checkpoint is saved as `ssd_model.pth`. Intermediate checkpoints are saved under `runs/ssd/ssd_latest.pth`.

Useful options:

```bash
python ssd.py --epochs 10 --batch-size 4 --learning-rate 0.001
```

## Train YOLO

The YOLO training workflow is also available in `Untitled.ipynb`. The dataset configuration is stored in `data.yaml`.

Example command from Python:

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.train(data="data.yaml", epochs=20, imgsz=640, batch=8)
```

## GitHub Upload

Before committing, confirm that generated and large files are ignored:

```bash
git init
git add .
git status
git commit -m "Add helmet detection app"
```

Do not commit the dataset, virtual environment, generated training runs, or model weights unless you intentionally use Git LFS.
