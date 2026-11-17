# Coffee Bean Classifier (`efficientnet_b0`)

Image-level coffee bean defect/quality classifier built with PyTorch and Torchvision (`efficientnet_b0`).  
Each input image is classified into **one** of 7 classes:

`black`, `broken`, `good`, `husk`, `immature`, `parchment`, `shell`

---

## Dependencies Required for `infer.py`

The following Python packages (listed in `requirements.txt`) are required to run `infer.py` (and `train.py`):

- **`torch`** (`>=2.0.0`) — PyTorch tensor & model inference/training runtime
- **`torchvision`** (`>=0.15.0`) — `efficientnet_b0` backbone and image transforms
- **`pillow`** (`>=10.0.0`) — image loading (`.jpg`, `.jpeg`, `.png`, case-insensitive) and EXIF orientation handling
- **`numpy`** (`>=1.24.0`) — array conversion for bean-region extraction
- **`opencv-python`** (`>=4.8.0`) — background estimation and bean-region segmentation

---

## Installation

1. Create and activate a Python virtual environment (recommended):

   ```bash
   python -m venv .venv
   # Windows (PowerShell)
   .venv\Scripts\Activate.ps1
   # Linux / macOS
   source .venv/bin/activate
   ```

2. Install all required dependencies from `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

   *(Optional)* If you have an NVIDIA GPU, install the CUDA-enabled PyTorch and Torchvision wheels from [pytorch.org](https://pytorch.org/get-started/locally/) before or instead of the default CPU wheels.

---

## Inference (`infer.py`)

Standard command:

```bash
python infer.py --input test/ --weights weights/model.pth --out outputs/
```

Arguments:
- `--input`: Input folder containing images or a single image file. Supports `.jpg`, `.jpeg`, and `.png` (case-insensitive).
- `--weights`: Path to a trained `.pth` checkpoint file or a folder containing weights (default: `weights`).
- `--out`: Output folder path (or a `.csv` file path). If omitted, defaults to `outputs`. Missing output and class folders are created automatically.
- `--device`: Optional device override (`cpu` or `cuda`). Auto-detected if omitted.
- `--batch-size`: Crop batch size during inference (default: `32`).

Output layout inside `--out` (e.g. `outputs/`):

```text
outputs/
├── black/
├── broken/
├── good/
├── husk/
├── immature/
├── parchment/
├── shell/
└── predictions.csv
```

- Each image is classified into **one** class (aggregating predictions across beans found in the image) and copied into the corresponding class subfolder.
- `predictions.csv` records `filename,class` where `class` matches one of the 7 class folder names.
- Exits with code `0` on success and a non-zero code on error (e.g., missing input/weights, invalid image extension, or no valid images found).

---

## Training (`train.py`)

Train an `efficientnet_b0` classifier using the class-organized images in `weights/`:

```bash
python train.py --data weights --out weights/model.pth
```

Useful options:
- `--epochs`: Number of training epochs (default: `25`).
- `--batch-size`: Mini-batch size (default: `16`).
- `--lr`: AdamW learning rate (default: `3e-4`).
- `--val-keyword`: Optional filename substring (e.g., `square`) to hold out for image-level validation.
- `--no-pretrained`: Train `efficientnet_b0` from scratch without downloading ImageNet pretrained weights.
