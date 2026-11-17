from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms

CLASSES: tuple[str, ...] = (
    "black",
    "broken",
    "good",
    "husk",
    "immature",
    "parchment",
    "shell",
)
VALID_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})
IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
WEIGHT_FILE_CANDIDATES = ("model.pth", "best_model.pth", "efficientnet_b0.pth")


def get_eval_transform(img_size: int = IMG_SIZE) -> transforms.Compose:
    """Return the deterministic preprocessing pipeline for inference."""
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_model(num_classes: int = len(CLASSES), pretrained: bool = False) -> nn.Module:
    """Construct an EfficientNet-B0 classifier with a custom output head."""
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def load_image_rgb(path: Path) -> Image.Image:
    """Open an image, apply EXIF orientation, and convert to RGB."""
    with Image.open(path) as img:
        return ImageOps.exif_transpose(img).convert("RGB")


def extract_bean_crops(img: Image.Image) -> list[Image.Image]:
    """Extract individual coffee bean crops from a sheet image, or return [img] for single-bean/fallback."""
    w, h = img.size
    if max(w, h) <= 600:
        return [img]

    rgb = np.asarray(img, dtype=np.uint8)
    scale = max(w, h) / 512.0
    sw = max(1, int(round(w / scale)))
    sh = max(1, int(round(h / scale)))

    # Estimate smooth paper illumination on a downscaled copy to separate beans from background.
    small = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_AREA)
    ksize = 35 if min(sw, sh) > 70 else (min(sw, sh) // 2) | 1
    bg_small = cv2.medianBlur(small, max(3, ksize))
    bg = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_LINEAR)

    diff = np.max(np.clip(bg.astype(np.int16) - rgb.astype(np.int16), 0, 255), axis=2).astype(
        np.uint8
    )
    mask = (diff > 15).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    min_area = (w * h) * 0.0004
    max_area = (w * h) * 0.05

    crops: list[Image.Image] = []
    for idx in range(1, num_labels):
        x, y, bw, bh, area = stats[idx]
        if x <= 5 or y <= 5 or x + bw >= w - 5 or y + bh >= h - 5:
            continue
        aspect = bw / max(int(bh), 1)
        if min_area <= area <= max_area and 0.3 <= aspect <= 3.0:
            side = int(max(bw, bh) * 1.35)
            cx = int(x + bw // 2)
            cy = int(y + bh // 2)
            x1 = max(0, cx - side // 2)
            y1 = max(0, cy - side // 2)
            x2 = min(w, x1 + side)
            y2 = min(h, y1 + side)
            crops.append(img.crop((x1, y1, x2, y2)))

    return crops if crops else [img]


def find_images(input_path: Path) -> list[Path]:
    """Collect valid image paths (.jpg, .jpeg, .png, case-insensitive) from a file or directory."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() not in VALID_EXTENSIONS:
            raise ValueError(
                f"Unsupported image extension '{input_path.suffix}'. Allowed: .jpg, .jpeg, .png"
            )
        return [input_path]

    if not input_path.is_dir():
        raise ValueError(f"Input path is neither a file nor a directory: {input_path}")

    images = sorted(
        p
        for p in input_path.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(
            f"No valid images (.jpg, .jpeg, .png) found in: {input_path}"
        )
    return images


def resolve_weights_path(weights_arg: Path) -> Path:
    """Resolve a checkpoint file from either a direct file path or a weights folder."""
    if not weights_arg.exists():
        raise FileNotFoundError(f"Weights path does not exist: {weights_arg}")

    if weights_arg.is_file():
        return weights_arg

    for name in WEIGHT_FILE_CANDIDATES:
        candidate = weights_arg / name
        if candidate.is_file():
            return candidate

    found = sorted(
        p
        for p in weights_arg.iterdir()
        if p.is_file() and p.suffix.lower() in {".pth", ".pt"}
    )
    if found:
        return found[0]

    raise FileNotFoundError(
        f"No model checkpoint (.pth/.pt) found inside weights directory: {weights_arg}"
    )


def resolve_output_paths(out_arg: Path) -> tuple[Path, Path]:
    """Resolve the output directory and predictions CSV path from a folder or .csv file argument."""
    if out_arg.suffix.lower() == ".csv":
        out_dir = out_arg.parent if str(out_arg.parent) else Path(".")
        csv_path = out_arg
    else:
        out_dir = out_arg
        csv_path = out_dir / "predictions.csv"
    return out_dir, csv_path


def ensure_output_structure(out_dir: Path, classes: Sequence[str]) -> None:
    """Create the output directory and all class subdirectories if they do not exist."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for cls_name in classes:
        (out_dir / cls_name).mkdir(parents=True, exist_ok=True)


def load_checkpoint(
    weights_path: Path, device: torch.device
) -> tuple[nn.Module, tuple[str, ...], int]:
    """Load an EfficientNet-B0 checkpoint and return (model, classes, img_size)."""
    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        classes = tuple(checkpoint.get("classes", CLASSES))
        img_size = int(checkpoint.get("img_size", IMG_SIZE))
        state_dict = checkpoint["model_state_dict"]
    else:
        classes = CLASSES
        img_size = IMG_SIZE
        state_dict = checkpoint

    model = build_model(num_classes=len(classes), pretrained=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, classes, img_size


@torch.inference_mode()
def predict_image(
    img_path: Path,
    model: nn.Module,
    transform: transforms.Compose,
    classes: Sequence[str],
    device: torch.device,
    batch_size: int = 32,
) -> str:
    """Predict a single class label for an entire picture by aggregating crop probabilities."""
    img = load_image_rgb(img_path)
    crops = extract_bean_crops(img)
    tensors = torch.stack([transform(crop) for crop in crops])

    prob_sum = torch.zeros(len(classes), dtype=torch.float32, device=device)
    for start in range(0, len(tensors), batch_size):
        batch = tensors[start : start + batch_size].to(device)
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
        prob_sum += probs.sum(dim=0)

    pred_idx = int(torch.argmax(prob_sum).item())
    return classes[pred_idx]


def export_image(src: Path, dst_dir: Path, used_paths: set[Path]) -> Path:
    """Copy a classified image into its predicted class folder, avoiding intra-run collisions."""
    dst = dst_dir / src.name
    if dst in used_paths:
        stem, suffix = src.stem, src.suffix
        counter = 1
        while (dst_dir / f"{stem}_{counter}{suffix}") in used_paths:
            counter += 1
        dst = dst_dir / f"{stem}_{counter}{suffix}"
    used_paths.add(dst)
    if not dst.exists() or dst.resolve() != src.resolve():
        shutil.copy2(src, dst)
    return dst


def write_predictions_csv(csv_path: Path, rows: Iterable[tuple[str, str]]) -> None:
    """Write image filename and predicted class enum rows to predictions.csv."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filename", "class"])
        writer.writerows(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify coffee bean images into 7 defect/quality classes (one class per picture)."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input image file or directory containing .jpg/.jpeg/.png images.",
    )
    parser.add_argument(
        "--weights",
        default="weights",
        help="Path to trained weights file (.pth) or directory containing weights (default: weights).",
    )
    parser.add_argument(
        "--out",
        default="outputs",
        help="Output folder or predictions .csv path (default: outputs).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Compute device (e.g. 'cpu' or 'cuda'). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for bean crop evaluation per image (default: 32).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        device = torch.device(
            args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        image_paths = find_images(Path(args.input))
        weights_path = resolve_weights_path(Path(args.weights))
        out_dir, csv_path = resolve_output_paths(Path(args.out))

        model, classes, img_size = load_checkpoint(weights_path, device)
        ensure_output_structure(out_dir, classes)
        transform = get_eval_transform(img_size)

        predictions: list[tuple[str, str]] = []
        used_paths: set[Path] = set()
        for img_path in image_paths:
            pred_class = predict_image(
                img_path=img_path,
                model=model,
                transform=transform,
                classes=classes,
                device=device,
                batch_size=args.batch_size,
            )
            exported_path = export_image(img_path, out_dir / pred_class, used_paths)
            predictions.append((exported_path.name, pred_class))
            print(f"{exported_path.name} -> {pred_class}")

        write_predictions_csv(csv_path, predictions)
        print(f"Saved {len(predictions)} predictions to {csv_path}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
