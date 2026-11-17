from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from infer import (
    CLASSES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMG_SIZE,
    VALID_EXTENSIONS,
    build_model,
    extract_bean_crops,
    get_eval_transform,
    load_image_rgb,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_train_transform(img_size: int = IMG_SIZE) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomChoice(
                [
                    transforms.RandomRotation((0, 0)),
                    transforms.RandomRotation((90, 90)),
                    transforms.RandomRotation((180, 180)),
                    transforms.RandomRotation((270, 270)),
                ]
            ),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.08, hue=0.015),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class BeanCropDataset(Dataset):
    """Dataset of extracted coffee bean PIL crops paired with class indices."""

    def __init__(
        self,
        samples: Sequence[tuple[Image.Image, int]],
        transform: transforms.Compose,
    ) -> None:
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        img, label = self.samples[index]
        return self.transform(img), label


def collect_dataset_images(
    data_dir: Path, classes: Sequence[str] = CLASSES
) -> list[tuple[Path, int]]:
    """Discover training images recursively under data_dir/<class_name>/."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Training data directory does not exist: {data_dir}")

    items: list[tuple[Path, int]] = []
    for class_idx, class_name in enumerate(classes):
        class_dir = data_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Expected class folder '{class_name}' in {data_dir}"
            )
        class_images = sorted(
            p
            for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
        )
        if not class_images:
            raise FileNotFoundError(f"No valid images found in class folder: {class_dir}")
        for img_path in class_images:
            items.append((img_path, class_idx))
    return items


def build_crop_samples(
    image_items: Sequence[tuple[Path, int]],
) -> tuple[list[tuple[Image.Image, int]], list[int]]:
    """Extract bean crops from each source image and track per-class crop counts."""
    samples: list[tuple[Image.Image, int]] = []
    class_counts = [0] * len(CLASSES)

    for img_path, class_idx in image_items:
        img = load_image_rgb(img_path)
        crops = extract_bean_crops(img)
        for crop in crops:
            samples.append((crop, class_idx))
            class_counts[class_idx] += 1

    return samples, class_counts


@torch.inference_mode()
def evaluate_images(
    model: nn.Module,
    image_items: Sequence[tuple[Path, int]],
    transform: transforms.Compose,
    device: torch.device,
) -> tuple[float, int, int]:
    """Evaluate image-level classification accuracy (one class predicted per image)."""
    model.eval()
    correct = 0
    total = len(image_items)

    for img_path, true_idx in image_items:
        img = load_image_rgb(img_path)
        crops = extract_bean_crops(img)
        batch = torch.stack([transform(c) for c in crops]).to(device)
        probs = torch.softmax(model(batch), dim=1).mean(dim=0)
        pred_idx = int(torch.argmax(probs).item())
        if pred_idx == true_idx:
            correct += 1

    acc = correct / max(total, 1)
    return acc, correct, total


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze or unfreeze the EfficientNet-B0 feature extractor backbone."""
    for param in model.features.parameters():
        param.requires_grad = trainable


def train_model(
    data_dir: Path,
    out_path: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    freeze_epochs: int,
    finetune_lr: float,
    label_smoothing: float,
    pretrained: bool,
    val_keyword: str | None,
    seed: int,
    device: torch.device,
) -> None:
    set_seed(seed)
    all_images = collect_dataset_images(data_dir, CLASSES)

    if val_keyword:
        kw = val_keyword.lower()
        train_images = [item for item in all_images if kw not in item[0].name.lower()]
        val_images = [item for item in all_images if kw in item[0].name.lower()]
        if not train_images or not val_images:
            raise ValueError(
                f"Validation keyword '{val_keyword}' produced empty split "
                f"(train={len(train_images)}, val={len(val_images)})."
            )
    else:
        train_images = all_images
        val_images = []

    print(
        f"Discovered {len(all_images)} images across {len(CLASSES)} classes "
        f"(train={len(train_images)}, val={len(val_images)})."
    )

    train_samples, class_counts = build_crop_samples(train_images)
    print(f"Extracted {len(train_samples)} training bean crops: {dict(zip(CLASSES, class_counts))}")

    train_ds = BeanCropDataset(train_samples, transform=get_train_transform(IMG_SIZE))
    train_loader = DataLoader(
        train_ds,
        batch_size=min(batch_size, len(train_ds)),
        shuffle=True,
        num_workers=0,
    )

    model = build_model(num_classes=len(CLASSES), pretrained=pretrained).to(device)

    counts_tensor = torch.tensor(class_counts, dtype=torch.float32, device=device)
    class_weights = counts_tensor.sum() / (len(CLASSES) * counts_tensor.clamp(min=1.0))
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    backbone_frozen = freeze_epochs > 0
    if backbone_frozen:
        set_backbone_trainable(model, trainable=False)
        optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(min(freeze_epochs, epochs), 1), eta_min=lr * 0.2
        )
        print(f"Freezing backbone for first {min(freeze_epochs, epochs)} epoch(s) (head lr={lr:.1e}).")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs, 1), eta_min=lr * 0.1
        )

    eval_transform = get_eval_transform(IMG_SIZE)

    best_score = -1.0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, epochs + 1):
        if backbone_frozen and epoch > freeze_epochs:
            backbone_frozen = False
            set_backbone_trainable(model, trainable=True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=finetune_lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(epochs - freeze_epochs, 1), eta_min=finetune_lr * 0.1
            )
            print(f"Unfreezing backbone at epoch {epoch} (lr={finetune_lr:.1e}).")

        model.train()
        running_loss = 0.0
        correct_crops = 0
        total_crops = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * inputs.size(0)
            preds = torch.argmax(logits, dim=1)
            correct_crops += int((preds == targets).sum().item())
            total_crops += inputs.size(0)

        scheduler.step()
        avg_loss = running_loss / max(total_crops, 1)
        crop_acc = correct_crops / max(total_crops, 1)

        if val_images:
            val_acc, val_ok, val_n = evaluate_images(model, val_images, eval_transform, device)
            score = val_acc
            status = f"val_img_acc={val_acc:.3f} ({val_ok}/{val_n})"
        else:
            score = crop_acc - 0.01 * avg_loss
            status = f"train_crop_acc={crop_acc:.3f}"

        if score >= best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"Epoch {epoch:02d}/{epochs:02d} | loss={avg_loss:.4f} | crop_acc={crop_acc:.3f} | {status}")

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    train_img_acc, ok, tot = evaluate_images(model, train_images, eval_transform, device)
    print(f"Final train image-level accuracy: {train_img_acc:.3f} ({ok}/{tot})")

    if val_images:
        val_acc, val_ok, val_n = evaluate_images(model, val_images, eval_transform, device)
        print(f"Final val image-level accuracy: {val_acc:.3f} ({val_ok}/{val_n})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "arch": "efficientnet_b0",
            "classes": list(CLASSES),
            "img_size": IMG_SIZE,
            "model_state_dict": best_state,
        },
        out_path,
    )
    print(f"Saved trained weights to {out_path}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train EfficientNet-B0 coffee bean classifier on existing class folders."
    )
    parser.add_argument(
        "--data",
        default="weights",
        help="Root directory containing the 7 class folders (default: weights).",
    )
    parser.add_argument(
        "--out",
        default="weights/model.pth",
        help="Output checkpoint file path (default: weights/model.pth).",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs (default: 30).")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16).")
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-3,
        help="Initial learning rate for the classifier head / unfrozen training (default: 2e-3).",
    )
    parser.add_argument(
        "--freeze-epochs",
        type=int,
        default=5,
        help="Number of initial epochs to freeze the backbone (default: 5).",
    )
    parser.add_argument(
        "--finetune-lr",
        type=float,
        default=1e-4,
        help="Learning rate after unfreezing the backbone (default: 1e-4).",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Cross-entropy label smoothing factor (default: 0.0).",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Train EfficientNet-B0 from scratch without ImageNet pretrained weights.",
    )
    parser.add_argument(
        "--val-keyword",
        default=None,
        help="Optional filename substring (e.g. 'square') to hold out for image-level validation.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument(
        "--device",
        default=None,
        help="Compute device ('cpu' or 'cuda'). Auto-detected if omitted.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        device = torch.device(
            args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        train_model(
            data_dir=Path(args.data),
            out_path=Path(args.out),
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            freeze_epochs=args.freeze_epochs,
            finetune_lr=args.finetune_lr,
            label_smoothing=args.label_smoothing,
            pretrained=not args.no_pretrained,
            val_keyword=args.val_keyword,
            seed=args.seed,
            device=device,
        )
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
