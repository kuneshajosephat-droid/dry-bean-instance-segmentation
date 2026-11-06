## Imports

import argparse
import copy
import csv
import hashlib
import io
import json
import random
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from shapely.geometry import Polygon
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_Weights,
    maskrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import v2
from torchvision.transforms.functional import pil_to_tensor
from torchvision.utils import draw_segmentation_masks


## Configuration

SEED = 42
BATCH_SIZE = 8
DISPLAY_SCORE = 0.5
MASK_THRESHOLD = 0.5

CLASS_NAMES = {
    0: "background",
    1: "intact",
    2: "broken",
}

CLASS_COLORS = {
    1: "lime",
    2: "orange",
}


## File utilities

def read_json(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def save_csv(path, rows):
    if not rows:
        return

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plain(tensor):
    return tensor.as_subclass(torch.Tensor)


## Dataset validation

def validate_dataset(root, output):
    combined = read_json(
        root / "annotations" / "instances_all.json"
    )

    split_data = {
        split: read_json(
            root / "annotations" / f"instances_{split}.json"
        )
        for split in ("train", "val", "test")
    }

    categories = {
        category["id"]: category["name"]
        for category in combined["categories"]
    }

    assert categories == {1: "intact", 2: "broken"}

    images = {
        image["id"]: image
        for image in combined["images"]
    }

    annotations = {
        annotation["id"]: annotation
        for annotation in combined["annotations"]
    }

    names = {
        image["file_name"]
        for image in combined["images"]
    }

    assert len(images) == len(combined["images"])
    assert len(names) == len(images)
    assert len(annotations) == len(combined["annotations"])

    split_names = {}
    assigned_annotations = []
    summary = []

    for split, data in split_data.items():
        image_ids = {
            image["id"]
            for image in data["images"]
        }

        filenames = [
            image["file_name"]
            for image in data["images"]
        ]

        assert len(image_ids) == len(data["images"])
        assert len(filenames) == len(set(filenames))
        assert data["categories"] == combined["categories"]

        split_names[split] = set(filenames)

        for image in data["images"]:
            assert image == images[image["id"]]

        for annotation in data["annotations"]:
            assert annotation["image_id"] in image_ids
            assert annotation == annotations[annotation["id"]]
            assigned_annotations.append(annotation["id"])

        counts = Counter(
            annotation["category_id"]
            for annotation in data["annotations"]
        )

        assert counts[1] > 0 and counts[2] > 0

        summary.append({
            "split": split,
            "photographs": len(data["images"]),
            "percentage": round(
                100 * len(data["images"]) / len(images), 2
            ),
            "bean_annotations": len(data["annotations"]),
            "intact": counts[1],
            "broken": counts[2],
        })

    for first, second in combinations(split_names, 2):
        assert split_names[first].isdisjoint(split_names[second])

    assert set.union(*split_names.values()) == names
    assert len(assigned_annotations) == len(set(assigned_annotations))
    assert set(assigned_annotations) == set(annotations)

    hashes = defaultdict(list)

    for image in combined["images"]:
        path = root / "images" / image["file_name"]
        assert path.is_file(), path

        with Image.open(path) as photo:
            photo = ImageOps.exif_transpose(photo).convert("RGB")

            assert photo.size == (
                image["width"],
                image["height"],
            )

            digest = hashlib.sha256(
                str(photo.size).encode() + photo.tobytes()
            ).hexdigest()

            hashes[digest].append(image["file_name"])

    assert all(len(group) == 1 for group in hashes.values())

    for annotation in combined["annotations"]:
        image = images[annotation["image_id"]]

        assert annotation["category_id"] in (1, 2)
        assert annotation.get("iscrowd", 0) == 0
        assert annotation["segmentation"]

        for segment in annotation["segmentation"]:
            assert len(segment) >= 6
            assert len(segment) % 2 == 0
            assert np.isfinite(segment).all()

            points = np.asarray(segment).reshape(-1, 2)
            polygon = Polygon(points)

            assert polygon.is_valid
            assert polygon.area > 0
            assert (points >= 0).all()
            assert (points[:, 0] <= image["width"]).all()
            assert (points[:, 1] <= image["height"]).all()

        rle = mask_utils.merge(
            mask_utils.frPyObjects(
                annotation["segmentation"],
                image["height"],
                image["width"],
            )
        )

        assert float(mask_utils.area(rle)) > 0

        assert np.isclose(
            annotation["area"],
            float(mask_utils.area(rle)),
        )

        assert np.allclose(
            annotation["bbox"],
            mask_utils.toBbox(rle),
        )

    save_csv(output / "split_summary.csv", summary)

    return split_data


## Custom dataset

class BeanDataset(Dataset):
    def __init__(self, root, split, transforms=None):
        self.root = Path(root)
        self.transforms = transforms

        annotation_file = (
            self.root / "annotations" / f"instances_{split}.json"
        )

        with redirect_stdout(io.StringIO()):
            self.coco = COCO(str(annotation_file))

        self.image_ids = sorted(self.coco.getImgIds())

        categories = {
            category["id"]: category["name"]
            for category in self.coco.loadCats(
                self.coco.getCatIds()
            )
        }

        assert categories == {1: "intact", 2: "broken"}

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        info = self.coco.loadImgs(image_id)[0]

        with Image.open(
            self.root / "images" / info["file_name"]
        ) as photo:
            photo = ImageOps.exif_transpose(photo).convert("RGB")
            image = pil_to_tensor(photo).float() / 255.0

        height, width = image.shape[-2:]

        assert (width, height) == (
            info["width"],
            info["height"],
        )

        annotations = self.coco.loadAnns(
            self.coco.getAnnIds(imgIds=[image_id])
        )

        masks = []
        boxes = []
        labels = []

        for annotation in annotations:
            mask = self.coco.annToMask(annotation).astype(np.uint8)
            ys, xs = np.where(mask > 0)

            if len(xs) == 0:
                raise ValueError(
                    f"Empty mask in {info['file_name']}"
                )

            masks.append(mask)

            boxes.append([
                xs.min(),
                ys.min(),
                xs.max() + 1,
                ys.max() + 1,
            ])

            labels.append(annotation["category_id"])

        masks = torch.from_numpy(
            np.stack(masks)
            if masks
            else np.zeros((0, height, width), dtype=np.uint8)
        )

        target = {
            "boxes": tv_tensors.BoundingBoxes(
                torch.tensor(
                    boxes, dtype=torch.float32
                ).reshape(-1, 4),
                format="XYXY",
                canvas_size=(height, width),
            ),
            "labels": torch.tensor(
                labels, dtype=torch.int64
            ),
            "masks": tv_tensors.Mask(masks),
            "image_id": torch.tensor(
                image_id, dtype=torch.int64
            ),
            "area": masks.flatten(1).sum(1).to(torch.float32),
            "iscrowd": torch.zeros(
                len(labels), dtype=torch.int64
            ),
        }

        image = tv_tensors.Image(image)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


## Data loaders

def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def create_loader(dataset):
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=False,
    )


## Ground-truth visualization

def target_overlay(image, target):
    rgb = (
        plain(image).detach().cpu()
        .mul(255).round().clamp(0, 255)
        .to(torch.uint8)
    )

    masks = plain(target["masks"]).detach().cpu().bool()

    if len(masks):
        colors = [
            CLASS_COLORS[label]
            for label in target["labels"].tolist()
        ]

        rgb = draw_segmentation_masks(
            rgb,
            masks,
            alpha=0.4,
            colors=colors,
        )

    return rgb.permute(1, 2, 0).numpy()


def visualize_annotations(dataset, output):
    prefixes = (
        "intact_",
        "broken_",
        "mixed_",
        "touching_",
    )

    fig, axes = plt.subplots(4, 2, figsize=(12, 20))

    for row, prefix in enumerate(prefixes):
        index = next(
            index
            for index, image_id in enumerate(dataset.image_ids)
            if dataset.coco.imgs[image_id]["file_name"]
            .startswith(prefix)
        )

        image, target = dataset[index]
        image_id = int(target["image_id"])
        filename = dataset.coco.imgs[image_id]["file_name"]

        axes[row, 0].imshow(
            plain(image).permute(1, 2, 0)
        )
        axes[row, 0].set_title(filename)

        axes[row, 1].imshow(
            target_overlay(image, target)
        )
        axes[row, 1].set_title("Ground truth")

        for axis in axes[row]:
            axis.axis("off")

    fig.tight_layout()

    fig.savefig(
        output / "ground_truth_examples.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


## Joint augmentation

def create_train_transforms():
    return v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
    ])


def visualize_augmentation(dataset, output):
    image, target = dataset[0]

    flipped_image, flipped_target = v2.RandomHorizontalFlip(
        p=1.0
    )(image, target)

    assert torch.equal(
        flipped_image,
        image.flip(-1),
    )

    assert torch.equal(
        flipped_target["masks"],
        target["masks"].flip(-1),
    )

    width = image.shape[-1]
    expected_boxes = target["boxes"].clone()

    expected_boxes[:, 0] = width - target["boxes"][:, 2]
    expected_boxes[:, 2] = width - target["boxes"][:, 0]

    assert torch.allclose(
        flipped_target["boxes"],
        expected_boxes,
    )

    assert torch.equal(
        flipped_target["labels"],
        target["labels"],
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 7))

    axes[0].imshow(target_overlay(image, target))
    axes[0].set_title("Original")

    axes[1].imshow(
        target_overlay(flipped_image, flipped_target)
    )
    axes[1].set_title("Horizontal flip")

    for axis in axes:
        axis.axis("off")

    fig.tight_layout()

    fig.savefig(
        output / "augmentation_example.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


## Pretrained model

def load_pretrained_model(device):
    weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1

    model = maskrcnn_resnet50_fpn(
        weights=weights
    ).to(device)

    model.eval()

    categories = weights.meta["categories"]

    assert model.roi_heads.box_predictor.cls_score.out_features == 91
    assert "intact" not in categories
    assert "broken" not in categories

    return model, weights, categories


## Model preprocessing

def verify_preprocessing(model, image, device):
    assert model.transform.image_mean == [
        0.485, 0.456, 0.406
    ]

    assert model.transform.image_std == [
        0.229, 0.224, 0.225
    ]

    assert model.transform.min_size == (800,)
    assert model.transform.max_size == 1333

    image = plain(image).to(device)

    with torch.no_grad():
        normalized = model.transform.normalize(image)

        mean = image.new_tensor(
            model.transform.image_mean
        )[:, None, None]

        std = image.new_tensor(
            model.transform.image_std
        )[:, None, None]

        assert torch.allclose(
            normalized,
            (image - mean) / std,
        )


## Prediction visualization

def visualize_prediction(
    image,
    target,
    prediction,
    label_names,
    title,
    path,
):
    rgb = (
        plain(image).detach().cpu()
        .mul(255).round().clamp(0, 255)
        .to(torch.uint8)
    )

    keep = prediction["scores"] >= DISPLAY_SCORE

    boxes = prediction["boxes"][keep]
    labels = prediction["labels"][keep]
    scores = prediction["scores"][keep]

    masks = (
        prediction["masks"][keep, 0] >= MASK_THRESHOLD
    )

    overlay = rgb.clone()

    if len(masks):
        overlay = draw_segmentation_masks(
            overlay,
            masks,
            alpha=0.4,
        )

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    axes[0].imshow(rgb.permute(1, 2, 0))
    axes[0].set_title("Input")

    axes[1].imshow(target_overlay(image, target))
    axes[1].set_title("Ground truth")

    axes[2].imshow(overlay.permute(1, 2, 0))
    axes[2].set_title("Prediction")

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.tolist()

        axes[2].add_patch(
            plt.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                color="red",
                linewidth=1,
            )
        )

        axes[2].text(
            x1,
            y1,
            f"{label_names[int(label)]} {float(score):.2f}",
            fontsize=8,
            color="black",
            bbox={
                "facecolor": "white",
                "alpha": 0.8,
                "pad": 1,
            },
        )

    for axis in axes:
        axis.axis("off")

    fig.suptitle(title)
    fig.tight_layout()

    fig.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


## Inference

def run_inference(
    model,
    loader,
    device,
    label_names,
    output,
    tag,
    collect_coco=False,
):
    folder = output / tag
    folder.mkdir(parents=True, exist_ok=True)

    model.eval()

    image_ids = []
    summaries = []
    mask_records = []
    box_records = []

    with torch.inference_mode():
        for batch_index, (images, targets) in enumerate(
            loader, start=1
        ):
            # Forward one image at a time to limit CPU memory.
            for position, (image, target) in enumerate(
                zip(images, targets)
            ):
                image_id = int(target["image_id"])

                filename = loader.dataset.coco.imgs[
                    image_id
                ]["file_name"]

                prediction = model([
                    plain(image).to(device)
                ])[0]

                prediction = {
                    key: value.detach().cpu()
                    for key, value in prediction.items()
                }

                keep = (
                    prediction["scores"] >= DISPLAY_SCORE
                )

                summaries.append({
                    "batch": batch_index,
                    "position": position + 1,
                    "image_id": image_id,
                    "file_name": filename,
                    "ground_truth_instances": len(
                        target["labels"]
                    ),
                    "predictions_returned": len(
                        prediction["scores"]
                    ),
                    "displayed_predictions": int(keep.sum()),
                })

                image_ids.append(image_id)

                if position == 0:
                    visualize_prediction(
                        image=image,
                        target=target,
                        prediction=prediction,
                        label_names=label_names,
                        title=f"{tag} | {filename}",
                        path=folder / (
                            f"batch_{batch_index:02d}_first.png"
                        ),
                    )

                if collect_coco:
                    # Retain all model-returned scores for AP.
                    for index in range(
                        len(prediction["scores"])
                    ):
                        category_id = int(
                            prediction["labels"][index]
                        )

                        assert category_id in (1, 2)

                        common = {
                            "image_id": image_id,
                            "category_id": category_id,
                            "score": float(
                                prediction["scores"][index]
                            ),
                        }

                        binary_mask = (
                            prediction["masks"][index, 0].numpy()
                            >= MASK_THRESHOLD
                        ).astype(np.uint8)

                        rle = mask_utils.encode(
                            np.asfortranarray(binary_mask)
                        )

                        rle["counts"] = rle["counts"].decode(
                            "ascii"
                        )

                        mask_records.append({
                            **common,
                            "segmentation": rle,
                        })

                        x1, y1, x2, y2 = (
                            prediction["boxes"][index].tolist()
                        )

                        box_records.append({
                            **common,
                            "bbox": [
                                x1,
                                y1,
                                x2 - x1,
                                y2 - y1,
                            ],
                        })

                del prediction

    assert len(image_ids) == len(set(image_ids))
    assert set(image_ids) == set(loader.dataset.image_ids)

    save_csv(
        folder / "inference_summary.csv",
        summaries,
    )

    if collect_coco:
        save_json(
            folder / "mask_predictions.json",
            mask_records,
        )

        save_json(
            folder / "box_predictions.json",
            box_records,
        )

    return mask_records, box_records


## Output-layer replacement

def adapt_model(pretrained_model, device):
    torch.manual_seed(SEED)

    model = copy.deepcopy(pretrained_model)

    box_features = (
        model.roi_heads.box_predictor.cls_score.in_features
    )

    model.roi_heads.box_predictor = FastRCNNPredictor(
        box_features,
        num_classes=3,
    )

    mask_features = (
        model.roi_heads.mask_predictor.conv5_mask.in_channels
    )

    hidden_channels = (
        model.roi_heads.mask_predictor.conv5_mask.out_channels
    )

    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        mask_features,
        hidden_channels,
        num_classes=3,
    )

    model.to(device).eval()

    assert model.roi_heads.box_predictor.cls_score.out_features == 3
    assert model.roi_heads.mask_predictor.mask_fcn_logits.out_channels == 3

    return model


## Model-state verification

def model_fingerprint(model):
    digest = hashlib.sha256()

    for name, tensor in model.state_dict().items():
        digest.update(name.encode())

        digest.update(
            tensor.detach().cpu().contiguous().numpy().tobytes()
        )

    return digest.hexdigest()


## COCO evaluation

def evaluate_coco(dataset, records, iou_type):
    with redirect_stdout(io.StringIO()):
        ground_truth = COCO()

        ground_truth.dataset = copy.deepcopy(
            dataset.coco.dataset
        )

        ground_truth.dataset.setdefault("info", {})
        ground_truth.createIndex()

        if records:
            predictions = ground_truth.loadRes(
                copy.deepcopy(records)
            )
        else:
            predictions = COCO()

            predictions.dataset = {
                "images": copy.deepcopy(
                    ground_truth.dataset["images"]
                ),
                "categories": copy.deepcopy(
                    ground_truth.dataset["categories"]
                ),
                "annotations": [],
            }

            predictions.createIndex()

        evaluator = COCOeval(
            ground_truth,
            predictions,
            iouType=iou_type,
        )

        evaluator.params.imgIds = dataset.image_ids
        evaluator.params.catIds = [1, 2]

        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    metrics = {
        "AP": float(evaluator.stats[0]),
        "AP50": float(evaluator.stats[1]),
        "AP75": float(evaluator.stats[2]),
        "AR100": float(evaluator.stats[8]),
    }

    for index, category_id in enumerate(
        evaluator.params.catIds
    ):
        precision = evaluator.eval["precision"][
            :, :, index, 0, -1
        ]

        valid = precision[precision > -1]

        metrics[f"AP_{CLASS_NAMES[category_id]}"] = (
            float(valid.mean())
            if valid.size
            else None
        )

    return metrics


## Loss evaluation

def average_loss(model, loader, device, output, split):
    fingerprint = model_fingerprint(model)

    # The training-mode forward branch returns detection losses.
    model.train()

    for module in model.modules():
        if isinstance(
            module,
            (
                torch.nn.modules.batchnorm._BatchNorm,
                torch.nn.modules.dropout._DropoutNd,
            ),
        ):
            module.eval()

    rows = []

    cuda_devices = (
        [device.index or 0]
        if device.type == "cuda"
        else []
    )

    try:
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(SEED)

            if device.type == "cuda":
                torch.cuda.manual_seed_all(SEED)

            with torch.no_grad():
                for images, targets in loader:
                    for image, target in zip(images, targets):
                        model_target = {
                            key: plain(value).to(device)
                            for key, value in target.items()
                        }

                        losses = model(
                            [plain(image).to(device)],
                            [model_target],
                        )

                        assert losses

                        assert all(
                            torch.isfinite(value).all()
                            for value in losses.values()
                        )

                        row = {
                            "image_id": int(target["image_id"])
                        }

                        row.update({
                            key: float(value)
                            for key, value in losses.items()
                        })

                        row["total_loss"] = sum(
                            float(value)
                            for value in losses.values()
                        )

                        rows.append(row)

    finally:
        model.eval()

    assert len(rows) == len(loader.dataset)

    assert len({
        row["image_id"]
        for row in rows
    }) == len(loader.dataset)

    assert model_fingerprint(model) == fingerprint

    averages = {
        key: float(np.mean([
            row[key]
            for row in rows
        ]))
        for key in rows[0]
        if key != "image_id"
    }

    save_csv(
        output / f"{split}_per_image_losses.csv",
        rows,
    )

    return averages


## Experiment

def main(dataset_root, output_root=None):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    root = Path(dataset_root).expanduser().resolve()

    output = (
        Path(output_root).expanduser().resolve()
        if output_root
        else root / "reports" / "update_baseline"
    )

    output.mkdir(parents=True, exist_ok=True)

    split_data = validate_dataset(root, output)

    train_dataset = BeanDataset(root, "train")
    val_dataset = BeanDataset(root, "val")

    # Augmentation is defined separately from baseline evaluation.
    train_aug_dataset = BeanDataset(
        root,
        "train",
        transforms=create_train_transforms(),
    )

    assert len(train_aug_dataset) == len(train_dataset)

    image, target = train_dataset[0]

    assert image.dtype == torch.float32
    assert 0 <= image.min() and image.max() <= 1

    assert (
        len(target["boxes"])
        == len(target["labels"])
        == len(target["masks"])
    )

    assert target["masks"].shape[-2:] == image.shape[-2:]

    train_loader = create_loader(train_dataset)
    val_loader = create_loader(val_dataset)

    visualize_annotations(train_dataset, output)
    visualize_augmentation(train_dataset, output)

    pretrained_model, weights, coco_categories = (
        load_pretrained_model(device)
    )

    verify_preprocessing(
        pretrained_model,
        image,
        device,
    )

    save_json(
        output / "run_config.json",
        {
            "seed": SEED,
            "device": str(device),
            "weights": weights.name,
            "loader_batch_size": BATCH_SIZE,
            "forward_batch_size": 1,
            "display_score_threshold": DISPLAY_SCORE,
            "mask_threshold": MASK_THRESHOLD,
            "image_mean": pretrained_model.transform.image_mean,
            "image_std": pretrained_model.transform.image_std,
            "min_size": list(
                pretrained_model.transform.min_size
            ),
            "max_size": pretrained_model.transform.max_size,
            "evaluation_augmentation": False,
            "training_steps": 0,
        },
    )

    ## Unchanged pretrained inference

    run_inference(
        model=pretrained_model,
        loader=train_loader,
        device=device,
        label_names=coco_categories,
        output=output,
        tag="pretrained_train",
    )

    ## Untrained adapted inference

    adapted_model = adapt_model(
        pretrained_model,
        device,
    )

    fingerprint = model_fingerprint(adapted_model)

    assert (
        pretrained_model.roi_heads.box_predictor
        .cls_score.out_features == 91
    )

    train_masks, train_boxes = run_inference(
        model=adapted_model,
        loader=train_loader,
        device=device,
        label_names=CLASS_NAMES,
        output=output,
        tag="adapted_train",
        collect_coco=True,
    )

    val_masks, val_boxes = run_inference(
        model=adapted_model,
        loader=val_loader,
        device=device,
        label_names=CLASS_NAMES,
        output=output,
        tag="adapted_val",
        collect_coco=True,
    )

    assert model_fingerprint(adapted_model) == fingerprint

    ## Baseline metrics

    metrics = {}

    for split, dataset, masks, boxes in (
        (
            "train",
            train_dataset,
            train_masks,
            train_boxes,
        ),
        (
            "val",
            val_dataset,
            val_masks,
            val_boxes,
        ),
    ):
        metrics[split] = {
            "mask": evaluate_coco(
                dataset,
                masks,
                "segm",
            ),
            "box": evaluate_coco(
                dataset,
                boxes,
                "bbox",
            ),
        }

    save_json(
        output / "baseline_metrics.json",
        metrics,
    )

    ## Baseline losses

    losses = {
        "train": average_loss(
            adapted_model,
            train_loader,
            device,
            output,
            "train",
        ),
        "val": average_loss(
            adapted_model,
            val_loader,
            device,
            output,
            "val",
        ),
    }

    save_json(
        output / "baseline_losses.json",
        losses,
    )

    assert model_fingerprint(adapted_model) == fingerprint

    ## Export summary

    summary = []

    for split in ("train", "val"):
        summary.append({
            "split": split,
            "photographs": len(
                split_data[split]["images"]
            ),
            "mask_AP": metrics[split]["mask"]["AP"],
            "mask_AP50": metrics[split]["mask"]["AP50"],
            "mask_AP75": metrics[split]["mask"]["AP75"],
            "mask_AP_intact": metrics[split]["mask"]["AP_intact"],
            "mask_AP_broken": metrics[split]["mask"]["AP_broken"],
            "box_AP": metrics[split]["box"]["AP"],
            "average_total_loss": losses[split]["total_loss"],
        })

    save_csv(
        output / "baseline_summary.csv",
        summary,
    )


## Command-line entry

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Dry-bean dataset checks and untrained "
            "Mask R-CNN baseline evaluation."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("dry_bean_dataset_corrected_196"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    arguments = parser.parse_args()

    main(
        dataset_root=arguments.dataset,
        output_root=arguments.output,
    )


## References

# He et al. (2017). Mask R-CNN.
# https://doi.org/10.1109/ICCV.2017.322
#
# Torchvision implementation:
# https://github.com/pytorch/vision
#
# Dataset and augmentation tutorial:
# https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html
#
# COCO evaluation:
# https://github.com/cocodataset/cocoapi
