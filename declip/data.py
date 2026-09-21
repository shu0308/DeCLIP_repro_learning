import json
from pathlib import Path
from typing import Optional, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset

from .protocols import COCO_CLASSES, VOC_CLASSES


class VOCMultiLabel(Dataset):
    """VOC2007 classification labels with task-level partial-label filtering."""

    def __init__(
        self,
        root: str,
        split: str,
        transform=None,
        current_classes: Optional[Sequence[int]] = None,
        require_current_positive: bool = False,
    ):
        self.root = Path(root)
        self.transform = transform
        main = self.root / "ImageSets" / "Main"
        ids = (main / f"{split}.txt").read_text().split()
        labels = torch.empty(len(ids), len(VOC_CLASSES), dtype=torch.float32)
        id_to_row = {image_id: i for i, image_id in enumerate(ids)}
        for class_idx, class_name in enumerate(VOC_CLASSES):
            entries = (main / f"{class_name}_{split}.txt").read_text().splitlines()
            for entry in entries:
                image_id, value = entry.split()
                if image_id in id_to_row:
                    # VOC uses 1 positive, -1 negative, and occasionally 0 difficult.
                    labels[id_to_row[image_id], class_idx] = float(int(value) == 1)
        if require_current_positive:
            if not current_classes:
                raise ValueError("current_classes is required when filtering training data")
            keep = labels[:, list(current_classes)].sum(1) > 0
            indices = keep.nonzero(as_tuple=False).flatten().tolist()
            ids = [ids[i] for i in indices]
            labels = labels[indices]
        self.ids, self.labels = ids, labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image = Image.open(self.root / "JPEGImages" / f"{self.ids[index]}.jpg").convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[index], self.ids[index]



class COCOMultiLabel(Dataset):
    """Multi-label classification view of an extracted COCO 2017 dataset."""

    def __init__(self, root: str, split: str, transform=None,
                 current_classes: Optional[Sequence[int]] = None,
                 require_current_positive: bool = False):
        if split not in {"train2014", "val2014", "train2017", "val2017"}:
            raise ValueError("COCO split must be train2014/val2014 or train2017/val2017")
        self.root, self.split, self.transform = Path(root), split, transform
        annotation_file = self.root / "annotations" / f"instances_{split}.json"
        if not annotation_file.is_file():
            raise FileNotFoundError(f"COCO annotation file not found: {annotation_file}")
        with annotation_file.open("r", encoding="utf-8") as handle:
            annotations = json.load(handle)
        name_to_index = {name: idx for idx, name in enumerate(COCO_CLASSES)}
        category_to_index = {category["id"]: name_to_index[category["name"]]
                             for category in annotations["categories"]
                             if category["name"] in name_to_index}
        if len(category_to_index) != len(COCO_CLASSES):
            present = {category["name"] for category in annotations["categories"]}
            raise ValueError("COCO annotations are missing categories: "
                             f"{sorted(set(COCO_CLASSES) - present)}")
        images = sorted(annotations["images"], key=lambda item: item["id"])
        image_to_row = {item["id"]: row for row, item in enumerate(images)}
        labels = torch.zeros(len(images), len(COCO_CLASSES), dtype=torch.float32)
        for annotation in annotations["annotations"]:
            row = image_to_row.get(annotation["image_id"])
            class_index = category_to_index.get(annotation["category_id"])
            if row is not None and class_index is not None:
                labels[row, class_index] = 1.0
        if require_current_positive:
            if not current_classes:
                raise ValueError("current_classes is required when filtering COCO data")
            indices = (labels[:, list(current_classes)].sum(1) > 0).nonzero(
                as_tuple=False).flatten().tolist()
            images = [images[index] for index in indices]
            labels = labels[indices]
        self.images, self.labels = images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        item = self.images[index]
        image = Image.open(self.root / self.split / item["file_name"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[index], str(item["id"])
