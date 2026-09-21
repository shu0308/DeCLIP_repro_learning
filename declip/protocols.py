from dataclasses import dataclass
from typing import List

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
    "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]

# KRT/CSC and DeCLIP use category-name lexicographic order for COCO stages.
COCO_CLASSES = sorted(COCO_CLASSES)



VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]


@dataclass(frozen=True)
class IncrementalProtocol:
    name: str
    classes: List[str]
    base: int
    increment: int

    @property
    def tasks(self) -> List[List[int]]:
        first = self.base if self.base > 0 else self.increment
        result = [list(range(0, first))]
        for start in range(first, len(self.classes), self.increment):
            result.append(list(range(start, min(start + self.increment, len(self.classes)))))
        return result

    def seen(self, task_id: int) -> List[int]:
        return [c for task in self.tasks[: task_id + 1] for c in task]


def voc_protocol(name: str) -> IncrementalProtocol:
    specs = {"b0-c4": (0, 4), "b4-c2": (4, 2)}
    key = name.lower()
    if key not in specs:
        raise ValueError(f"Unknown VOC protocol {name!r}; choose from {sorted(specs)}")
    base, increment = specs[key]
    return IncrementalProtocol(key, VOC_CLASSES, base, increment)



def coco_protocol(name: str) -> IncrementalProtocol:
    specs = {"b0-c10": (0, 10), "b40-c10": (40, 10)}
    key = name.lower()
    if key not in specs:
        raise ValueError(f"Unknown COCO protocol {name!r}; choose from {sorted(specs)}")
    base, increment = specs[key]
    return IncrementalProtocol(key, COCO_CLASSES, base, increment)
