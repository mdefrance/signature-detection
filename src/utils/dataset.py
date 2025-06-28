"""Utility functions for converting datasets to COCO format and defining a custom dataset class
for signature detection.
"""

import torchvision


import os
import json
from tqdm import tqdm
from .config import TARGET


def convert_to_coco(dataset, output_dir, split_name, save_images=False, image_dir=None):
    """Convert a dataset to COCO format and save it as JSON."""
    os.makedirs(output_dir, exist_ok=True)
    if save_images:
        image_dir = image_dir or os.path.join(output_dir, "images", split_name)
        os.makedirs(image_dir, exist_ok=True)

    images = []
    annotations = []
    categories = [{"id": 0, "name": TARGET}]
    ann_id = 1  # global annotation ID counter

    for idx, example in tqdm(
        enumerate(dataset), total=len(dataset), desc=f"Processing {split_name}"
    ):
        image_id = example["image_id"]
        width = example["width"]
        height = example["height"]
        file_name = f"{image_id}.jpg"

        if save_images:
            image_path = os.path.join(image_dir, file_name)
            example["image"].save(image_path)

        images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": width,
                "height": height,
            }
        )

        objects = example["objects"]
        for i in range(len(objects["id"])):
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": objects["category"][i],
                    "bbox": objects["bbox"][i],
                    "area": objects["area"][i],
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
        "info": {
            "description": "Signature Dataset",
            "version": "1.0",
            "year": 2025,
            "contributor": "",
            "date_created": "",
        },
        "licenses": [{"id": 1, "name": "Apache 2.0", "url": ""}],
    }

    json_path = os.path.join(output_dir, f"{split_name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(coco_dict, f)


class SignatureDataset(torchvision.datasets.CocoDetection):
    """Custom dataset class for signature detection."""

    def __init__(self, img_folder, ann_file, image_processor):
        """Initialize the SignatureDataset."""
        super().__init__(img_folder, ann_file)
        self.image_processor = image_processor

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        images, annotations = super().__getitem__(idx)

        # preprocess image and target (converting target to DETR format, resizing +
        # normalization of both image and target)
        annotations = {"image_id": self.ids[idx], "annotations": annotations}
        encoding = self.image_processor(images=images, annotations=annotations, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()  # remove batch dimension
        target = encoding["labels"][0]  # remove batch dimension

        return pixel_values, target

