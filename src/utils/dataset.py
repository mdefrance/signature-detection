from torch.utils.data import Dataset
from pycocotools.coco import COCO


class SignatureDataset(Dataset):
    """Custom dataset class for signature detection."""

    def __init__(self, dataset, processor):
        """Initialize the SignatureDataset."""
        self.dataset = dataset
        self.processor = processor

        # Construct COCO-style dict
        self.coco_dict = self.build_coco_dict()
        self.coco = COCO()
        self.coco.dataset = self.coco_dict
        self.coco.createIndex()  # Required to use coco.getAnnIds, etc.

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.dataset)

    def build_coco_dict(self):
        """Build full COCO dictionary from internal dataset."""
        images = []
        annotations = []
        categories = set()
        ann_id = 1

        for item in self.dataset:
            image_id = item["image_id"]
            images.append(
                {
                    "id": image_id,
                    "width": item.get("width", 0),  # Replace with actual width
                    "height": item.get("height", 0),  # Replace with actual height
                    "file_name": item.get("file_name", f"{image_id}.jpg"),
                }
            )

            objs = item["objects"]
            for i in range(len(objs["id"])):
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": objs["category"][i],
                        "bbox": objs["bbox"][i],
                        "area": objs["area"][i],
                        "iscrowd": 0,
                    }
                )
                categories.add(objs["category"][i])
                ann_id += 1

        categories = [{"id": c, "name": str(c)} for c in sorted(categories)]

        return {
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
            "licenses": [{"id": 1, "name": "Unknown", "url": ""}],
        }

    def ensure_coco_format(self, annotations):
        """Ensure the dataset is in COCO format."""
        # initiating annotations in COCO format
        coco_annotations = {"image_id": annotations["image_id"], "annotations": []}

        # iterating over annototed objects
        for object_i in range(len(annotations["objects"]["id"])):
            # getting the annotation for each object
            annotation = {
                "id": annotations["objects"]["id"][object_i],
                "image_id": annotations["image_id"],
                "category_id": annotations["objects"]["category"][object_i],
                "bbox": annotations["objects"]["bbox"][object_i],
                "area": annotations["objects"]["area"][object_i],
                "iscrowd": 0,
            }
            coco_annotations["annotations"].append(annotation)

        return coco_annotations

    def __getitem__(self, idx):
        """Get an item from the dataset."""
        # Get the item from the dataset
        item = self.dataset[idx]
        image = item["image"]
        annotations = self.ensure_coco_format(item)

        # Process the image and annotations
        encoding = self.processor(images=image, annotations=annotations, return_tensors="pt")

        # Remove batch dimension
        pixel_values = encoding["pixel_values"][0].squeeze()
        labels = encoding["labels"][0]

        return pixel_values, labels


def get_collate_fn(processor):
    """Get a custom collate function for the dataset."""

    def collate_fn(batch):
        """Custom collate function to handle batches of data."""
        pixel_values = [item[0] for item in batch]
        encoding = processor.pad(pixel_values, return_tensors="pt")
        labels = [item[1] for item in batch]
        batch = {}
        batch["pixel_values"] = encoding["pixel_values"]
        batch["labels"] = labels
        return batch

    return collate_fn
