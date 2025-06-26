from torch.utils.data import Dataset, DataLoader


class SignatureDataset(Dataset):
    """Custom dataset class for signature detection."""

    def __init__(self, dataset, processor):
        """Initialize the SignatureDataset."""
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.dataset)

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
