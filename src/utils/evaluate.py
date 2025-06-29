"""Utility functions for evaluating object detection models with COCO format."""

import torch

import time
import numpy as np
from coco_eval import CocoEvaluator
from tqdm.notebook import tqdm


def convert_to_xywh(boxes):
    """Convert bounding boxes from (xmin, ymin, xmax, ymax) to (xmin, ymin, width, height)."""
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def prepare_for_coco_detection(predictions):
    """Convert model predictions to COCO format."""
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        boxes = prediction["boxes"]
        boxes = convert_to_xywh(boxes).tolist()
        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results


def evaluate_model(coco_gt, dataloader, model, image_processor):
    """Evaluate the model on the given dataloader."""

    # getting model device and batch size
    batch_size = dataloader.batch_size
    device = model.device

    # initiating coco evaluator
    evaluator = CocoEvaluator(coco_gt=coco_gt, iou_types=["bbox"])
    times = []
    print("Running evaluation...")

    # iterating over batches
    for idx, batch in enumerate(tqdm(dataloader)):

        # getting image and targets
        pixel_values = batch["pixel_values"].to(device)
        labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

        # timed inference
        tic = time.time()
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
        times.append(time.time() - tic)

        # preparing outputs
        orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
        results = image_processor.post_process_object_detection(
            outputs, target_sizes=orig_target_sizes, threshold=0
        )

        # converting outputs to COCO format
        predictions = {target["image_id"].item(): output for target, output in zip(labels, results)}
        predictions = prepare_for_coco_detection(predictions)

        # updating evaluator with predictions
        evaluator.update(predictions)

    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()
    print(f"Average inference time per batch: {np.mean(times):.4f} seconds")
    print(f"Average inference time: {np.mean(times)/batch_size:.4f} seconds")

    # getting total number of parameters in the model
    total_params = sum(p.numel() for p in model.parameters())

    return {
        "mAP50-95": evaluator.coco_eval["bbox"].stats[0],
        "mAP50": evaluator.coco_eval["bbox"].stats[1],
        "device": str(device),
        "batch_size": batch_size,
        "inference_time_per_batch": np.mean(times),
        "inference_time": np.mean(times) / batch_size,
        "total_params": total_params,
    }
