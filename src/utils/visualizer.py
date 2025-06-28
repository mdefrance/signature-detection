import numpy as np

from PIL import Image, ImageDraw


def convert_tensor_to_pil_image(pixel_values):
    """Convert pixel values to PIL images."""

    tensor_image = (pixel_values - pixel_values.min()) / (
        pixel_values.max() - pixel_values.min()
    )  # Normalize to [0, 1]
    numpy_image = tensor_image.permute(1, 2, 0).numpy()  # Change from (C, H, W) to (H, W, C)
    numpy_image = (numpy_image * 255).astype(np.uint8)  # Scale to [0, 255]
    return Image.fromarray(numpy_image)


def add_label_bbox(bboxes, image):
    """Add label and bounding box to the item."""
    # getting image dimensions
    image_width, image_height = image.size

    # iterate through each bounding box
    for x_center, y_center, width, height in bboxes:
        # fixing the bounding box coordinates/dimensions
        x_center *= image_width
        y_center *= image_height
        width *= image_width
        height *= image_height

        # calculate the bounding box coordinates
        x_min = (x_center - width / 2) / image_width
        y_min = (y_center - height / 2) / image_height
        x_max = (x_center + width / 2) / image_width
        y_max = (y_center + height / 2) / image_height

        # draw the bounding box
        draw = ImageDraw.Draw(image)
        draw.rectangle(
            [x_min * image_width, y_min * image_height, x_max * image_width, y_max * image_height],
            outline="red",
            width=2,
        )

        # add label text
        draw.text((x_min * image_width, y_min * image_height), "signature", fill="red")

    return image


def visualize_item(item):
    """Visualize a single item from the dataset."""
    image = convert_tensor_to_pil_image(item[0])

    return add_label_bbox(item[1]["boxes"], image)


def _visualize_pred(image, predictions):
    """
    Display an image with bounding boxes drawn from the predictions.

    Args:
        image (PIL.Image): The input image.
        predictions (list): A list of dictionaries containing the bounding box coordinates and other details.
    """
    # Create a draw object
    draw = ImageDraw.Draw(image)

    # Draw each bounding box
    for prediction in predictions:
        box = prediction["box"]
        xmin = box["xmin"]
        ymin = box["ymin"]
        xmax = box["xmax"]
        ymax = box["ymax"]
        label = prediction["label"]
        score = prediction["score"]

        # Draw the bounding box
        draw.rectangle([xmin, ymin, xmax, ymax], outline="blue", width=2)

        # Optionally, add a label and score
        label_text = f"{label}: {score:.2f}"
        draw.text((xmin, ymin), label_text, fill="blue")

    return image


def visualize_prediction(item, prediction):
    """Visualize predictions alongside the expected item."""

    # getting expected and predicted images
    expected_image = visualize_item(item)
    predicted_image = _visualize_pred(convert_tensor_to_pil_image(item[0]), prediction)

    # Create a new image with combined width
    combined = Image.new(
        "RGB",
        (
            expected_image.width + predicted_image.width,
            max(expected_image.height, predicted_image.height),
        ),
    )
    combined.paste(expected_image, (0, 0))
    combined.paste(predicted_image, (expected_image.width, 0))
    return combined
