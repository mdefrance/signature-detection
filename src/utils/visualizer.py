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
