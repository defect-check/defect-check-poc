import os
import sys
import random
import itertools
import colorsys

import numpy as np
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches, lines
from matplotlib.patches import Polygon
import IPython.display
from PIL import Image
from .config import PALETTE
from . import utils


def compose_masks_predicted(target, data, palette=PALETTE, cache={}):
    result = target
    for i in range(len(data["labels"])):
        score = data["scores"][i]
        label = data["labels"][i]
        if score < 0.2:
            continue
        mask = Image.fromarray(
            255 - data["masks"][i, 0].mul(255 * score).byte().cpu().numpy(), "L"
        )

        if label not in cache:
            cache[label] = Image.new("RGB", target.size, PALETTE[int(label)])
        result = Image.composite(result, cache[label], mask)
    return result

def compose_masks_expected(target, data, palette=PALETTE, cache={}):
    result = target
    for i in range(len(data["labels"])):
        label = data["labels"][i]
        mask = Image.fromarray(
            255 - data["masks"][i].mul(255).byte().cpu().numpy(), "L"
        )

        if label not in cache:
            cache[label] = Image.new("RGB", target.size, PALETTE[int(label)])
        result = Image.composite(result, cache[label], mask)
    return result


"""
Mask R-CNN
Display and Visualization Functions.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

############################################################
#  Visualization
############################################################


def display_images(
    images, titles=None, cols=4
):
    """Display the given set of images, optionally with titles.
    images: list or array of image tensors in HWC format.
    titles: optional. A list of titles to display with each image.
    cols: number of images per row
    cmap: Optional. Color map to use. For example, "Blues".
    norm: Optional. A Normalize instance to map values to colors.
    interpolation: Optional. Image interpolation to use for display.
    """
    titles = titles if titles is not None else [""] * len(images)
    rows = len(images) // cols + 1
    plt.figure(figsize=(14, 14 * rows // cols))
    i = 1
    for image, title in zip(images, titles):
        plt.subplot(rows, cols, i)
        plt.title(title, fontsize=9)
        plt.axis("off")
        plt.imshow(image)
        i += 1
    plt.show()


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_instances(
    image,
    boxes,
    masks,
    class_ids,
    class_names,
    scores=None,
    title="",
    figsize=(16, 16),
    ax=None,
    show_mask=True,
    show_mask_polygon=True,
    show_bbox=True,
    colors=None,
    captions=None,
    show_caption=True,
    save_fig_path=None,
):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    show_mask_polygon (Ahmed Gad): Show the mask polygon or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    show_caption (Ahmed Gad): Whether to show the caption or not
    save_fig_path (Ahmed Gad): Path to save the figure
    """

    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis("off")
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                alpha=0.7,  # linestyle="dashed",
                edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(p)

        if show_caption:
            # Label
            if not captions:
                class_id = class_ids[i]
                score = scores[i] if scores is not None else None
                label = class_names[class_id]
                caption = "{} {:.3f}".format(label, score) if score else label
            else:
                caption = captions[i]
            ax.text(x1, y1 + 8, caption, color="w", size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        if show_mask_polygon:
            # Pad to ensure proper polygons for masks that touch image edges.
            padded_mask = np.zeros(
                (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8
            )
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
    if not (save_fig_path is None):
        plt.savefig(save_fig_path, bbox_inches="tight")
    if auto_show:
        plt.show()


def display_instances_RPN(
    image,
    boxes,
    title="",
    figsize=(16, 16),
    ax=None,
    show_bbox=True,
    colors=None,
    save_fig_path=None,
):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    title: (optional) Figure title
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    save_fig_path (Ahmed Gad): Path to save the figure
    """

    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis("off")
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=5,
                alpha=0.7,  # linestyle="dashed",
                edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(p)

    ax.imshow(masked_image.astype(np.uint8))
    if not (save_fig_path is None):
        plt.savefig(save_fig_path, bbox_inches="tight")
    if auto_show:
        plt.show()


# TODO: Replace with matplotlib equivalent?
def draw_box(image, box, color):
    """Draw 3-pixel width bounding boxes on the given image array.
    color: list of 3 int values for RGB.
    """
    y1, x1, y2, x2 = box
    image[y1 : y1 + 2, x1:x2] = color
    image[y2 : y2 + 2, x1:x2] = color
    image[y1:y2, x1 : x1 + 2] = color
    image[y1:y2, x2 : x2 + 2] = color
    return image

def draw_boxes(
    image,
    boxes=None,
    refined_boxes=None,
    masks=None,
    captions=None,
    visibilities=None,
    title="",
    ax=None,
):
    """Draw bounding boxes and segmentation masks with different
    customizations.

    boxes: [N, (y1, x1, y2, x2, class_id)] in image coordinates.
    refined_boxes: Like boxes, but draw with solid lines to show
        that they're the result of refining 'boxes'.
    masks: [N, height, width]
    captions: List of N titles to display on each box
    visibilities: (optional) List of values of 0, 1, or 2. Determine how
        prominent each bounding box should be.
    title: An optional title to show over the image
    ax: (optional) Matplotlib axis to draw on.
    """
    # Number of boxes
    assert boxes is not None or refined_boxes is not None
    N = boxes.shape[0] if boxes is not None else refined_boxes.shape[0]

    # Matplotlib Axis
    if not ax:
        _, ax = plt.subplots(1, figsize=(12, 12))

    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    margin = image.shape[0] // 10
    ax.set_ylim(image.shape[0] + margin, -margin)
    ax.set_xlim(-margin, image.shape[1] + margin)
    ax.axis("off")

    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        # Box visibility
        visibility = visibilities[i] if visibilities is not None else 1
        if visibility == 0:
            color = "gray"
            style = "dotted"
            alpha = 0.5
        elif visibility == 1:
            color = colors[i]
            style = "dotted"
            alpha = 1
        elif visibility == 2:
            color = colors[i]
            style = "solid"
            alpha = 1

        # Boxes
        if boxes is not None:
            if not np.any(boxes[i]):
                # Skip this instance. Has no bbox. Likely lost in cropping.
                continue
            y1, x1, y2, x2 = boxes[i]
            p = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                alpha=alpha,
                linestyle=style,
                edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(p)

        # Refined boxes
        if refined_boxes is not None and visibility > 0:
            ry1, rx1, ry2, rx2 = refined_boxes[i].astype(np.int32)
            p = patches.Rectangle(
                (rx1, ry1),
                rx2 - rx1,
                ry2 - ry1,
                linewidth=2,
                edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(p)
            # Connect the top-left corners of the anchor and proposal
            if boxes is not None:
                ax.add_line(lines.Line2D([x1, rx1], [y1, ry1], color=color))

        # Captions
        if captions is not None:
            caption = captions[i]
            # If there are refined boxes, display captions on them
            if refined_boxes is not None:
                y1, x1, y2, x2 = ry1, rx1, ry2, rx2
            ax.text(
                x1,
                y1,
                caption,
                size=11,
                verticalalignment="top",
                color="w",
                backgroundcolor="none",
                bbox={"facecolor": color, "alpha": 0.5, "pad": 2, "edgecolor": "none"},
            )

        # Masks
        if masks is not None:
            mask = masks[:, :, i]
            masked_image = apply_mask(masked_image, mask, color)
            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            padded_mask = np.zeros(
                (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8
            )
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
