import os
import json
import numpy as np
from skimage.draw import polygon
import random
import torch
import torch.utils.data
from PIL import Image

from .line_to_poly import clip_to_bounds, line_to_poly

# The names of the classes used for this model
CLASS_NAME = {
    "sagging conductor": 1,
    "good conductor": 2,
    "v": 2,
    "sagging": 1,
    "good": 2,
}
REGION_ATTRIBUTE = "conductor"
SUB_DIRECTORY = "Compressed"
# Our palette is used to store the different types of objects in each image
# Each image in the training set must have at least one object or else there will be errors
# This palette can handle 9 different objects (0 is background)
VIA_PROJECT_JSON = "via_region_data.json"


class CustomDataset(torch.utils.data.Dataset):
    """
    The CustomDataset class wraps utility methods for working with data.
    """

    def __init__(self, config, transforms=None):
        self.imgs = []
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        # Load annotations
        # VGG Image Annotator saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        annotations_dict = json.load(
            open(
                os.path.join(config.DATA_DIRECTORY, VIA_PROJECT_JSON), encoding="utf-8"
            )
        )
        # don't need the dict keys
        annotations = list(annotations_dict.values())

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a["regions"]]

        # Add images
        for item in annotations:
            image_path = os.path.join(
                config.DATA_DIRECTORY, SUB_DIRECTORY, item["filename"]
            )

            self.add_image(
                image_id=item["filename"],  # use file name as a unique image id
                path=image_path,
                shapes=item["regions"],
            )

    def add_image(self, **kwargs):
        self.imgs.append(kwargs)

    def __getitem__(self, idx):
        # load images and masks
        img = Image.open(self.imgs[idx]["path"])
        img = img.convert("RGB")
        height, width = img.size

        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = self.load_mask(height, width, idx)
        obj_ids = np.unique(mask)
        # 0 is the background, so remove it
        obj_ids = obj_ids[obj_ids != 0]

        # split the color-encoded mask into a set
        # of binary masks for each class
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)

        # Bounding boxes of each mask
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # Labels of each mask
        labels = torch.as_tensor(obj_ids, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

    def load_mask(self, height: int, width: int, image_id: int):
        """Generate instance masks for an image.
        Returns:
         masks: A bool array of shape [height, width, instance count] with
             one mask per instance.
         class_ids: a 1D array of class IDs of the instance masks.
        """

        # Convert shapes to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.imgs[image_id]
        mask = np.zeros([height, width], dtype=np.float)

        for i, shape in enumerate(info["shapes"]):
            p = shape["shape_attributes"]
            # Get indexes of pixels inside the polygon and set them to 1
            points = p["all_points_y"], p["all_points_x"]
            if p["name"] == "polyline":
                points = clip_to_bounds(line_to_poly(*points, 10), (width, height))
            rr, cc = polygon(*points)
            mask[rr, cc] = CLASS_NAME[shape["region_attributes"]["conductor"]]
        return mask
