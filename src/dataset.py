import os
import json
import numpy as np
from skimage.io import imread
from skimage.draw import polygon
import random
from mrcnn import utils
from .line_to_poly import line_to_poly

# The names of the classes used for this model
CLASS_NAME = {"sagging conductor": 1, "good conductor": 2}
REGION_ATTRIBUTE = "conductor"
SUB_DIRECTORY = "Compressed"

VIA_PROJECT_JSON = "via_region_data.json"

_seed = random.randint(0,500)
def split_train_test_validate(dataset: list, ratio=(5, 1, 1)):
    """
    Splits a dataset into training, testing and validation sets.
    dataset: The list of training items
    ratio: A tuple of 3 items showing the ratio in which to split the items
    """
    dataset = dataset.copy()
    random.Random(_seed).shuffle(dataset)
    num_dataset = len(dataset)
    guide = []
    total = ratio[0] + ratio[1] + ratio[2]
    ratio = [round(num_dataset * ratio[0] / total),
             round(num_dataset * ratio[1] / total), 0]
    ratio[2] = num_dataset - ratio[0] - ratio[1]
    for i in range(3):
        for _ in range(ratio[i]):
            guide.append(i)

    def collect(j):
        return list(i[1] for i in filter(
            lambda i: guide[i[0]] == j, enumerate(dataset)))
    return {
        "train": collect(0),
        "test": collect(1),
        "val": collect(2),
    }

class CustomDataset(utils.Dataset):
    """
    The CustomDataset class wraps utility methods for working with data.
    """
    def load_custom(self, dataset_dir, subset):
        """Load a subset of the dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes (change name as needed).
        for class_name, key in CLASS_NAME.items():
            self.add_class(class_name, key, class_name)

        # Train or validation dataset?
        assert subset in ["train", "val"]

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
            open(os.path.join(dataset_dir, VIA_PROJECT_JSON), encoding="utf-8"))
        # don't need the dict keys
        annotations = list(annotations_dict.values())

        annotations = split_train_test_validate(annotations)[subset]
        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a["regions"]]

        # Add images
        for item in annotations:
            # print(item)
            # Get the x, y coordinates of points of the shapes that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            shapes = [r["shape_attributes"] for r in item["regions"]]
            class_names_each_region = [
                r["region_attributes"][REGION_ATTRIBUTE]
                for r in item["regions"]
            ]

            # load_mask() needs the image size to convert shapes to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, SUB_DIRECTORY, item["filename"])
            image = imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                source=class_names_each_region,
                image_id=item["filename"],  # use file name as a unique image id
                path=image_path,
                width=width,
                height=height,
                shapes=shapes,
            )

    def load_mask(self, image_id):
        """Generate instance masks for an image.
        Returns:
         masks: A bool array of shape [height, width, instance count] with
             one mask per instance.
         class_ids: a 1D array of class IDs of the instance masks.
        """
        
        # Convert shapes to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros(
            [info["height"], info["width"], len(info["shapes"])], dtype=np.uint8
        )
        for i, p in enumerate(info["shapes"]):
            # Get indexes of pixels inside the polygon and set them to 1
            points =  p["all_points_y"], p["all_points_x"]
            if p["name"] == "polyline":
                points = line_to_poly(*points,4)
            rr, cc = polygon(*points)
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        # return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)
        class_ids = []
        for cls_name in info["source"]:
            class_ids.append(CLASS_NAME[cls_name])
        return mask.astype(np.bool), np.array(class_ids, dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "objects":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
