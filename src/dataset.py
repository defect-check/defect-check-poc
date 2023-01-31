from mrcnn import utils

# The names of the classes used for this model
CLASS_NAME = {"sagging conductor": 1, "good conductor": 2}

VIA_PROJECT_JSON = "via_region_data.json"


def split_train_test_validation(dataset, ratio=[5, 1, 1]):
    m = len(dataset)
    guide = []
    total = ratio[0] + ratio[1] + ratio[2]
    ratio = [round(m * ratio[0] / total), round(m * ratio[1] / total), 0]
    ratio[2] = m - ratio[0] - ratio[1]
    for i in range(ratio):
        for j in range(ratio[i]):
            guide.append(i)
    return {
        "train": filter(lambda i: guide[i[0]] == "train", enumerate(dataset)),
        "test": filter(lambda i: guide[i[0]] == "test", enumerate(dataset)),
        "val": filter(lambda i: guide[i[0]] == "val", enumerate(dataset)),
    }


class CustomDataset(utils.Dataset):
    def load_custom(self, dataset_dir, subset):
        """Load a subset of the dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes (change name as needed).
        for class_name in CLASS_NAME.keys():
            self.add_class(class_name, CLASS_NAME[class_name], class_name)

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
        annotations_dict = json.load(open(os.path.join(dataset_dir, VIA_PROJECT_JSON)))
        annotations = list(annotations_dict.values())  # don't need the dict keys

        annotations = split_train_test_validation(annotations)[subset]
        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a["regions"]]

        # Add images
        for a in annotations:
            # print(a)
            # Get the x, y coordinates of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            polygons = [r["shape_attributes"] for r in a["regions"]]
            class_names_each_region = [
                r["region_attributes"]["type"]
                if "type" in r["region_attributes"]
                else r["region_attributes"]["category_id"]
                for r in a["regions"]
            ]

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a["filename"])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                source=class_names_each_region,
                image_id=a["filename"],  # use file name as a unique image id
                path=image_path,
                width=width,
                height=height,
                polygons=polygons,
            )

    def load_mask(self, image_id):
        """Generate instance masks for an image.
        Returns:
         masks: A bool array of shape [height, width, instance count] with
             one mask per instance.
         class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a class1 dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        # if image_info["source"] != "damage":
        #     return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros(
            [info["height"], info["width"], len(info["polygons"])], dtype=np.uint8
        )
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p["all_points_y"], p["all_points_x"])
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
