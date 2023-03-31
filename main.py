"""
Mask R-CNN in Keras with TensorFlow backend.
Train on a single or multiple class dataset and
run inference on image, webcam or video.
Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
Modified by Micheleen Harris (2020)
Modified by Owologba Oro (2022)
"""
# Disable formating so black does not mess up sys.path ordering
# fmt: off
import os
import sys
# Root directory of the project
ROOT_DIR = os.path.abspath(".")

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Import Mask RCNN
sys.path.append(ROOT_DIR)
from mrcnn.utils import download_trained_weights
from mrcnn.model import MaskRCNN
from src.detect_defects import detect_defects
from src.train_model import train_model
from src import config
from src.config import CustomConfig
# fmt: on


# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Path to dataset
DATA_DIRECTORY = os.path.join(ROOT_DIR, "data")

# Directory to save logs and model checkpoints
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect good and bad conductor sagging.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train', 'detect'")
    parser.add_argument('--weights',
                        metavar="/path/to/weights.h5",
                        required=False,
                        default="coco",
                        help="'coco', 'last', 'imagenet' or <Path to weights .h5 file>")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to detect defects on')
    parser.add_argument('--epochs', required=False,
                        default=40,
                        type=int,
                        help='Number of epochs to train on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "detect":
        assert args.image or args.video,\
            "Provide --image to apply detction on."

    print("Weights: ", args.weights)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = CustomConfig()
    else:
        class InferenceConfig(CustomConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = MaskRCNN(mode="training", config=config,
                         model_dir=args.logs)
    else:
        model = MaskRCNN(mode="inference", config=config,
                         model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()[1]
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or run inference
    if args.command == "train":
        train_model(model, DATA_DIRECTORY, config, args.epochs)
    elif args.command == 'detect':
        detect_defects(model, image_path=args.image)
    else:
        print(f"'{args.command}' is not recognized. "
              "Use 'train', or 'detect'")
