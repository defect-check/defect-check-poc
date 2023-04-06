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
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
# Root directory of the project
ROOT_DIR = os.path.abspath(".")

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Import Mask RCNN
sys.path.append(ROOT_DIR)
from src.run_model import run_model
from src.train_model import train_model
from src import config
from src.config import CustomConfig
# fmt: on


def load_model(config, pretrained):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained)
    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, config.NUM_CLASSES)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, config.NUM_CLASSES
    )


if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train Mask R-CNN to detect good and bad conductor sagging."
    )
    parser.add_argument("command", metavar="<command>", help="'train', 'detect'")
    parser.add_argument(
        "--weights",
        metavar="/path/to/weights.h5",
        required=False,
        default="coco",
        help="'coco', 'last', 'imagenet' or <Path to weights .h5 file>",
    )
    parser.add_argument(
        "--logs",
        required=False,
        default=config.DEFAULT_LOGS_DIR,
        metavar="/path/to/logs/",
        help="Logs and checkpoints directory (default=logs/)",
    )
    parser.add_argument(
        "--image",
        required=False,
        metavar="path or URL to image",
        help="Image to detect defects on",
    )
    parser.add_argument(
        "--epochs",
        required=False,
        default=40,
        type=int,
        help="Number of epochs to train on",
    )
    args = parser.parse_args()

    # Validate arguments
    if args.command == "detect":
        assert args.image or args.video, "Provide --image to apply detction on."

    print("Weights: ", args.weights)
    print("Logs: ", args.logs)

    config = CustomConfig()
    # Load weights
    weights_path = args.weights
    print("Loading weights ", weights_path)
    model = load_model(config, pretrained=(not weights_path))
    if weights_path:
        model.load_state_dict(torch.load(weights_path))
    model.eval()

    # Train or run inference
    if args.command == "train":
        train_model(model, config, args.epochs)
    elif args.command == "detect":
        run_model(model, config, image_path=args.image)
    else:
        print(f"'{args.command}' is not recognized. " "Use 'train', or 'detect'")
