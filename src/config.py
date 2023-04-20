from .dataset import CLASS_NAME
import os

ROOT_DIR = os.path.abspath(".")

# Color used for visualization
PALETTE = {1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0)}


class CustomConfig:
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """

    # Give the configuration a recognizable name
    NAME = "defect-detect"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes - single image supported only now (including background)
    NUM_CLASSES = 1 + len(CLASS_NAME.keys())  # Background + object(s) of interest

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 10

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    # Input image dim for network
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 448

    # Learning rate
    LEARNING_RATE = 0.0001

    # Number of workers used for training
    NUM_WORKERS = 4

    # Path to dataset
    DATA_DIRECTORY = os.path.join(ROOT_DIR, "data")
    # Directory to save logs and model checkpoints
    LOGS_DIRECTORY = os.path.join(ROOT_DIR, "logs")
