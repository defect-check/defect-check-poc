from mrcnn.config import Config
from .dataset import CLASS_NAME

class CustomConfig(Config):
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
    LEARNING_RATE=0.0001
