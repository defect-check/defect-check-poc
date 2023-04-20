import torch
from PIL import Image
from .train_model import load_model, get_transform
from .visualize import compose_masks


def run_model(model, image_path):
    """
    This function is in charge or running the defect detect model in application mode
    It accepts an image path as input an returns the location of sagging conductors in
    the image if any.
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    img, _ = get_transform(False)(Image.open(image_path).convert("RGB"), {"boxes": []})
    model = model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])

    target = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy(), "RGB")

    return compose_masks(target, prediction, use_scores=True)
