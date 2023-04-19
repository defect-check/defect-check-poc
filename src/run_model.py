import torch
from PIL import Image
from .train_model import load_model, get_transform
from src.config import CustomConfig


def run_model(model, image_path):
    """
    This function is in charge or running the defect detect model in application mode
    It accepts an image path as input an returns the location of sagging conductors in
    the image if any.
    """
    img = get_transform(False)(Image.open(image_path).convert("RGB"), {"boxes": []})
    config = CustomConfig()

    model = load_model(config, pretrained=True)
    model = model.eval()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    with torch.no_grad():
        prediction = model([img.to(device)])

    mask = Image.fromarray(
        255 - prediction[0]["masks"][3, 0].mul(255).byte().cpu().numpy(), "L"
    )
    target = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy(), "RGB")

    red = Image.new("RGB", target.size, (255, 0, 0))

    # create a mask using RGBA to define an alpha channel to make the overlay transparent
    # mask = Image.new('RGBA',target.size,(0,0,0,123))

    return Image.composite(target, red, mask)
