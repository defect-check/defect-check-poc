import torch
from PIL import Image
from .train_model import load_model, get_transform


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

    red = Image.new("RGB", target.size, (255, 0, 0))
    green = Image.new("RGB", target.size, (0, 255, 0))

    for i in range(len(prediction[0]["labels"])):
        mask = Image.fromarray(
            255
            - prediction[0]["masks"][i, 0]
            .mul(255 * prediction[0]["scores"][i])
            .byte()
            .cpu()
            .numpy(),
            "L",
        )
        color = red if prediction[0]["labels"][i] == 1 else green
        target = Image.composite(target, color, mask)

    return target
