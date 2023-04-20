from PIL import Image
from .config import PALETTE


def compose_masks(target, data, use_scores=False, palette=PALETTE, cache={}):
    result = target
    for i in range(len(data["labels"])):
        score = data["scores"][i] if use_scores else 1
        label = data["labels"][i]
        if score < 0.2:
            continue
        mask = Image.fromarray(
            255 - data["masks"][i, 0].mul(255 * score).byte().cpu().numpy(), "L"
        )

        if label not in cache:
            cache[label] = Image.new("RGB", target.size, PALETTE[int(label)])
        result = Image.composite(result, cache[label], mask)
    return result
