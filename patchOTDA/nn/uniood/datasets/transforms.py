
from ..models import CLIP_MODELS

def build_transform(image_augmentation,
                    backbone_name,
                    **kwargs):
    """Build transformation function.

    Args:
        image_augmentation (str): name of image augmentation method. If none, just use center crop.
    """

    if image_augmentation == "none":
        transform = lambda x: x

    return transform