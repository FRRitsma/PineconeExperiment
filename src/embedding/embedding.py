# %%
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from PIL.JpegImagePlugin import JpegImageFile as JpegImageFile
from PIL.PngImagePlugin import PngImageFile
from torchvision.models import ResNet101_Weights


def load_image_from_file(image_name: str | Path) -> Union[JpegImageFile, PngImageFile]:
    """
    Returns a locally stored image as a PIL image

    Args:
        image_name (str | Path):

    Returns:
        JpegImageFile:
    """
    img = Image.open(image_name)
    return img


class Embedder:
    """
    For embedding a PIL image as a torch.Tensor.
    Implemented as class to prevent continuous reloading of the sizeable resnet model
    """

    def __init__(self):
        resnet = models.resnet101(weights=ResNet101_Weights.DEFAULT)
        modules = list(resnet.children())[:-1]
        self.resnet = torch.nn.Sequential(*modules)

    def embed(self, img: Union[JpegImageFile, PngImageFile]) -> np.ndarray:
        img = image_transform(img)
        return self.resnet(img.unsqueeze(0)).flatten().detach().numpy()


def image_transform(img: Union[JpegImageFile, PngImageFile]) -> torch.Tensor:
    image_resize: int = 256
    transform = transforms.Compose(
        [
            transforms.Resize(image_resize),
            transforms.CenterCrop(
                image_resize
            ),  # TODO: Center crop seems to destroy information. Is there an alternative?
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(img)
