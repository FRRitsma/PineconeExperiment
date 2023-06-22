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

from src.extract import ImageWithMetadata


def load_image_from_file(image_name: str | Path) -> Union[JpegImageFile, PngImageFile]:
    img = Image.open(image_name)
    return img


class Embedder:
    """
    For embedding a PIL image as a torch.Tensor.
    Implemented as class to prevent continuous reloading of the sizeable resnet model
    """

    embedding_dimension: int = 2048

    def __init__(self):
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        resnet = models.resnet101(weights=ResNet101_Weights.DEFAULT).to(self.device)
        modules = list(resnet.children())[:-1]
        self.resnet = torch.nn.Sequential(*modules)

    def embed(self, image_with_metadata: ImageWithMetadata) -> np.ndarray:
        with Image.open(image_with_metadata.image_path) as img:
            transformed_img = image_transform(img).to(self.device)
        return (
            self.resnet(transformed_img.unsqueeze(0)).flatten().detach().cpu().numpy()
        )


def image_transform_pad_to_square(
    img: Union[JpegImageFile, PngImageFile] | torch.Tensor
):
    size = max(np.array(img).shape)
    width, height = img.size
    delta_w = size - width
    delta_h = size - height
    left = delta_w // 2
    right = delta_w - left
    top = delta_h // 2
    bottom = delta_h - top
    transform = transforms.Pad(padding=(left, top, right, bottom))
    return transform(img)


def image_transform_one_channel_to_three_channels(img: torch.Tensor) -> torch.Tensor:
    if min(img.shape) == 3:
        return img
    gray_transform = transforms.Grayscale()
    copy_transform = transforms.Lambda(lambda x: torch.cat([x, x, x], 0))
    transform = transforms.Compose([gray_transform, copy_transform])
    return transform(img)


def image_transform_to_tensor_and_resize(img: torch.Tensor, size: int) -> torch.Tensor:
    resize_and_transform_to_tensor = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
        ]
    )
    return resize_and_transform_to_tensor(img)


def image_transform_normalize(img: torch.Tensor) -> torch.Tensor:
    transform = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    return transform(img)


def image_transform(img: Union[JpegImageFile, PngImageFile]) -> torch.Tensor:
    IMAGE_RESIZE: int = 256

    img = image_transform_pad_to_square(img)
    img = image_transform_to_tensor_and_resize(img, IMAGE_RESIZE)
    img = image_transform_one_channel_to_three_channels(img)
    img = image_transform_normalize(img)

    return img
