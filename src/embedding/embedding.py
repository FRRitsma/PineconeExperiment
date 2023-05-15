# %%
from pathlib import Path

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from PIL.JpegImagePlugin import JpegImageFile as JpegImageFile
from torchvision.models import ResNet101_Weights

# TODO: Fix PIL image types. Currently only accepts jpeg
# Dev only modules:


def load_image_from_file(image_name: str | Path) -> JpegImageFile:
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

    def embed(self, img: JpegImageFile) -> np.ndarray:
        img = image_transform(img)
        return self.resnet(img.unsqueeze(0)).flatten().detach().numpy()


def image_transform(img: JpegImageFile) -> torch.Tensor:

    image_resize: int = 256  # Reported input size for resnet101 is 224
    transform = transforms.Compose(
        [
            transforms.Resize(image_resize),
            transforms.CenterCrop(image_resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(img)


if __name__ == "__main__":
    # Function used only for development:
    # def visualize_transform(img: torch.Tensor) -> None:
    #     plt.imshow(img.permute(1, 2, 0))

    # Random development functions:
    base_directory = Path.cwd().parent.parent
    image_path = Path.joinpath(base_directory, "tests", "input.jpg")
    img = load_image_from_file(image_path)
    img_trans = image_transform(img)
    # visualize_transform(img_trans)
