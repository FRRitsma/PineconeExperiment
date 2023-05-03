# %%
from pathlib import Path

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from PIL.JpegImagePlugin import JpegImageFile as JpegImageFile

# TODO: Fix PIL image types. Currently only accepts jpeg
# Dev only modules:
if __name__ == "__main__":
    import matplotlib.pyplot as plt


def load_image_from_file(image_name: str | Path) -> JpegImageFile:
    img = Image.open(image_name)
    return img


class Embedder:
    def __init__(self):
        resnet = models.resnet101(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = torch.nn.Sequential(*modules)

    def embed(self, img: JpegImageFile):
        img = image_transform(img)
        return self.resnet(img.unsqueeze(0))


def embed_image(img: JpegImageFile):
    # Load the ResNet-101 model
    # TODO: Loading model should not be done inside function
    resnet = models.resnet101(pretrained=True)

    # Remove the last layer or two of linear layers coupled with softmax activation for classification
    modules = list(resnet.children())[:-1]
    resnet = torch.nn.Sequential(*modules)

    # Transform image:
    img = image_transform(img)

    # Pass the preprocessed image through the ResNet-101 model to obtain its image embedding
    embedding = resnet(img.unsqueeze(0))

    return embedding


def image_transform(img: JpegImageFile) -> torch.Tensor:
    transform = transforms.Compose(
        [
            # TODO: Verify if 256 is indeed the input size for resnet101
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(img)


def visualize_transform(img: torch.Tensor) -> None:
    plt.imshow(img.permute(1, 2, 0))


if __name__ == "__main__":

    base_directory = Path.cwd().parent.parent
    image_path = Path.joinpath(base_directory, "tests", "input.jpg")
    img = load_image_from_file(image_path)
    img_trans = image_transform(img)
    visualize_transform(img_trans)
    #    visualize_transform(img)
    print("yo")
