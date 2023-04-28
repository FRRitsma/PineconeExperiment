# %%
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


def embed_image():
    # Load the ResNet-101 model
    resnet = models.resnet101(pretrained=True)

    # Remove the last layer or two of linear layers coupled with softmax activation for classification
    modules = list(resnet.children())[:-1]
    resnet = torch.nn.Sequential(*modules)

    # Load the input image and preprocess it
    img = Image.open("input.jpg")
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img = transform(img)

    # Pass the preprocessed image through the ResNet-101 model to obtain its image embedding
    embedding = resnet(img.unsqueeze(0))

    # Save the image embedding to a file or database for later use
    torch.save(embedding, "embedding.pt")
