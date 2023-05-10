from src.extract.extract import extract_images
from src.extract.extract import SetPath


def test_train_images_extraction():
    n_labels: int = 5
    n_images: int = 7
    train_images = extract_images(n_labels, n_images, SetPath.train)
    assert len(train_images) == n_labels * n_images
    assert isinstance(train_images, list)
    first_element = train_images[0]
    assert isinstance(first_element, dict)


def test_val_images_extraction():
    n_labels: int = 5
    n_images: int = 7
    val_images = extract_images(n_labels, n_images, SetPath.val)
    assert len(val_images) == n_labels * n_images
    assert isinstance(val_images, list)
    first_element = val_images[0]
    assert isinstance(first_element, dict)
