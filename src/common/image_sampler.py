import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt


class ImageSampler:

    # Input: Root directory of the dataset.
    # Output: None.
    # Purpose: Store the dataset location for reusable image sampling operations.
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)

    # Input: Camera name, class name, and number of requested images.
    # Output: A list of randomly selected image paths.
    # Purpose: Select representative random images from a dataset category.
    def sample_images(self, camera, class_name, count=4):
        folder = self.dataset_path / camera / class_name
        images = list(folder.glob("*.jpg"))

        if len(images) < count:
            raise ValueError(
                f"Requested {count} images, but only {len(images)} are available."
            )

        return random.sample(images, count)

    # Input: List of image paths and a display title.
    # Output: A displayed figure containing the selected images.
    # Purpose: Visually inspect dataset images before model development.
    def display_images(self, image_paths, title):
        figure, axes = plt.subplots(1, len(image_paths))

        if len(image_paths) == 1:
            axes = [axes]

        for axis, image_path in zip(axes, image_paths):
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            axis.imshow(image)
            axis.set_title(image_path.name)
            axis.axis("off")

        figure.suptitle(title)
        plt.tight_layout()
        plt.show()