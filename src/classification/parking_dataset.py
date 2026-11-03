from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class ParkingSpaceDataset(Dataset):

    # Input: Dataset root path, camera names, class-to-index mapping, and image transform.
    # Output: A reusable PyTorch dataset object.
    # Purpose: Load CNRPark parking-space images for free/busy classification.
    def __init__(
        self,
        dataset_root,
        cameras,
        class_to_index,
        transform=None
    ):
        self.dataset_root = Path(dataset_root)
        self.cameras = cameras
        self.class_to_index = class_to_index
        self.transform = transform

        self.samples = self._load_samples()

    # Input: None.
    # Output: List of tuples containing image path, class index, camera, and class name.
    # Purpose: Build the dataset index from the selected CNRPark cameras and classes.
    def _load_samples(self):
        samples = []

        for camera in self.cameras:
            for class_name, class_index in self.class_to_index.items():
                folder = self.dataset_root / camera / class_name

                for image_path in folder.glob("*.jpg"):
                    samples.append(
                        (
                            image_path,
                            class_index,
                            camera,
                            class_name
                        )
                    )

        return samples

    # Input: None.
    # Output: Total number of indexed samples.
    # Purpose: Allow PyTorch to determine the dataset size.
    def __len__(self):
        return len(self.samples)

    # Input: Sample index.
    # Output: Image tensor, class label, camera name, and image path.
    # Purpose: Load one parking-space image and its associated metadata.
    def __getitem__(self, index):
        image_path, label, camera, class_name = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label, camera, str(image_path)