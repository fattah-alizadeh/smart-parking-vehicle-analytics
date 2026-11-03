import random
from collections import defaultdict

from torch.utils.data import Subset


class DatasetSplitter:

    # Input: Training, validation, and test ratios plus a fixed random seed.
    # Output: A reusable dataset splitter configuration.
    # Purpose: Store the rules used to create reproducible dataset splits.
    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42
    ):
        total_ratio = train_ratio + val_ratio + test_ratio

        if abs(total_ratio - 1.0) > 0.0001:
            raise ValueError("Train, validation, and test ratios must add up to 1.")

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed


    # Input: Image path and camera name.
    # Output: A string identifying the image's capture group.
    # Purpose: Keep images captured by the same camera at the same time in one split.
    def _get_group_key(self, image_path, camera):
        stem = image_path.stem

        parts = stem.split("_")

        if len(parts) >= 3:
            capture_time = "_".join(parts[:-1])
        else:
            capture_time = stem

        return f"{camera}_{capture_time}"


    # Input: ParkingSpaceDataset containing image paths and camera metadata.
    # Output: Three PyTorch Subset objects for training, validation, and testing.
    # Purpose: Split the dataset while reducing leakage between similar images.
    def split(self, dataset):
        groups = defaultdict(list)

        for index, sample in enumerate(dataset.samples):
            image_path, label, camera, class_name = sample

            group_key = self._get_group_key(
                image_path,
                camera
            )

            groups[group_key].append(index)

        group_keys = list(groups.keys())

        random_generator = random.Random(self.random_seed)
        random_generator.shuffle(group_keys)

        total_groups = len(group_keys)

        train_end = int(total_groups * self.train_ratio)
        val_end = train_end + int(total_groups * self.val_ratio)

        train_groups = group_keys[:train_end]
        val_groups = group_keys[train_end:val_end]
        test_groups = group_keys[val_end:]

        train_indices = self._collect_indices(groups, train_groups)
        val_indices = self._collect_indices(groups, val_groups)
        test_indices = self._collect_indices(groups, test_groups)

        return (
            Subset(dataset, train_indices),
            Subset(dataset, val_indices),
            Subset(dataset, test_indices)
        )


    # Input: Dictionary of grouped sample indices and selected group keys.
    # Output: One flat list of sample indices.
    # Purpose: Convert selected capture groups into indices usable by PyTorch Subset.
    def _collect_indices(self, groups, selected_groups):
        indices = []

        for group_key in selected_groups:
            indices.extend(groups[group_key])

        return indices