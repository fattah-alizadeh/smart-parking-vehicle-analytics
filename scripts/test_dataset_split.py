import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.dataset_splitter import DatasetSplitter
from src.classification.parking_dataset import ParkingSpaceDataset
from src.classification.transforms import build_default_transform


# Input: A PyTorch Subset created from ParkingSpaceDataset.
# Output: Counts of Free/Busy samples and Camera A/B samples.
# Purpose: Inspect the composition of one dataset split.
def summarize_subset(subset):
    class_counts = Counter()
    camera_counts = Counter()

    for index in subset.indices:
        image_path, label, camera, class_name = subset.dataset.samples[index]

        class_counts[class_name] += 1
        camera_counts[camera] += 1

    return class_counts, camera_counts


# Input: Split name and corresponding PyTorch Subset.
# Output: Printed split statistics.
# Purpose: Display the size, class balance, and camera balance of a dataset split.
def print_split_summary(name, subset):
    class_counts, camera_counts = summarize_subset(subset)

    print(f"\n{name}")
    print("-" * len(name))
    print("Total:", len(subset))
    print("Classes:", dict(class_counts))
    print("Cameras:", dict(camera_counts))


# Input: None.
# Output: Train, validation, and test split summaries.
# Purpose: Verify that the grouped dataset splitter works correctly.
def main():
    dataset = ParkingSpaceDataset(
        dataset_root=PROJECT_ROOT / "data/public/cnrpark",
        cameras=["A", "B"],
        class_to_index={
            "Free": 0,
            "Busy": 1
        },
        transform=build_default_transform()
    )

    splitter = DatasetSplitter(
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42
    )

    train_dataset, val_dataset, test_dataset = splitter.split(dataset)

    print_split_summary("TRAIN", train_dataset)
    print_split_summary("VALIDATION", val_dataset)
    print_split_summary("TEST", test_dataset)


if __name__ == "__main__":
    main()