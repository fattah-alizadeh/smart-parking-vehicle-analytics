import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.parking_dataset import ParkingSpaceDataset
from src.classification.transforms import build_default_transform


# Input: None.
# Output: Prints dataset size and one sample's properties.
# Purpose: Verify that the parking dataset loader and transform pipeline work correctly.
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

    image, label, camera, image_path = dataset[0]

    print("Dataset size:", len(dataset))
    print("Image tensor shape:", image.shape)
    print("Label:", label)
    print("Camera:", camera)
    print("Image path:", image_path)


if __name__ == "__main__":
    main()