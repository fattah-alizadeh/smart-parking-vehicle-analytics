import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.model_factory import ModelFactory
from src.classification.parking_dataset import ParkingSpaceDataset
from src.classification.transforms import build_default_transform


# Input: None.
# Output: Prints GPU/device information and model output shape for one batch.
# Purpose: Verify that the classifier runs correctly on the available GPU.
def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    dataset = ParkingSpaceDataset(
        dataset_root=PROJECT_ROOT / "data/public/cnrpark",
        cameras=["A", "B"],
        class_to_index={
            "Free": 0,
            "Busy": 1
        },
        transform=build_default_transform()
    )

    data_loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True
    )

    model = ModelFactory.create_mobilenet_v3_small(
        num_classes=2
    )

    model = model.to(device)
    model.eval()

    images, labels, cameras, image_paths = next(iter(data_loader))

    images = images.to(device)

    with torch.no_grad():
        outputs = model(images)

    print("Input batch shape:", images.shape)
    print("Output shape:", outputs.shape)
    print("Labels:", labels.tolist())


if __name__ == "__main__":
    main()