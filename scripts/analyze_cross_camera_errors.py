import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.error_analyzer import ErrorAnalyzer
from src.classification.model_factory import ModelFactory
from src.classification.parking_dataset import ParkingSpaceDataset
from src.classification.transforms import build_default_transform


# Input: None.
# Output: Displays examples of Busy parking spaces incorrectly predicted as Free.
# Purpose: Visually analyze the main failure type in the Camera A to Camera B experiment.
def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    camera_b_dataset = ParkingSpaceDataset(
        dataset_root=PROJECT_ROOT / "data/public/cnrpark",
        cameras=["B"],
        class_to_index={
            "Free": 0,
            "Busy": 1
        },
        transform=build_default_transform()
    )

    test_loader = DataLoader(
        camera_b_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    model = ModelFactory.create_mobilenet_v3_small(
        num_classes=2
    )

    model_path = (
        PROJECT_ROOT
        / "results"
        / "models"
        / "mobilenet_v3_cameraA_to_cameraB_best.pth"
    )

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )

    model = model.to(device)

    analyzer = ErrorAnalyzer(
        model=model,
        device=device
    )

    errors = analyzer.collect_errors(
        test_loader
    )

    print("Total errors:", len(errors))

    # Busy = 1 and Free = 0.
    busy_as_free = [
        error
        for error in errors
        if error["true_label"] == 1
        and error["predicted_label"] == 0
    ]

    print(
        "Busy predicted as Free:",
        len(busy_as_free)
    )

    selected_errors = analyzer.sample_errors(
        errors=errors,
        true_label=1,
        predicted_label=0,
        count=12
    )

    analyzer.display_errors(
        selected_errors,
        index_to_class={
            0: "Free",
            1: "Busy"
        }
    )


if __name__ == "__main__":
    main()