import sys
from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.dataset_splitter import DatasetSplitter
from src.classification.model_factory import ModelFactory
from src.classification.parking_dataset import ParkingSpaceDataset
from src.classification.transforms import build_default_transform


# Input: Trained model, test DataLoader, and computation device.
# Output: True labels and predicted labels.
# Purpose: Run inference on the held-out test dataset.
def collect_predictions(model, data_loader, device):
    model.eval()

    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for images, labels, cameras, image_paths in data_loader:
            images = images.to(device)

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            true_labels.extend(labels.tolist())
            predicted_labels.extend(predictions.cpu().tolist())

    return true_labels, predicted_labels


# Input: None.
# Output: Prints test accuracy, precision, recall, F1-score, and confusion matrix.
# Purpose: Evaluate the saved parking classifier on unseen test data.
def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

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

    test_loader = DataLoader(
        test_dataset,
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
        / "mobilenet_v3_parking_best.pth"
    )

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )

    model = model.to(device)

    true_labels, predicted_labels = collect_predictions(
        model,
        test_loader,
        device
    )

    accuracy = accuracy_score(
        true_labels,
        predicted_labels
    )

    print("Test Accuracy:", round(accuracy, 4))

    print("\nClassification Report:")
    print(
        classification_report(
            true_labels,
            predicted_labels,
            target_names=["Free", "Busy"]
        )
    )

    print("Confusion Matrix:")
    print(
        confusion_matrix(
            true_labels,
            predicted_labels
        )
    )


if __name__ == "__main__":
    main()