import sys
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.dataset_splitter import DatasetSplitter
from src.classification.model_factory import ModelFactory
from src.classification.parking_dataset import ParkingSpaceDataset
from src.classification.trainer import Trainer
from src.classification.transforms import (
    build_train_transform,
    build_eval_transform
)


# Input: Trained model, DataLoader, and computation device.
# Output: True labels and predicted labels.
# Purpose: Collect model predictions for final cross-camera evaluation.
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

            true_labels.extend(
                labels.tolist()
            )

            predicted_labels.extend(
                predictions.cpu().tolist()
            )

    return true_labels, predicted_labels


# Input: None.
# Output: Trains on Camera A with augmentation and evaluates on unseen Camera B.
# Purpose: Test whether data augmentation improves cross-camera generalization.
def main():

    train_camera = "B"
    test_camera = "A"

    model_name = "resnet18"

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # Training images receive augmentation.
    train_transform = build_train_transform()

    # Validation and testing images do not receive augmentation.
    eval_transform = build_eval_transform()

    # Input: Selected training camera name.
    # Output: Training dataset with augmentation.
    # Purpose: Load the source camera used for model training.
    train_camera_dataset = ParkingSpaceDataset(
        dataset_root=PROJECT_ROOT / "data/public/cnrpark",
        cameras=[train_camera],
        class_to_index={
            "Free": 0,
            "Busy": 1
        },
        transform=train_transform
    )

    # Input: Selected training camera name.
    # Output: Evaluation version of the same training-camera dataset.
    # Purpose: Create validation data without augmentation.
    train_camera_eval_dataset = ParkingSpaceDataset(
        dataset_root=PROJECT_ROOT / "data/public/cnrpark",
        cameras=[train_camera],
        class_to_index={
            "Free": 0,
            "Busy": 1
        },
        transform=eval_transform
    )

    # Input: Selected unseen test camera name.
    # Output: Dataset used only for final cross-camera testing.
    # Purpose: Measure generalization to a completely unseen camera.
    test_camera_dataset = ParkingSpaceDataset(
        dataset_root=PROJECT_ROOT / "data/public/cnrpark",
        cameras=[test_camera],
        class_to_index={
            "Free": 0,
            "Busy": 1
        },
        transform=eval_transform
    )

    splitter = DatasetSplitter(
        train_ratio=0.85,
        val_ratio=0.15,
        test_ratio=0.0,
        random_seed=42
    )

    # Split train Camera dataset using the clean evaluation dataset.
    train_index_subset, val_dataset, _ = splitter.split(
        train_camera_eval_dataset
    )

    train_dataset = Subset(
        train_camera_dataset,
        train_index_subset.indices
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_camera_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    print("Model:", model_name)
    print("Training camera:", train_camera)
    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_dataset))
    print("Test camera:", test_camera)
    print("Test images:", len(test_camera_dataset))

    if model_name == "mobilenet_v3_small":
        model = ModelFactory.create_mobilenet_v3_small(
            num_classes=2
        )

    elif model_name == "resnet18":
        model = ModelFactory.create_resnet18(
            num_classes=2
        )

    else:
        raise ValueError(
            f"Unsupported model: {model_name}"
        )

    model = model.to(device)

    loss_function = nn.CrossEntropyLoss()

    optimizer = Adam(
        model.parameters(),
        lr=0.0001
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        device=device
    )

    model_output_folder = (
        PROJECT_ROOT
        / "results"
        / "models"
    )

    model_output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

   
    best_model_path = (
       model_output_folder
        / f"{model_name}_camera{train_camera}_to_camera{test_camera}_augmented_best.pth"
    )

    best_val_accuracy = 0.0

    epochs = 5

    for epoch in range(epochs):

        train_loss, train_accuracy = trainer.train_one_epoch(
            train_loader
        )

        val_loss, val_accuracy = trainer.evaluate(
            val_loader
        )

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            torch.save(
                model.state_dict(),
                best_model_path
            )

            print(
                "Best model saved:",
                best_model_path
            )

    # Reload the best Camera A validation model.
    model.load_state_dict(
        torch.load(
            best_model_path,
            map_location=device
        )
    )

    model = model.to(device)

    # Test only now on completely unseen Camera B.
    true_labels, predicted_labels = collect_predictions(
        model,
        test_loader,
        device
    )

    accuracy = accuracy_score(
        true_labels,
        predicted_labels
    )

    print("\nCROSS-CAMERA TEST RESULTS")
    print("-------------------------")
    print("Training camera:", train_camera)
    print("Unseen test camera:", test_camera)
    print("Training augmentation: YES")
    print("Test Accuracy:", round(accuracy, 4))

    print("\nClassification Report:")

    print(
        classification_report(
            true_labels,
            predicted_labels,
            target_names=[
                "Free",
                "Busy"
            ]
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