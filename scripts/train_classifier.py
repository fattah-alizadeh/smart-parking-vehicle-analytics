import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.dataset_splitter import DatasetSplitter
from src.classification.model_factory import ModelFactory
from src.classification.parking_dataset import ParkingSpaceDataset
from src.classification.trainer import Trainer
from src.classification.transforms import build_default_transform


# Input: None.
# Output: Trains and validates the parking classifier.
# Purpose: Connect dataset, model, optimizer, loss function, and trainer into one experiment.
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

    splitter = DatasetSplitter(
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42
    )

    train_dataset, val_dataset, test_dataset = splitter.split(dataset)

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

    model = ModelFactory.create_mobilenet_v3_small(
        num_classes=2
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

    best_val_accuracy = 0.0

    model_output_path = PROJECT_ROOT / "results" / "models"
    model_output_path.mkdir(parents=True, exist_ok=True)

    best_model_path = model_output_path / "mobilenet_v3_parking_best.pth"

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

            print(f"Best model saved to: {best_model_path}")


if __name__ == "__main__":
    main()