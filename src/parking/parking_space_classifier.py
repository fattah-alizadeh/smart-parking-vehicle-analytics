from pathlib import Path

import cv2
import torch
from PIL import Image

from src.classification.model_factory import ModelFactory
from src.classification.transforms import build_eval_transform


class ParkingSpaceClassifier:

    # Input: Saved model path and computation device.
    # Output: Initialized classifier ready for parking-space inference.
    # Purpose: Load the trained MobileNetV3 model once and reuse it for many parking spaces.
    def __init__(self, model_path, device):
        self.device = device
        self.transform = build_eval_transform()

        self.model = ModelFactory.create_mobilenet_v3_small(
            num_classes=2
        )

        self.model.load_state_dict(
            torch.load(
                Path(model_path),
                map_location=device
            )
        )

        self.model = self.model.to(device)
        self.model.eval()


    # Input: Full OpenCV image and one parking-space definition.
    # Output: Cropped parking-space image.
    # Purpose: Extract one parking-space region from the full parking-lot image.
    def _crop_space(self, image, parking_space):
        x = parking_space["x"]
        y = parking_space["y"]
        width = parking_space["width"]
        height = parking_space["height"]

        return image[
            y:y + height,
            x:x + width
        ]


    # Input: OpenCV parking-space crop.
    # Output: PyTorch tensor prepared for MobileNetV3.
    # Purpose: Convert an OpenCV BGR image into the model's expected input format.
    def _prepare_crop(self, crop):
        rgb_crop = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2RGB
        )

        pil_image = Image.fromarray(
            rgb_crop
        )

        return self.transform(
            pil_image
        )


    # Input: Full parking-lot image and parking-space definitions.
    # Output: List of predictions containing slot ID, class, and confidence.
    # Purpose: Classify all parking spaces together using one GPU batch.
    def classify_spaces(self, image, parking_spaces):
        tensors = []
        valid_spaces = []

        for parking_space in parking_spaces:
            crop = self._crop_space(
                image,
                parking_space
            )

            if crop.size == 0:
                continue

            tensor = self._prepare_crop(
                crop
            )

            tensors.append(tensor)
            valid_spaces.append(parking_space)

        if not tensors:
            return []

        batch = torch.stack(
            tensors
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(
                batch
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidences, predictions = torch.max(
                probabilities,
                dim=1
            )

        results = []

        for parking_space, prediction, confidence in zip(
            valid_spaces,
            predictions.cpu(),
            confidences.cpu()
        ):
            label = (
                "Free"
                if prediction.item() == 0
                else "Busy"
            )

            results.append(
                {
                    "slot_id": parking_space["slot_id"],
                    "x": parking_space["x"],
                    "y": parking_space["y"],
                    "width": parking_space["width"],
                    "height": parking_space["height"],
                    "label": label,
                    "confidence": confidence.item()
                }
            )

        return results