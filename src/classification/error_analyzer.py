import random

import matplotlib.pyplot as plt
import torch


class ErrorAnalyzer:

    # Input: Trained model and computation device.
    # Output: ErrorAnalyzer instance.
    # Purpose: Store the model and device used for misclassification analysis.
    def __init__(self, model, device):
        self.model = model
        self.device = device


    # Input: DataLoader containing images and their true labels.
    # Output: List containing information about incorrectly classified images.
    # Purpose: Find model mistakes and store their true label, prediction, confidence, and path.
    def collect_errors(self, data_loader):
        self.model.eval()

        errors = []

        with torch.no_grad():
            for images, labels, cameras, image_paths in data_loader:
                images = images.to(self.device)

                outputs = self.model(images)

                probabilities = torch.softmax(
                    outputs,
                    dim=1
                )

                confidences, predictions = torch.max(
                    probabilities,
                    dim=1
                )

                predictions = predictions.cpu()
                confidences = confidences.cpu()

                for index in range(len(labels)):
                    if predictions[index] != labels[index]:
                        errors.append(
                            {
                                "image_path": image_paths[index],
                                "true_label": labels[index].item(),
                                "predicted_label": predictions[index].item(),
                                "confidence": confidences[index].item(),
                                "camera": cameras[index]
                            }
                        )

        return errors


    # Input: Error list, true label, predicted label, and number of samples.
    # Output: Random subset of matching errors.
    # Purpose: Select a specific type of model mistake for visual inspection.
    def sample_errors(
        self,
        errors,
        true_label,
        predicted_label,
        count=12
    ):
        matching_errors = [
            error
            for error in errors
            if error["true_label"] == true_label
            and error["predicted_label"] == predicted_label
        ]

        sample_count = min(
            count,
            len(matching_errors)
        )

        return random.sample(
            matching_errors,
            sample_count
        )


    # Input: Selected errors and mapping from class index to class name.
    # Output: Figure displaying misclassified parking-space images.
    # Purpose: Visually inspect why the classifier is making mistakes.
    def display_errors(self, errors, index_to_class):
        if not errors:
            print("No matching errors found.")
            return

        columns = 4
        rows = (len(errors) + columns - 1) // columns

        figure, axes = plt.subplots(
            rows,
            columns,
            figsize=(12, 3 * rows)
        )

        axes = axes.flatten()

        for axis in axes:
            axis.axis("off")

        for axis, error in zip(axes, errors):
            image = plt.imread(
                error["image_path"]
            )

            axis.imshow(image)

            true_name = index_to_class[
                error["true_label"]
            ]

            predicted_name = index_to_class[
                error["predicted_label"]
            ]

            axis.set_title(
                f"Actual: {true_name}\n"
                f"Predicted: {predicted_name}\n"
                f"Confidence: {error['confidence']:.2%}"
            )

            axis.axis("off")

        plt.tight_layout()
        plt.show()