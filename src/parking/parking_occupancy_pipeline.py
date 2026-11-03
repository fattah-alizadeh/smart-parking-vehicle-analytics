import cv2

from src.parking.parking_space_classifier import ParkingSpaceClassifier
from src.parking.parking_space_loader import ParkingSpaceLoader
from src.parking.parking_visualizer import ParkingVisualizer


class ParkingOccupancyPipeline:

    # Input: Trained model path and computation device.
    # Output: Initialized complete parking-occupancy pipeline.
    # Purpose: Combine space loading, classification, and visualization in one reusable component.
    def __init__(self, model_path, device):
        self.space_loader = ParkingSpaceLoader()

        self.classifier = ParkingSpaceClassifier(
            model_path=model_path,
            device=device
        )

        self.visualizer = ParkingVisualizer()


    # Input: Full parking image path and corresponding camera CSV path.
    # Output: Original image, classification results, and visualized output image.
    # Purpose: Run the complete occupancy pipeline on one parking-lot image.
    def process_image(self, image_path, csv_path):
        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            raise ValueError(
                f"Could not load image: {image_path}"
            )

        parking_spaces = self.space_loader.load_spaces(
            csv_path
        )

        results = self.classifier.classify_spaces(
            image=image,
            parking_spaces=parking_spaces
        )

        output_image = self.visualizer.draw_results(
            image=image,
            results=results
        )

        return image, results, output_image


    # Input: Parking-space classification results.
    # Output: Dictionary containing total, free, and busy parking-space counts.
    # Purpose: Produce simple occupancy statistics for one parking-lot image.
    def calculate_summary(self, results):
        free_count = sum(
            1
            for result in results
            if result["label"] == "Free"
        )

        busy_count = sum(
            1
            for result in results
            if result["label"] == "Busy"
        )

        return {
            "total": len(results),
            "free": free_count,
            "busy": busy_count
        }