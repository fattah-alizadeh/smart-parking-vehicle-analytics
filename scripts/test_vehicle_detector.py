import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.vehicle_detector import VehicleDetector
from src.detection.vehicle_visualizer import VehicleVisualizer


# Input: None.
# Output: Prints detected vehicle classes, confidences, and bounding boxes.
# Purpose: Test the pretrained vehicle detector on one full parking-lot image.
def main():
    image_path = (
        PROJECT_ROOT
        / "data"
        / "public"
        / "cnrpark"
        / "nrpark_full"
        / "CNR-EXT_FULL_IMAGE_1000x750"
        / "FULL_IMAGE_1000x750"
        / "SUNNY"
        / "2015-11-12"
        / "camera1"
        / "2015-11-12_1009.jpg"
    )

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise ValueError(
            f"Could not load image: {image_path}"
        )

    detector = VehicleDetector(
        model_name="yolo11n.pt",
        confidence_threshold=0.25,
        image_size=960
    )


    detections = detector.detect(
        image
    )

    print("Vehicles detected:", len(detections))
    print("----------------------")

    for detection in detections:
        print(
            f"{detection['class_name']} -> "
            f"{detection['confidence']:.2%} | "
            f"Box: "
            f"({detection['x1']}, {detection['y1']}) "
            f"to "
            f"({detection['x2']}, {detection['y2']})"
        )

    visualizer = VehicleVisualizer()

    output_image = visualizer.draw_detections(
        image=image,
        detections=detections
    )

    output_folder = (
        PROJECT_ROOT
        / "results"
        / "detection"
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_folder
        / f"{image_path.stem}_vehicles.jpg"
    )

    cv2.imwrite(
        str(output_path),
        output_image
    )

    cv2.imshow(
        "Vehicle Detection",
        output_image
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(
        "\nDetection image saved to:",
        output_path
    )

if __name__ == "__main__":
    main()