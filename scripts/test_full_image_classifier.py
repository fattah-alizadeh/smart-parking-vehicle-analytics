import sys
from pathlib import Path

import cv2
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.parking.parking_occupancy_pipeline import ParkingOccupancyPipeline


# Input: Full parking image path and corresponding camera CSV path.
# Output: Runs the complete parking-occupancy pipeline and displays/saves the result.
# Purpose: Test the reusable ParkingOccupancyPipeline on any selected parking image.
def run_pipeline(image_path, csv_path):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model_path = (
        PROJECT_ROOT
        / "results"
        / "models"
        / "mobilenet_v3_cameraB_to_cameraA_augmented_best.pth"
    )

    pipeline = ParkingOccupancyPipeline(
        model_path=model_path,
        device=device
    )

    _, results, output_image = pipeline.process_image(
        image_path=image_path,
        csv_path=csv_path
    )

    summary = pipeline.calculate_summary(
        results
    )

    print("\nPARKING OCCUPANCY SUMMARY")
    print("-------------------------")
    print("Total:", summary["total"])
    print("Free:", summary["free"])
    print("Busy:", summary["busy"])

    print("\nINDIVIDUAL PARKING SPACES")
    print("-------------------------")

    for result in results:
        print(
            f"Slot {result['slot_id']} -> "
            f"{result['label']} "
            f"({result['confidence']:.2%})"
        )

    output_folder = (
        PROJECT_ROOT
        / "results"
        / "parking"
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_folder
        / f"{Path(image_path).stem}_result.jpg"
    )

    cv2.imwrite(
        str(output_path),
        output_image
    )

    print(
        "\nResult saved to:",
        output_path
    )

    cv2.imshow(
        "Parking Occupancy Result",
        output_image
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Input: Command-line image path and matching camera CSV path.
# Output: Starts the parking pipeline for the selected image.
# Purpose: Allow arbitrary parking images to be tested without changing the source code.
def main():
    if len(sys.argv) != 3:
        print(
            "Usage:\n"
            "python scripts/test_full_image_classifier.py "
            "<image_path> <csv_path>"
        )
        return

    image_path = Path(
        sys.argv[1]
    )

    csv_path = Path(
        sys.argv[2]
    )

    run_pipeline(
        image_path=image_path,
        csv_path=csv_path
    )


if __name__ == "__main__":
    main()