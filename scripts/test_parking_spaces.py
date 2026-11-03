
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.parking.parking_space_loader import ParkingSpaceLoader


# Input: Full parking-lot image and scaled parking-space definitions.
# Output: Image with parking-space rectangles drawn on it.
# Purpose: Visually verify that the CSV coordinates align with real parking spaces.
def draw_parking_spaces(image, parking_spaces):
    output = image.copy()

    for space in parking_spaces:
        x = space["x"]
        y = space["y"]
        width = space["width"]
        height = space["height"]

        cv2.rectangle(
            output,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            2
        )

        cv2.putText(
            output,
            str(space["slot_id"]),
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1
        )

    return output


# Input: None.
# Output: Displays one full Camera 1 image with all parking-space boxes.
# Purpose: Test parking-space coordinate scaling and alignment.
def main():
    dataset_root = (
        PROJECT_ROOT
        / "data"
        / "public"
        / "cnrpark"
        / "nrpark_full"
        / "CNR-EXT_FULL_IMAGE_1000x750"
    )

    csv_path = dataset_root / "camera1.csv"

    image_path = (
        dataset_root
        / "FULL_IMAGE_1000x750"
        / "SUNNY"
        / "2015-11-12"
        / "camera1"
        / "2015-11-12_1009.jpg"
    )

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"Could not load image: {image_path}"
        )

    loader = ParkingSpaceLoader()

    parking_spaces = loader.load_spaces(
        csv_path
    )

    print(
        "Parking spaces loaded:",
        len(parking_spaces)
    )

    output = draw_parking_spaces(
        image,
        parking_spaces
    )

    cv2.imshow(
        "Camera 1 Parking Spaces",
        output
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()