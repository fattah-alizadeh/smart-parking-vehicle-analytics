import csv
from pathlib import Path


class ParkingSpaceLoader:

    # Input: Original image size and resized image size.
    # Output: Initialized ParkingSpaceLoader object.
    # Purpose: Store image dimensions needed to scale parking-space coordinates.
    def __init__(
        self,
        original_width=2592,
        original_height=1944,
        target_width=1000,
        target_height=750
    ):
        self.original_width = original_width
        self.original_height = original_height
        self.target_width = target_width
        self.target_height = target_height


    # Input: Path to a camera CSV file.
    # Output: List of parking-space dictionaries with scaled coordinates.
    # Purpose: Load parking-space definitions and scale them to the target image size.
    def load_spaces(self, csv_path):
        csv_path = Path(csv_path)

        scale_x = self.target_width / self.original_width
        scale_y = self.target_height / self.original_height

        parking_spaces = []

        with csv_path.open(
            mode="r",
            newline="",
            encoding="utf-8-sig"
        ) as csv_file:

            reader = csv.DictReader(csv_file)

            for row in reader:
                parking_space = {
                    "slot_id": int(row["SlotId"]),
                    "x": round(float(row["X"]) * scale_x),
                    "y": round(float(row["Y"]) * scale_y),
                    "width": round(float(row["W"]) * scale_x),
                    "height": round(float(row["H"]) * scale_y)
                }

                parking_spaces.append(parking_space)

        return parking_spaces