import csv
from pathlib import Path


class VehicleAnalyticsExporter:

    # Input: List of vehicle analytics records and output CSV path.
    # Output: A CSV file containing the vehicle analytics records.
    # Purpose: Export per-vehicle movement analytics for reporting and visualization.
    def export_to_csv(self, records, output_path):

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        fieldnames = [
            "track_id",
            "class_name",
            "first_frame",
            "last_frame",
            "trajectory_points",
            "duration_seconds",
            "distance_pixels",
            "crossing_direction"
        ]

        with open(
            output_path,
            mode="w",
            newline="",
            encoding="utf-8"
        ) as csv_file:

            writer = csv.DictWriter(
                csv_file,
                fieldnames=fieldnames
            )

            writer.writeheader()

            for record in records:
                writer.writerow(record)

        return output_path