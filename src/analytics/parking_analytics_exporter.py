
import csv
from pathlib import Path


class ParkingAnalyticsExporter:

    # Input: Enriched parking occupancy summaries and output CSV path.
    # Output: CSV file containing occupancy measurements over time.
    # Purpose: Export parking occupancy data for interactive dashboard charts.
    def export_occupancy(self, summaries, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "image_name",
            "timestamp",
            "total",
            "free",
            "busy",
            "occupancy_percentage"
        ]

        with output_path.open(
            mode="w",
            newline="",
            encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

            for summary in summaries:
                timestamp = summary["timestamp"]

                writer.writerow({
                    "image_name": summary["image_name"],
                    "timestamp": timestamp.isoformat(),
                    "total": summary["total"],
                    "free": summary["free"],
                    "busy": summary["busy"],
                    "occupancy_percentage": summary["occupancy_percentage"]
                })

        return output_path

    # Input: Per-slot utilization analytics and output CSV path.
    # Output: CSV file containing parking-space utilization statistics.
    # Purpose: Export slot-level analytics for dashboard charts and tables.
    def export_slot_utilization(self, slot_analytics, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "slot_id",
            "busy_percentage",
            "state_changes",
            "estimated_busy_minutes"
        ]

        with output_path.open(
            mode="w",
            newline="",
            encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

            for item in slot_analytics:
                writer.writerow({
                    key: item[key]
                    for key in fieldnames
                })

        return output_path
