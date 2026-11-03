
import csv
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics.vehicle_analytics_exporter import VehicleAnalyticsExporter
from src.analytics.parking_analytics_exporter import ParkingAnalyticsExporter


class TestAnalyticsExporters(unittest.TestCase):

    # Input: None.
    # Output: Temporary directory and initialized exporters.
    # Purpose: Keep tests independent of actual project analytics files.
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)

        self.vehicle_exporter = VehicleAnalyticsExporter()
        self.parking_exporter = ParkingAnalyticsExporter()

    # Input: None.
    # Output: Removed temporary test files.
    # Purpose: Clean up automatically after each test.
    def tearDown(self):
        self.temp_dir.cleanup()

    # Input: Path to a CSV file.
    # Output: List of CSV records.
    # Purpose: Read exported files for validation.
    def read_csv(self, path):
        with path.open(newline="", encoding="utf-8") as file:
            return list(csv.DictReader(file))

    # Input: One vehicle analytics record.
    # Output: CSV containing the correct vehicle data.
    # Purpose: Verify vehicle analytics export.
    def test_vehicle_export(self):
        records = [
            {
                "track_id": 12,
                "class_name": "car",
                "first_frame": 10,
                "last_frame": 40,
                "trajectory_points": 31,
                "duration_seconds": 3.0,
                "distance_pixels": 150.5,
                "crossing_direction": "A_TO_B"
            }
        ]

        output_path = self.output_dir / "vehicles.csv"

        result = self.vehicle_exporter.export_to_csv(
            records, output_path
        )

        self.assertEqual(result, output_path)
        self.assertTrue(output_path.exists())

        rows = self.read_csv(output_path)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["track_id"], "12")
        self.assertEqual(rows[0]["class_name"], "car")
        self.assertEqual(rows[0]["crossing_direction"], "A_TO_B")
        self.assertAlmostEqual(float(rows[0]["distance_pixels"]), 150.5)

    # Input: Empty vehicle records.
    # Output: CSV containing only the header.
    # Purpose: Verify safe export of empty vehicle datasets.
    def test_empty_vehicle_export(self):
        output_path = self.output_dir / "empty_vehicles.csv"

        self.vehicle_exporter.export_to_csv([], output_path)

        self.assertTrue(output_path.exists())
        self.assertEqual(len(self.read_csv(output_path)), 0)

    # Input: Parking occupancy summary with a datetime timestamp.
    # Output: CSV containing occupancy measurements.
    # Purpose: Verify occupancy export and timestamp serialization.
    def test_parking_occupancy_export(self):
        summaries = [
            {
                "image_name": "2015-11-12_0809.jpg",
                "timestamp": datetime(2015, 11, 12, 8, 9),
                "total": 35,
                "free": 10,
                "busy": 25,
                "occupancy_percentage": 71.4286
            }
        ]

        output_path = self.output_dir / "parking.csv"

        result = self.parking_exporter.export_occupancy(
            summaries, output_path
        )

        self.assertEqual(result, output_path)
        self.assertTrue(output_path.exists())

        rows = self.read_csv(output_path)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["total"], "35")
        self.assertEqual(rows[0]["busy"], "25")
        self.assertEqual(rows[0]["free"], "10")
        self.assertEqual(
            rows[0]["timestamp"],
            "2015-11-12T08:09:00"
        )

        self.assertAlmostEqual(
            float(rows[0]["occupancy_percentage"]),
            71.4286
        )

    # Input: Parking-slot utilization record.
    # Output: CSV containing slot statistics.
    # Purpose: Verify individual parking-space analytics export.
    def test_slot_utilization_export(self):
        records = [
            {
                "slot_id": 7,
                "busy_percentage": 80.0,
                "state_changes": 3,
                "estimated_busy_minutes": 240.0
            }
        ]

        output_path = self.output_dir / "slots.csv"

        self.parking_exporter.export_slot_utilization(
            records, output_path
        )

        rows = self.read_csv(output_path)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["slot_id"], "7")
        self.assertEqual(rows[0]["state_changes"], "3")
        self.assertAlmostEqual(
            float(rows[0]["busy_percentage"]),
            80.0
        )

    # Input: Empty parking occupancy and slot utilization lists.
    # Output: Two valid CSV files containing only headers.
    # Purpose: Verify empty parking analytics exports.
    def test_empty_parking_exports(self):
        occupancy_path = self.output_dir / "empty_parking.csv"
        slot_path = self.output_dir / "empty_slots.csv"

        self.parking_exporter.export_occupancy(
            [], occupancy_path
        )

        self.parking_exporter.export_slot_utilization(
            [], slot_path
        )

        self.assertTrue(occupancy_path.exists())
        self.assertTrue(slot_path.exists())

        self.assertEqual(len(self.read_csv(occupancy_path)), 0)
        self.assertEqual(len(self.read_csv(slot_path)), 0)

    # Input: Vehicle export path inside a nonexistent nested folder.
    # Output: Automatically created folders and CSV file.
    # Purpose: Verify that the vehicle exporter creates output directories.
    def test_vehicle_nested_output_directory(self):
        output_path = (
            self.output_dir / "nested" / "analytics" / "vehicles.csv"
        )

        self.vehicle_exporter.export_to_csv([], output_path)

        self.assertTrue(output_path.exists())

    # Input: Parking export path inside a nonexistent nested folder.
    # Output: Automatically created folders and CSV file.
    # Purpose: Verify that the parking exporter creates output directories.
    def test_parking_nested_output_directory(self):
        output_path = (
            self.output_dir / "nested" / "analytics" / "parking.csv"
        )

        self.parking_exporter.export_occupancy([], output_path)

        self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
