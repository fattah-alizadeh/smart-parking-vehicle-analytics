
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics.vehicle_movement_analyzer import VehicleMovementAnalyzer


class TestVehicleMovementAnalyzer(unittest.TestCase):

    # Input: None.
    # Output: Initialized analyzer for each test.
    # Purpose: Prepare an independent analyzer instance.
    def setUp(self):
        self.analyzer = VehicleMovementAnalyzer()

    # Input: Three trajectory points forming a 3-4-5 triangle.
    # Output: Total distance equal to 10 pixels.
    # Purpose: Verify Euclidean trajectory distance calculation.
    def test_trajectory_distance(self):
        history = [
            {"frame": 0, "x": 0, "y": 0},
            {"frame": 1, "x": 3, "y": 4},
            {"frame": 2, "x": 6, "y": 8}
        ]

        distance = self.analyzer.calculate_trajectory_distance(history)

        self.assertAlmostEqual(distance, 10.0)

    # Input: A trajectory spanning 30 frames at 10 FPS.
    # Output: Duration equal to 3 seconds.
    # Purpose: Verify conversion from frame differences to seconds.
    def test_duration_seconds(self):
        history = [
            {"frame": 10, "x": 0, "y": 0},
            {"frame": 40, "x": 10, "y": 10}
        ]

        duration = self.analyzer.calculate_duration_seconds(
            history,
            fps=10
        )

        self.assertAlmostEqual(duration, 3.0)

    # Input: A trajectory containing only one observation.
    # Output: Zero duration and zero distance.
    # Purpose: Verify handling of insufficient trajectory data.
    def test_single_point_history(self):
        history = [
            {"frame": 5, "x": 10, "y": 20}
        ]

        self.assertEqual(
            self.analyzer.calculate_trajectory_distance(history),
            0.0
        )

        self.assertEqual(
            self.analyzer.calculate_duration_seconds(history, fps=10),
            0.0
        )

    # Input: An empty trajectory.
    # Output: Zero distance and zero duration.
    # Purpose: Verify that empty histories are handled safely.
    def test_empty_history(self):
        history = []

        self.assertEqual(
            self.analyzer.calculate_trajectory_distance(history),
            0.0
        )

        self.assertEqual(
            self.analyzer.calculate_duration_seconds(history, fps=10),
            0.0
        )

    # Input: A trajectory and invalid FPS value.
    # Output: Zero duration.
    # Purpose: Prevent division by zero for invalid frame rates.
    def test_invalid_fps(self):
        history = [
            {"frame": 0, "x": 0, "y": 0},
            {"frame": 10, "x": 5, "y": 5}
        ]

        self.assertEqual(
            self.analyzer.calculate_duration_seconds(history, fps=0),
            0.0
        )


if __name__ == "__main__":
    unittest.main()
