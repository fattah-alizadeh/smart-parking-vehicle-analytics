
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.tracking.line_crossing_counter import LineCrossingCounter


class TestLineCrossingCounter(unittest.TestCase):

    # Input: None.
    # Output: A counter with a vertical line at x=100.
    # Purpose: Prepare an independent counter for every test.
    def setUp(self):
        self.counter = LineCrossingCounter(
            start_point=(100, 0),
            end_point=(100, 200)
        )

    # Input: Tracking ID and bounding-box center coordinates.
    # Output: Vehicle dictionary compatible with LineCrossingCounter.
    # Purpose: Generate synthetic tracked vehicles for testing.
    def make_vehicle(self, track_id, x, y=100):
        return {
            "track_id": track_id,
            "x1": x - 5,
            "y1": y - 5,
            "x2": x + 5,
            "y2": y + 5
        }

    # Input: A vehicle moving from left to right across the line.
    # Output: Exactly one crossing.
    # Purpose: Verify crossing detection in the first direction.
    def test_left_to_right_crossing(self):
        self.counter.update([self.make_vehicle(1, 80)])
        self.counter.update([self.make_vehicle(1, 120)])

        self.assertEqual(self.counter.get_total_count(), 1)

    # Input: A vehicle moving from right to left across the line.
    # Output: Exactly one crossing.
    # Purpose: Verify crossing detection in the opposite direction.
    def test_right_to_left_crossing(self):
        self.counter.update([self.make_vehicle(2, 120)])
        self.counter.update([self.make_vehicle(2, 80)])

        self.assertEqual(self.counter.get_total_count(), 1)

    # Input: A vehicle staying on the same side of the line.
    # Output: Zero crossings.
    # Purpose: Prevent counting vehicles that never cross the line.
    def test_no_crossing(self):
        self.counter.update([self.make_vehicle(3, 50)])
        self.counter.update([self.make_vehicle(3, 70)])
        self.counter.update([self.make_vehicle(3, 90)])

        self.assertEqual(self.counter.get_total_count(), 0)

    # Input: One vehicle crossing the line multiple times.
    # Output: Exactly one crossing.
    # Purpose: Verify deduplication by tracking ID.
    def test_same_vehicle_counted_once(self):
        self.counter.update([self.make_vehicle(4, 80)])
        self.counter.update([self.make_vehicle(4, 120)])
        self.counter.update([self.make_vehicle(4, 80)])
        self.counter.update([self.make_vehicle(4, 120)])

        self.assertEqual(self.counter.get_total_count(), 1)

    # Input: Two vehicles crossing in opposite directions.
    # Output: Two total crossings, one in each direction.
    # Purpose: Verify directional counts and total consistency.
    def test_directional_counts(self):
        self.counter.update([
            self.make_vehicle(5, 80),
            self.make_vehicle(6, 120)
        ])

        self.counter.update([
            self.make_vehicle(5, 120),
            self.make_vehicle(6, 80)
        ])

        self.assertEqual(self.counter.get_total_count(), 2)
        self.assertEqual(self.counter.get_a_to_b_count(), 1)
        self.assertEqual(self.counter.get_b_to_a_count(), 1)

    # Input: No tracked vehicles.
    # Output: Zero crossings.
    # Purpose: Verify safe handling of empty frames.
    def test_empty_frame(self):
        self.counter.update([])

        self.assertEqual(self.counter.get_total_count(), 0)


if __name__ == "__main__":
    unittest.main()
