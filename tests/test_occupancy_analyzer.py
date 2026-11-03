
import sys
import unittest
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics.occupancy_analyzer import OccupancyAnalyzer


class TestOccupancyAnalyzer(unittest.TestCase):

    # Input: None.
    # Output: Initialized analyzer and sample parking records.
    # Purpose: Prepare consistent test data for each test.
    def setUp(self):
        self.analyzer = OccupancyAnalyzer()

        self.summaries = [
            {
                "image_name": "2015-11-12_0809.jpg",
                "total": 35,
                "busy": 10,
                "free": 25
            },
            {
                "image_name": "2015-11-12_0839.jpg",
                "total": 35,
                "busy": 20,
                "free": 15
            },
            {
                "image_name": "2015-11-12_0909.jpg",
                "total": 35,
                "busy": 30,
                "free": 5
            }
        ]

    # Input: Three parking occupancy summaries.
    # Output: Three enriched summaries with timestamps and percentages.
    # Purpose: Verify enrichment preserves the number of records.
    def test_enrich_summaries(self):
        enriched = self.analyzer.enrich_summaries(self.summaries)

        self.assertEqual(len(enriched), 3)
        self.assertEqual(
            enriched[0]["timestamp"],
            datetime(2015, 11, 12, 8, 9)
        )
        self.assertEqual(enriched[0]["busy"], 10)
        self.assertEqual(enriched[0]["free"], 25)

    # Input: A parking record containing 20 busy spaces out of 35.
    # Output: Occupancy percentage of approximately 57.14%.
    # Purpose: Verify correct occupancy percentage calculation.
    def test_occupancy_percentage(self):
        enriched = self.analyzer.enrich_summaries(self.summaries)

        self.assertAlmostEqual(
            enriched[1]["occupancy_percentage"],
            20 / 35 * 100,
            places=4
        )

    # Input: A parking record with zero total spaces.
    # Output: Zero occupancy percentage.
    # Purpose: Verify safe handling of division by zero.
    def test_zero_total_spaces(self):
        summaries = [
            {
                "image_name": "2015-11-12_1009.jpg",
                "total": 0,
                "busy": 0,
                "free": 0
            }
        ]

        enriched = self.analyzer.enrich_summaries(summaries)

        self.assertEqual(
            enriched[0]["occupancy_percentage"],
            0
        )

    # Input: Three enriched parking occupancy summaries.
    # Output: Peak busy count of 30 at 09:09.
    # Purpose: Verify peak occupancy detection.
    def test_peak_occupancy(self):
        enriched = self.analyzer.enrich_summaries(self.summaries)
        statistics = self.analyzer.calculate_statistics(enriched)

        self.assertEqual(statistics["peak_busy"], 30)
        self.assertEqual(
            statistics["peak_time"],
            datetime(2015, 11, 12, 9, 9)
        )

    # Input: Three enriched parking occupancy summaries.
    # Output: Minimum busy count of 10 at 08:09.
    # Purpose: Verify minimum occupancy detection.
    def test_minimum_occupancy(self):
        enriched = self.analyzer.enrich_summaries(self.summaries)
        statistics = self.analyzer.calculate_statistics(enriched)

        self.assertEqual(statistics["min_busy"], 10)
        self.assertEqual(
            statistics["min_time"],
            datetime(2015, 11, 12, 8, 9)
        )

    # Input: Three enriched parking occupancy summaries.
    # Output: Average occupancy of approximately 57.14%.
    # Purpose: Verify average occupancy calculation.
    def test_average_occupancy(self):
        enriched = self.analyzer.enrich_summaries(self.summaries)
        statistics = self.analyzer.calculate_statistics(enriched)

        expected_average = (10 + 20 + 30) / (3 * 35) * 100

        self.assertAlmostEqual(
            statistics["average_occupancy_percentage"],
            expected_average,
            places=4
        )

    # Input: Empty list of parking occupancy summaries.
    # Output: Zero statistics and no peak/minimum timestamps.
    # Purpose: Verify safe handling of empty datasets.
    def test_empty_statistics(self):
        statistics = self.analyzer.calculate_statistics([])

        self.assertEqual(statistics["peak_busy"], 0)
        self.assertIsNone(statistics["peak_time"])
        self.assertEqual(statistics["min_busy"], 0)
        self.assertIsNone(statistics["min_time"])
        self.assertEqual(
            statistics["average_occupancy_percentage"],
            0
        )


if __name__ == "__main__":
    unittest.main()
