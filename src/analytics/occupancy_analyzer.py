from datetime import datetime


class OccupancyAnalyzer:

    # Input: Occupancy summaries produced by ParkingFolderProcessor.
    # Output: List of summaries enriched with timestamp and occupancy percentage.
    # Purpose: Convert raw parking counts into useful analytical values.
    def enrich_summaries(self, summaries):
        enriched = []

        for summary in summaries:
            timestamp = datetime.strptime(
                summary["image_name"],
                "%Y-%m-%d_%H%M.jpg"
            )

            occupancy_percentage = (
                summary["busy"] / summary["total"] * 100
                if summary["total"] > 0
                else 0
            )

            enriched.append(
                {
                    **summary,
                    "timestamp": timestamp,
                    "occupancy_percentage": occupancy_percentage
                }
            )

        return enriched


    # Input: Enriched occupancy summaries.
    # Output: Dictionary containing overall parking analytics.
    # Purpose: Calculate peak occupancy, minimum occupancy, and average occupancy.
    def calculate_statistics(self, summaries):
        if not summaries:
            return {
                "peak_busy": 0,
                "peak_time": None,
                "min_busy": 0,
                "min_time": None,
                "average_occupancy_percentage": 0
            }

        peak_summary = max(
            summaries,
            key=lambda item: item["busy"]
        )

        min_summary = min(
            summaries,
            key=lambda item: item["busy"]
        )

        average_occupancy_percentage = sum(
            item["occupancy_percentage"]
            for item in summaries
        ) / len(summaries)

        return {
            "peak_busy": peak_summary["busy"],
            "peak_time": peak_summary["timestamp"],
            "min_busy": min_summary["busy"],
            "min_time": min_summary["timestamp"],
            "average_occupancy_percentage": average_occupancy_percentage
        }