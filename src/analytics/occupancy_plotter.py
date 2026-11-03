from datetime import datetime

import matplotlib.pyplot as plt


class OccupancyPlotter:

    # Input: Parking occupancy summaries containing image names and Free/Busy counts.
    # Output: Lists of timestamps, Free counts, and Busy counts.
    # Purpose: Convert folder-processing results into time-series data for plotting.
    def _prepare_data(self, summaries):
        timestamps = []
        free_counts = []
        busy_counts = []

        for summary in summaries:
            timestamp = datetime.strptime(
                summary["image_name"],
                "%Y-%m-%d_%H%M.jpg"
            )

            timestamps.append(timestamp)
            free_counts.append(summary["free"])
            busy_counts.append(summary["busy"])

        return timestamps, free_counts, busy_counts


    # Input: Parking occupancy summaries and optional output file path.
    # Output: Displays the occupancy chart and optionally saves it to disk.
    # Purpose: Visualize how Free and Busy parking-space counts change over time.
    def plot(self, summaries, output_path=None):
        timestamps, free_counts, busy_counts = self._prepare_data(
            summaries
        )

        plt.figure(
            figsize=(11, 6)
        )

        plt.plot(
            timestamps,
            busy_counts,
            marker="o",
            label="Busy"
        )

        plt.plot(
            timestamps,
            free_counts,
            marker="o",
            label="Free"
        )

        plt.xlabel("Time")
        plt.ylabel("Number of Parking Spaces")
        plt.title("Parking Occupancy Over Time")

        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)

        plt.tight_layout()

        if output_path is not None:
            plt.savefig(
                output_path,
                dpi=150
            )

        plt.show()