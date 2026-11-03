import math


class VehicleMovementAnalyzer:

    # Input: One vehicle trajectory history.
    # Output: Total travelled distance in pixels.
    # Purpose: Measure how far a tracked vehicle moved across the image.
    def calculate_trajectory_distance(self, history):
        total_distance = 0.0

        for current, next_point in zip(
            history[:-1],
            history[1:]
        ):
            dx = next_point["x"] - current["x"]
            dy = next_point["y"] - current["y"]

            total_distance += math.sqrt(
                dx * dx + dy * dy
            )

        return total_distance


    # Input: One vehicle trajectory history and video FPS.
    # Output: Approximate track duration in seconds.
    # Purpose: Measure how long the vehicle was tracked.
    def calculate_duration_seconds(
        self,
        history,
        fps
    ):
        if len(history) < 2 or fps <= 0:
            return 0.0

        first_frame = history[0]["frame"]
        last_frame = history[-1]["frame"]

        return (
            last_frame - first_frame
        ) / fps


    # Input: All trajectory histories, video FPS, crossing directions, and minimum required trajectory points.
    # Output: List of reliable per-vehicle movement records enriched with class and crossing direction.
    # Purpose: Convert raw tracking histories into useful vehicle movement analytics.
    def analyze(
        self,
        histories,
        fps,
        crossing_directions,
        min_trajectory_points=5
    ):
        records = []

        for track_id, history in histories.items():
            if len(history) < min_trajectory_points:
                continue

            first_frame = history[0]["frame"]
            last_frame = history[-1]["frame"]

            class_name = history[0]["class_name"]

            crossing_direction = crossing_directions.get(
                track_id,
                "NOT_CROSSED"
            )

            records.append(
                {
                    "track_id": track_id,
                    "class_name": class_name,
                    "first_frame": first_frame,
                    "last_frame": last_frame,
                    "trajectory_points": len(history),
                    "duration_seconds":
                        self.calculate_duration_seconds(
                            history,
                            fps
                        ),
                    "distance_pixels":
                        self.calculate_trajectory_distance(
                            history
                        ),
                    "crossing_direction": crossing_direction
                }
            )

        return records