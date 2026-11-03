from collections import defaultdict


class TrackHistory:

    # Input: None.
    # Output: Initialized storage for vehicle trajectories.
    # Purpose: Store the center position of each tracked vehicle over time.
    def __init__(self):
        self.histories = defaultdict(list)


    # Input: Tracked vehicles and current frame index.
    # Output: None.
    # Purpose: Add the current center position of every tracked vehicle to its history.
    def update(
        self,
        tracked_vehicles,
        frame_index
    ):
        for vehicle in tracked_vehicles:
            x1 = vehicle["x1"]
            y1 = vehicle["y1"]
            x2 = vehicle["x2"]
            y2 = vehicle["y2"]

            center_x = int(
                (x1 + x2) / 2
            )

            center_y = int(
                (y1 + y2) / 2
            )

            track_id = vehicle["track_id"]

            self.histories[track_id].append(
                {
                    "frame": frame_index,
                    "x": center_x,
                    "y": center_y,
                    "class_name": vehicle["class_name"]
                }
            )

    # Input: Vehicle track ID.
    # Output: Stored trajectory for that vehicle.
    # Purpose: Retrieve the complete movement history of one tracked vehicle.
    def get_history(
        self,
        track_id
    ):
        return self.histories.get(
            track_id,
            []
        )


    # Input: None.
    # Output: Dictionary containing trajectories for all tracked vehicles.
    # Purpose: Retrieve all recorded vehicle trajectories.
    def get_all_histories(self):
        return dict(
            self.histories
        )