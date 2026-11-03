class LineCrossingCounter:

    # Input: Start and end points of a counting line segment.
    # Output: Initialized direction-aware line-crossing counter.
    # Purpose: Count unique tracked vehicles crossing the selected line in each direction.
    def __init__(self, start_point, end_point):
        self.start_point = start_point
        self.end_point = end_point

        self.counted_track_ids = set()
        self.previous_positions = {}
        self.crossing_directions = {}

        self.total_count = 0
        self.direction_a_to_b = 0
        self.direction_b_to_a = 0


    # Input: Point and counting-line endpoints.
    # Output: Signed value indicating which side of the line the point lies on.
    # Purpose: Determine a vehicle's side relative to the counting line.
    def _side_of_line(self, point):
        x, y = point

        x1, y1 = self.start_point
        x2, y2 = self.end_point

        return (
            (x2 - x1) * (y - y1)
            - (y2 - y1) * (x - x1)
        )


    # Input: Previous and current vehicle center points.
    # Output: Crossing direction or None.
    # Purpose: Detect whether movement crossed the finite line segment and determine direction.
    def _get_crossing_direction(
        self,
        previous_point,
        current_point
    ):
        previous_side = self._side_of_line(
            previous_point
        )

        current_side = self._side_of_line(
            current_point
        )

        # No side change means no crossing.
        if previous_side * current_side > 0:
            return None

        # Ignore the rare case where both points are exactly on the line.
        if previous_side == 0 and current_side == 0:
            return None

        x1, y1 = self.start_point
        x2, y2 = self.end_point

        line_min_x = min(x1, x2)
        line_max_x = max(x1, x2)
        line_min_y = min(y1, y2)
        line_max_y = max(y1, y2)

        previous_x, previous_y = previous_point
        current_x, current_y = current_point

        movement_min_x = min(
            previous_x,
            current_x
        )

        movement_max_x = max(
            previous_x,
            current_x
        )

        movement_min_y = min(
            previous_y,
            current_y
        )

        movement_max_y = max(
            previous_y,
            current_y
        )

        overlaps_x = (
            movement_max_x >= line_min_x
            and movement_min_x <= line_max_x
        )

        overlaps_y = (
            movement_max_y >= line_min_y
            and movement_min_y <= line_max_y
        )

        if not (
            overlaps_x
            and overlaps_y
        ):
            return None

        if previous_side < 0 and current_side >= 0:
            return "A_TO_B"

        if previous_side > 0 and current_side <= 0:
            return "B_TO_A"

        return None


    # Input: Tracked vehicles from the current frame.
    # Output: Number of new crossings detected in this frame.
    # Purpose: Count each tracked vehicle once and update directional totals.
    def update(self, tracked_vehicles):
        new_crossings = 0

        for vehicle in tracked_vehicles:
            track_id = vehicle["track_id"]

            center_x = int(
                (
                    vehicle["x1"]
                    + vehicle["x2"]
                )
                / 2
            )

            center_y = int(
                (
                    vehicle["y1"]
                    + vehicle["y2"]
                )
                / 2
            )

            current_point = (
                center_x,
                center_y
            )

            if track_id in self.previous_positions:
                previous_point = self.previous_positions[
                    track_id
                ]

                direction = self._get_crossing_direction(
                    previous_point,
                    current_point
                )

                if (
                    direction is not None
                    and track_id not in self.counted_track_ids
                ):
                    self.counted_track_ids.add(
                        track_id
                    )

                    self.crossing_directions[track_id] = direction

                    self.total_count += 1
                    new_crossings += 1

                    if direction == "A_TO_B":
                        self.direction_a_to_b += 1

                    elif direction == "B_TO_A":
                        self.direction_b_to_a += 1

            self.previous_positions[
                track_id
            ] = current_point

        return new_crossings


    # Input: None.
    # Output: Total number of unique vehicles crossing the line.
    # Purpose: Return the accumulated crossing count.
    def get_total_count(self):
        return self.total_count


    # Input: None.
    # Output: Number of vehicles crossing from side A to side B.
    # Purpose: Return the first directional crossing count.
    def get_a_to_b_count(self):
        return self.direction_a_to_b


    # Input: None.
    # Output: Number of vehicles crossing from side B to side A.
    # Purpose: Return the second directional crossing count.
    def get_b_to_a_count(self):
        return self.direction_b_to_a

    # Input: None.
    # Output: Dictionary mapping track IDs to their crossing directions.
    # Purpose: Provide per-vehicle crossing information for later analytics.
    def get_crossing_directions(self):
        return dict(self.crossing_directions)    