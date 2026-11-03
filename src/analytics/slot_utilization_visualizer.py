import cv2


class SlotUtilizationVisualizer:

    # Input: Full parking image, parking-space definitions, and slot analytics.
    # Output: Copy of the image with utilization information drawn on each parking space.
    # Purpose: Visualize how frequently each parking space was occupied over time.
    def draw_utilization(
        self,
        image,
        parking_spaces,
        slot_analytics
    ):
        output = image.copy()

        analytics_by_slot = {
            item["slot_id"]: item
            for item in slot_analytics
        }

        for space in parking_spaces:
            slot_id = space["slot_id"]

            if slot_id not in analytics_by_slot:
                continue

            analytics = analytics_by_slot[slot_id]
            busy_percentage = analytics["busy_percentage"]

            color = self._get_utilization_color(
                busy_percentage
            )

            x = space["x"]
            y = space["y"]
            width = space["width"]
            height = space["height"]

            cv2.rectangle(
                output,
                (x, y),
                (x + width, y + height),
                color,
                2
            )

            text = (
                f"{slot_id} "
                f"{busy_percentage:.0f}%"
            )

            cv2.putText(
                output,
                text,
                (x, max(y - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA
            )

        return output


    # Input: Parking-space busy percentage.
    # Output: BGR color representing low, medium, or high utilization.
    # Purpose: Convert utilization percentage into an easy-to-read visual category.
    def _get_utilization_color(
        self,
        busy_percentage
    ):
        if busy_percentage < 40:
            return (0, 255, 0)

        if busy_percentage < 75:
            return (0, 255, 255)

        return (0, 0, 255)