import cv2


class ParkingVisualizer:

    # Input: Full parking-lot image and parking classification results.
    # Output: Copy of the image with colored parking boxes, labels, and summary.
    # Purpose: Visualize Free/Busy predictions on the complete parking-lot image.
    def draw_results(self, image, results):
        output = image.copy()

        free_count = 0
        busy_count = 0

        for result in results:
            x = result["x"]
            y = result["y"]
            width = result["width"]
            height = result["height"]

            label = result["label"]
            confidence = result["confidence"]
            slot_id = result["slot_id"]

            if label == "Free":
                color = (0, 255, 0)
                free_count += 1
            else:
                color = (0, 0, 255)
                busy_count += 1

            cv2.rectangle(
                output,
                (x, y),
                (x + width, y + height),
                color,
                2
            )

            text = (
                f"{slot_id} "
                f"{label} "
                f"{confidence:.0%}"
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

        self._draw_summary(
            output,
            total=len(results),
            free_count=free_count,
            busy_count=busy_count
        )

        return output


    # Input: Image and parking occupancy counts.
    # Output: None; modifies the image directly.
    # Purpose: Draw overall parking occupancy statistics on the image.
    def _draw_summary(
        self,
        image,
        total,
        free_count,
        busy_count
    ):
        summary_lines = [
            f"Total: {total}",
            f"Free: {free_count}",
            f"Busy: {busy_count}"
        ]

        x = 15
        y = 30

        for line in summary_lines:
            cv2.putText(
                image,
                line,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            y += 30