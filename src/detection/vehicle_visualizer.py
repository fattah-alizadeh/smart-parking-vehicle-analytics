import cv2


class VehicleVisualizer:

    # Input: Full image and vehicle detections.
    # Output: Copy of the image with vehicle bounding boxes and labels.
    # Purpose: Visually inspect vehicle detection results.
    def draw_detections(self, image, detections):
        output = image.copy()

        for detection in detections:
            x1 = detection["x1"]
            y1 = detection["y1"]
            x2 = detection["x2"]
            y2 = detection["y2"]

            class_name = detection["class_name"]
            confidence = detection["confidence"]

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            text = (
                f"{class_name} "
                f"{confidence:.0%}"
            )

            cv2.putText(
                output,
                text,
                (x1, max(y1 - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )

        return output