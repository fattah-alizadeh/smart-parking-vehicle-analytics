from ultralytics import YOLO


class VehicleTracker:

    # Input: YOLO model name, tracker config, confidence threshold, and inference image size.
    # Output: Initialized vehicle tracker.
    # Purpose: Track vehicles across consecutive video frames using persistent IDs.
    def __init__(
        self,
        model_name="yolo11n.pt",
        tracker_config="bytetrack.yaml",
        confidence_threshold=0.25,
        image_size=960
    ):
        self.model = YOLO(model_name)
        self.tracker_config = tracker_config
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size

        # COCO vehicle classes:
        # 2 = car
        # 3 = motorcycle
        # 5 = bus
        # 7 = truck
        self.vehicle_class_ids = {
            2,
            3,
            5,
            7
        }


    # Input: One consecutive video frame.
    # Output: List of tracked vehicles with persistent track IDs.
    # Purpose: Detect and track vehicles while preserving IDs between frames.
    def track_frame(self, frame):
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_config,
            conf=self.confidence_threshold,
            imgsz=self.image_size,
            verbose=False
        )

        tracked_vehicles = []

        for result in results:
            boxes = result.boxes

            if boxes is None:
                continue

            for box in boxes:
                class_id = int(
                    box.cls.item()
                )

                if class_id not in self.vehicle_class_ids:
                    continue

                # Tracking ID may be unavailable briefly for some detections.
                if box.id is None:
                    continue

                track_id = int(
                    box.id.item()
                )

                confidence = float(
                    box.conf.item()
                )

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                tracked_vehicles.append(
                    {
                        "track_id": track_id,
                        "class_id": class_id,
                        "class_name": self.model.names[class_id],
                        "confidence": confidence,
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2)
                    }
                )

        return tracked_vehicles