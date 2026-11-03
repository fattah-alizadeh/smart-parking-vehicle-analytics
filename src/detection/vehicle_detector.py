from ultralytics import YOLO


class VehicleDetector:

    # Input: Model name, confidence threshold, and inference image size.
    # Output: Initialized vehicle detector.
    # Purpose: Configure the pretrained detector for vehicle detection.
    def __init__(
        self,
        model_name="yolo11n.pt",
        confidence_threshold=0.25,
        image_size=640
    ):
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size

        self.vehicle_class_ids = {
            2,  # car
            3,  # motorcycle
            5,  # bus
            7   # truck
        }

    # Input: Full parking-lot image.
    # Output: List of detected vehicles with box, class, and confidence.
    # Purpose: Detect vehicle objects in one full parking image.
    def detect(self, image):
        print("YOLO inference image size:", self.image_size)

        predictions = self.model.predict(
            source=image,
            conf=self.confidence_threshold,
            imgsz=self.image_size,
            verbose=False
        )

        detections = []

        for prediction in predictions:
            boxes = prediction.boxes

            if boxes is None:
                continue

            for box in boxes:
                class_id = int(box.cls.item())

                if class_id not in self.vehicle_class_ids:
                    continue

                confidence = float(box.conf.item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detections.append(
                    {
                        "class_id": class_id,
                        "class_name": self.model.names[class_id],
                        "confidence": confidence,
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2)
                    }
                )

        return detections