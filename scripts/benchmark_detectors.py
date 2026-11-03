import sys
from pathlib import Path

import cv2
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.detector_benchmark import DetectorBenchmark
from src.detection.vehicle_detector import VehicleDetector


# Input: None.
# Output: Prints detection speed for YOLO11n and YOLO11s at 960 input size.
# Purpose: Compare detector runtime before choosing a model for tracking.
def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    image_path = (
        PROJECT_ROOT
        / "data"
        / "public"
        / "cnrpark"
        / "nrpark_full"
        / "CNR-EXT_FULL_IMAGE_1000x750"
        / "FULL_IMAGE_1000x750"
        / "SUNNY"
        / "2015-11-12"
        / "camera1"
        / "2015-11-12_1009.jpg"
    )

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"Could not load image: {image_path}"
        )

    benchmark = DetectorBenchmark(
        device=device
    )

    for model_name in [
        "yolo11n.pt",
        "yolo11s.pt"
    ]:
        detector = VehicleDetector(
            model_name=model_name,
            confidence_threshold=0.25,
            image_size=960
        )

        detections = detector.detect(image)

        latency_ms, fps = benchmark.measure_speed(
            detector=detector,
            image=image
        )

        print("\nModel:", model_name)
        print("Detections:", len(detections))
        print(
            f"Average latency: {latency_ms:.2f} ms/frame"
        )
        print(
            f"Throughput: {fps:.2f} FPS"
        )


if __name__ == "__main__":
    main()