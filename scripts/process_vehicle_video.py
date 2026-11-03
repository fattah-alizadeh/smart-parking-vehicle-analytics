
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from src.common.config_loader import ConfigLoader
from src.tracking.vehicle_tracker import VehicleTracker
from src.tracking.track_history import TrackHistory
from src.tracking.line_crossing_counter import LineCrossingCounter
from src.tracking.vehicle_video_processor import VehicleVideoProcessor

from src.analytics.vehicle_movement_analyzer import VehicleMovementAnalyzer
from src.analytics.vehicle_analytics_exporter import VehicleAnalyticsExporter

from scripts.test_vehicle_tracking import draw_tracking_output


# Input: A path relative to the project root, or an absolute path.
# Output: Resolved absolute Path object.
# Purpose: Make file paths independent of the current working directory.
def resolve_project_path(path_value):
    path = Path(path_value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


# Input: Path to the source video.
# Output: Video frame rate.
# Purpose: Read the actual video FPS for accurate movement analytics.
def get_video_fps(video_path):
    capture = cv2.VideoCapture(str(video_path))

    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = capture.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            raise ValueError(f"Invalid video FPS: {fps}")

        return fps
    finally:
        capture.release()


# Input: None.
# Output: Annotated video, analytics CSV, and processing summary.
# Purpose: Process a video using settings loaded from YAML configuration.
def main():

    # Load configuration.
    config_path = PROJECT_ROOT / "config" / "vehicle_tracking.yaml"

    loader = ConfigLoader()
    config = loader.load(config_path)

    video_config = config["video"]
    model_config = config["model"]
    counting_config = config["counting"]
    analytics_config = config["analytics"]

    # Resolve input and output paths.
    video_path = resolve_project_path(
        video_config["input_path"]
    )

    output_path = resolve_project_path(
        video_config["output_path"]
    )

    csv_path = resolve_project_path(
        analytics_config["output_csv"]
    )

    # Read counting line coordinates.
    line_start = tuple(counting_config["line_start"])
    line_end = tuple(counting_config["line_end"])

    # Read actual video FPS.
    fps = get_video_fps(video_path)

    print("\nVEHICLE VIDEO PROCESSING")
    print("------------------------")
    print(f"Input video: {video_path}")
    print(f"Model: {model_config['name']}")
    print(f"Confidence: {model_config['confidence_threshold']}")
    print(f"Image size: {model_config['image_size']}")
    print(f"Tracker: {model_config['tracker']}")
    print(f"FPS: {fps}")
    print(f"Counting line: {line_start} -> {line_end}")

    # Initialize tracking components.
    tracker = VehicleTracker(
        model_name=model_config["name"],
        tracker_config=model_config["tracker"],
        confidence_threshold=model_config["confidence_threshold"],
        image_size=model_config["image_size"]
    )

    track_history = TrackHistory()

    crossing_counter = LineCrossingCounter(
        start_point=line_start,
        end_point=line_end
    )

    processor = VehicleVideoProcessor()

    # Input: Current frame, tracked vehicles, track history, counter, and frame index.
    # Output: Annotated video frame.
    # Purpose: Reuse the existing visualization function with configured line coordinates.
    def draw_frame(
        frame,
        tracked_vehicles,
        history,
        counter,
        frame_index
    ):
        return draw_tracking_output(
            frame,
            tracked_vehicles,
            history,
            line_start,
            line_end,
            counter.get_total_count(),
            counter.get_a_to_b_count(),
            counter.get_b_to_a_count()
        )

    # Process the complete video.
    processed_frames = processor.process(
        video_path=video_path,
        output_path=output_path,
        tracker=tracker,
        track_history=track_history,
        crossing_counter=crossing_counter,
        draw_frame=draw_frame
    )

    # Generate vehicle movement analytics.
    analyzer = VehicleMovementAnalyzer()

    vehicle_records = analyzer.analyze(
        histories=track_history.get_all_histories(),
        fps=fps,
        crossing_directions=crossing_counter.get_crossing_directions(),
        min_trajectory_points=analytics_config["min_trajectory_points"]
    )

    # Export analytics to CSV.
    exporter = VehicleAnalyticsExporter()

    exporter.export_to_csv(
        records=vehicle_records,
        output_path=csv_path
    )

    # Print processing summary.
    print("\nVIDEO PROCESSING COMPLETED")
    print("--------------------------")
    print(f"Processed frames: {processed_frames}")
    print(f"Raw tracked IDs: {len(track_history.get_all_histories())}")
    print(f"Reliable vehicle tracks: {len(vehicle_records)}")
    print(f"Total crossings: {crossing_counter.get_total_count()}")
    print(f"A -> B: {crossing_counter.get_a_to_b_count()}")
    print(f"B -> A: {crossing_counter.get_b_to_a_count()}")
    print(f"Annotated video: {output_path}")
    print(f"Analytics CSV: {csv_path}")


if __name__ == "__main__":
    main()
