import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.tracking.line_crossing_counter import LineCrossingCounter
from src.tracking.track_history import TrackHistory
from src.tracking.vehicle_tracker import VehicleTracker
from src.analytics.vehicle_movement_analyzer import VehicleMovementAnalyzer
from src.analytics.vehicle_analytics_exporter import VehicleAnalyticsExporter


selected_points = []


# Input: OpenCV mouse event information.
# Output: Updates selected_points with clicked coordinates.
# Purpose: Allow the user to define the counting line interactively.
def select_line_point(
    event,
    x,
    y,
    flags,
    param
):
    if (
        event == cv2.EVENT_LBUTTONDOWN
        and len(selected_points) < 2
    ):
        selected_points.append(
            (x, y)
        )

        print(
            "Selected point:",
            (x, y)
        )


# Input: First video frame.
# Output: Two selected line endpoints.
# Purpose: Let the user visually choose the counting line with the mouse.
def choose_counting_line(frame):
    global selected_points

    selected_points = []

    window_name = "Select Counting Line"

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL
    )

    cv2.setMouseCallback(
        window_name,
        select_line_point
    )

    while True:
        display = frame.copy()

        for point in selected_points:
            cv2.circle(
                display,
                point,
                6,
                (0, 0, 255),
                -1
            )

        if len(selected_points) == 2:
            cv2.line(
                display,
                selected_points[0],
                selected_points[1],
                (0, 0, 255),
                3
            )

        cv2.putText(
            display,
            "Click two points to define counting line",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow(
            window_name,
            display
        )

        key = cv2.waitKey(20) & 0xFF

        if len(selected_points) == 2:
            break

        if key == ord("q"):
            break

    cv2.destroyWindow(
        window_name
    )

    if len(selected_points) != 2:
        raise ValueError(
            "Two counting-line points were not selected."
        )

    return (
        selected_points[0],
        selected_points[1]
    )


# Input: Frame, tracked vehicles, histories, counting-line points, and crossing count.
# Output: Annotated tracking frame.
# Purpose: Draw vehicle tracks, trajectories, selected counting line, and total count.
def draw_tracking_output(
    frame,
    tracked_vehicles,
    track_history,
    line_start,
    line_end,
    total_crossings,
    a_to_b_count,
    b_to_a_count
):
    output = frame.copy()

    for vehicle in tracked_vehicles:
        x1 = vehicle["x1"]
        y1 = vehicle["y1"]
        x2 = vehicle["x2"]
        y2 = vehicle["y2"]

        track_id = vehicle["track_id"]
        class_name = vehicle["class_name"]
        confidence = vehicle["confidence"]

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        text = (
            f"{class_name} "
            f"ID {track_id} "
            f"{confidence:.0%}"
        )

        cv2.putText(
            output,
            f"A -> B: {a_to_b_count}",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            output,
            f"B -> A: {b_to_a_count}",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            output,
            f"Total: {total_crossings}",
            (20, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

        history = track_history.get_history(
            track_id
        )

        for current, next_point in zip(
            history[:-1],
            history[1:]
        ):
            cv2.line(
                output,
                (current["x"], current["y"]),
                (next_point["x"], next_point["y"]),
                (255, 0, 0),
                2
            )

    cv2.line(
        output,
        line_start,
        line_end,
        (0, 0, 255),
        3
    )

   
    return output


# Input: None.
# Output: Interactive line selection followed by vehicle tracking and crossing count.
# Purpose: Count tracked vehicles crossing a user-selected road segment.
def main():
    video_path = (
        PROJECT_ROOT
        / "data"
        / "public"
        / "aicity"
        / "cam_1.mp4"
    )

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise ValueError(
            f"Could not open video: {video_path}"
        )

    fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    max_frames = int(
        fps * 30
    )

    success, first_frame = capture.read()

    if not success:
        raise ValueError(
            "Could not read first video frame."
        )

    line_start, line_end = choose_counting_line(
        first_frame
    )

    print(
        "Counting line:",
        line_start,
        "->",
        line_end
    )

    # Restart video from frame zero after line selection.
    capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        0
    )

    tracker = VehicleTracker(
        model_name="yolo11n.pt",
        tracker_config="bytetrack.yaml",
        confidence_threshold=0.25,
        image_size=960
    )

    track_history = TrackHistory()

    crossing_counter = LineCrossingCounter(
        start_point=line_start,
        end_point=line_end
    )

    frame_index = 0

    while capture.isOpened() and frame_index < max_frames:
        success, frame = capture.read()

        if not success:
            break

        tracked_vehicles = tracker.track_frame(
            frame
        )

        track_history.update(
            tracked_vehicles=tracked_vehicles,
            frame_index=frame_index
        )

        crossing_counter.update(
            tracked_vehicles
        )

        output = draw_tracking_output(
            frame=frame,
            tracked_vehicles=tracked_vehicles,
            track_history=track_history,
            line_start=line_start,
            line_end=line_end,
            total_crossings=crossing_counter.get_total_count(),
            a_to_b_count=crossing_counter.get_a_to_b_count(),
            b_to_a_count=crossing_counter.get_b_to_a_count()
        )

        cv2.putText(
            output,
            f"Frame: {frame_index}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow(
            "Vehicle Tracking - ByteTrack",
            output
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        frame_index += 1

    capture.release()
    cv2.destroyAllWindows()

    analyzer = VehicleMovementAnalyzer()

    
    vehicle_records = analyzer.analyze(
        histories=track_history.get_all_histories(),
        fps=fps,
        crossing_directions=crossing_counter.get_crossing_directions(),
        min_trajectory_points=5
    )

    exporter = VehicleAnalyticsExporter()

    output_path = PROJECT_ROOT / "results" / "analytics" / "vehicle_movements.csv"

    exporter.export_to_csv(
        records=vehicle_records,
        output_path=output_path
    )

    print(f"\nVehicle analytics saved to: {output_path}")
    print(f"Exported vehicle records: {len(vehicle_records)}")

    print("\nVEHICLE MOVEMENT ANALYTICS")
    print("--------------------------")

    
    for record in vehicle_records:
        print(
            f"ID {record['track_id']} -> "
            f"Class: {record['class_name']} | "
            f"First frame: {record['first_frame']} | "
            f"Last frame: {record['last_frame']} | "
            f"Duration: {record['duration_seconds']:.1f}s | "
            f"Trajectory points: {record['trajectory_points']} | "
            f"Distance: {record['distance_pixels']:.1f}px | "
            f"Crossing: {record['crossing_direction']}"
        )

    print(
        "Tracked vehicle IDs:",
        len(track_history.get_all_histories())
    )

    print(
        "Vehicles crossed line:",
        crossing_counter.get_total_count()
    )

    print(
        "A -> B:",
        crossing_counter.get_a_to_b_count()
    )

    print(
        "B -> A:",
        crossing_counter.get_b_to_a_count()
    )

if __name__ == "__main__":
    main()