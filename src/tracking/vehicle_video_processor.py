import cv2


class VehicleVideoProcessor:

    # Input: Video path, output path, tracker, track history, crossing counter, and visualization function.
    # Output: Annotated MP4 video and updated tracking/analytics objects.
    # Purpose: Process an entire video, track vehicles, count crossings, and save annotated frames.
    def process(
        self,
        video_path,
        output_path,
        tracker,
        track_history,
        crossing_counter,
        draw_frame
    ):
        capture = cv2.VideoCapture(str(video_path))

        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = capture.get(cv2.CAP_PROP_FPS)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0:
            capture.release()
            raise ValueError("Video FPS must be greater than zero.")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"Cannot create output video: {output_path}")

        frame_index = 0

        try:
            while True:
                success, frame = capture.read()

                if not success:
                    break

                tracked_vehicles = tracker.track_frame(frame)

                track_history.update(
                    tracked_vehicles,
                    frame_index
                )

                crossing_counter.update(tracked_vehicles)

                annotated_frame = draw_frame(
                    frame,
                    tracked_vehicles,
                    track_history,
                    crossing_counter,
                    frame_index
                )

                writer.write(annotated_frame)

                frame_index += 1

                if frame_index % 100 == 0:
                    print(
                        f"Processed {frame_index}/{total_frames} frames"
                    )

        finally:
            capture.release()
            writer.release()

        return frame_index