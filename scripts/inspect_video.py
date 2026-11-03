import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# Input: Path to a video file.
# Output: FPS, resolution, frame count, and duration.
# Purpose: Inspect basic video properties before detection and tracking.
def inspect_video(video_path):
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise ValueError(
            f"Could not open video: {video_path}"
        )

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    width = int(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    duration_seconds = (
        frame_count / fps
        if fps > 0
        else 0
    )

    capture.release()

    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_seconds": duration_seconds
    }


# Input: None.
# Output: Prints basic properties of cam_1.mp4.
# Purpose: Verify the video is suitable for vehicle tracking.
def main():
    video_path = (
        PROJECT_ROOT
        / "data"
        / "public"
        / "aicity"
        / "cam_1.mp4"
    )

    info = inspect_video(
        video_path
    )

    print("VIDEO INFORMATION")
    print("-----------------")
    print("FPS:", info["fps"])
    print("Frames:", info["frame_count"])
    print(
        "Resolution:",
        f"{info['width']}x{info['height']}"
    )
    print(
        "Duration:",
        f"{info['duration_seconds']:.2f} seconds"
    )


if __name__ == "__main__":
    main()