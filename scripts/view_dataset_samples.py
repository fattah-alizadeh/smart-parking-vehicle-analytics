import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.image_sampler import ImageSampler


# Input: None.
# Output: Displays random free and occupied parking-space images.
# Purpose: Visually inspect representative samples from the CNRPark dataset.
def main():
    sampler = ImageSampler(PROJECT_ROOT / "data/public/cnrpark")

    free_images = sampler.sample_images(
        camera="A",
        class_name="Free",
        count=4
    )

    busy_images = sampler.sample_images(
        camera="A",
        class_name="Busy",
        count=4
    )

    sampler.display_images(
        free_images,
        "Camera A - Free Parking Spaces"
    )

    sampler.display_images(
        busy_images,
        "Camera A - Busy Parking Spaces"
    )


if __name__ == "__main__":
    main()
