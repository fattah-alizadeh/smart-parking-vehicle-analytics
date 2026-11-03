from pathlib import Path


class ParkingFolderProcessor:

    # Input: Reusable parking occupancy pipeline.
    # Output: Initialized folder processor.
    # Purpose: Apply the parking pipeline to multiple images from the same camera.
    def __init__(self, pipeline):
        self.pipeline = pipeline


    # Input: Folder containing parking images and matching camera CSV path.
    # Output: Occupancy summaries including individual slot predictions for each image.
    # Purpose: Process parking images over time while preserving slot-level occupancy information.
    def process_folder(
        self,
        image_folder,
        csv_path
    ):
        image_folder = Path(image_folder)

        image_paths = sorted(
            image_folder.glob("*.jpg")
        )

        summaries = []

        for image_path in image_paths:

            _, results, _ = self.pipeline.process_image(
                image_path=image_path,
                csv_path=csv_path
            )

            summary = self.pipeline.calculate_summary(
                results
            )

            slot_states = {
                result["slot_id"]: {
                    "label": result["label"],
                    "confidence": result["confidence"]
                }
                for result in results
            }

            summaries.append(
                {
                    "image_name": image_path.name,
                    "total": summary["total"],
                    "free": summary["free"],
                    "busy": summary["busy"],
                    "slot_states": slot_states
                }
            )

        return summaries