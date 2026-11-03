from pathlib import Path

class DatasetInspector:

    # Input: Root directory of the CNRPark dataset
    # Output: None
    # Purpose: Store the dataset location for later inspection methods
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)

    # Input: Camera name such as "A" or "B" and class name such as "Free" or "Busy"
    # Output: Number of JPG images in the requested folder
    # Purpose: Count parking-space images belonging to a specific camera and class
    def count_images(self, camera, class_name):
        folder = self.dataset_path / camera / class_name
        return len(list(folder.glob("*.jpg")))

    # Input: None
    # Output: Dictionary containing image counts for each camera and class
    # Purpose: Produce a simple summary of the CNRPark dataset
    def get_summary(self):
        return {
            "A_Free": self.count_images("A", "Free"),
            "A_Busy": self.count_images("A", "Busy"),
            "B_Free": self.count_images("B", "Free"),
            "B_Busy": self.count_images("B", "Busy"),
        }
