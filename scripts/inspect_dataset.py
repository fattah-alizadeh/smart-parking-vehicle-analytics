import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.dataset_inspector import DatasetInspector


# Input: None
# Output: Prints the CNRPark dataset summary
# Purpose: Run the dataset inspector and display image counts
def main():
    dataset_path = PROJECT_ROOT / "data/public/cnrpark"

    inspector = DatasetInspector(dataset_path)
    summary = inspector.get_summary()

    print("\nCNRPark Dataset Summary")
    print("-----------------------")

    total = 0

    for category, count in summary.items():
        print(f"{category}: {count}")
        total += count

    print("-----------------------")
    print(f"Total images: {total}")


if __name__ == "__main__":
    main()
