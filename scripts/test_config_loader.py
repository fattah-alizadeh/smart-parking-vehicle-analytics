import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config_loader import ConfigLoader


# Input: None.
# Output: Loaded configuration printed to the terminal.
# Purpose: Verify that the YAML configuration can be loaded successfully.
def main():

    config_path = (
        PROJECT_ROOT
        / "config"
        / "vehicle_tracking.yaml"
    )

    loader = ConfigLoader()
    config = loader.load(config_path)

    print("\nCONFIGURATION LOADED SUCCESSFULLY")
    print("---------------------------------")
    print(f"Input video: {config['video']['input_path']}")
    print(f"Model: {config['model']['name']}")
    print(f"Confidence: {config['model']['confidence_threshold']}")
    print(f"Image size: {config['model']['image_size']}")
    print(f"Tracker: {config['model']['tracker']}")
    print(f"Counting line: {config['counting']['line_start']} -> {config['counting']['line_end']}")
    print(f"Minimum trajectory points: {config['analytics']['min_trajectory_points']}")


if __name__ == "__main__":
    main()