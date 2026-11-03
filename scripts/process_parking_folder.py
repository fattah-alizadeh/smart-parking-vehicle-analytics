
import cv2
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics.occupancy_analyzer import OccupancyAnalyzer
from src.analytics.occupancy_plotter import OccupancyPlotter
from src.analytics.parking_analytics_exporter import ParkingAnalyticsExporter
from src.analytics.slot_occupancy_analyzer import SlotOccupancyAnalyzer
from src.analytics.slot_utilization_visualizer import SlotUtilizationVisualizer

from src.parking.parking_folder_processor import ParkingFolderProcessor
from src.parking.parking_occupancy_pipeline import ParkingOccupancyPipeline
from src.parking.parking_space_loader import ParkingSpaceLoader


# Input: Folder containing full parking images and matching camera CSV path.
# Output: Occupancy charts, analytics CSV files, utilization heatmap, and summary.
# Purpose: Analyze parking occupancy across images and export dashboard-ready results.
def main():

    if len(sys.argv) != 3:
        print(
            "Usage:\n"
            "python scripts/process_parking_folder.py "
            "<image_folder> <csv_path>"
        )
        return

    image_folder = Path(sys.argv[1])
    csv_path = Path(sys.argv[2])

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model_path = (
        PROJECT_ROOT
        / "results"
        / "models"
        / "mobilenet_v3_cameraB_to_cameraA_augmented_best.pth"
    )

    pipeline = ParkingOccupancyPipeline(
        model_path=model_path,
        device=device
    )

    processor = ParkingFolderProcessor(
        pipeline=pipeline
    )

    summaries = processor.process_folder(
        image_folder=image_folder,
        csv_path=csv_path
    )

    if not summaries:
        print("No parking occupancy records were produced.")
        return

    print("\nPARKING OCCUPANCY OVER TIME")
    print("---------------------------")

    for summary in summaries:
        print(
            f"{summary['image_name']} -> "
            f"Total: {summary['total']} | "
            f"Free: {summary['free']} | "
            f"Busy: {summary['busy']}"
        )

    output_folder = (
        PROJECT_ROOT
        / "results"
        / "analytics"
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    # Generate the occupancy chart.
    plotter = OccupancyPlotter()

    chart_path = (
        output_folder
        / "parking_occupancy_over_time.png"
    )

    plotter.plot(
        summaries=summaries,
        output_path=chart_path
    )

    print(f"\nOccupancy chart saved to: {chart_path}")

    # Calculate overall occupancy analytics.
    analyzer = OccupancyAnalyzer()

    enriched_summaries = analyzer.enrich_summaries(
        summaries
    )

    statistics = analyzer.calculate_statistics(
        enriched_summaries
    )

    print("\nPARKING ANALYTICS")
    print("-----------------")

    print(f"Peak busy count: {statistics['peak_busy']}")

    print(
        "Peak time:",
        statistics["peak_time"].strftime("%H:%M")
        if statistics["peak_time"] is not None
        else "N/A"
    )

    print(f"Minimum busy count: {statistics['min_busy']}")

    print(
        "Minimum time:",
        statistics["min_time"].strftime("%H:%M")
        if statistics["min_time"] is not None
        else "N/A"
    )

    print(
        "Average occupancy:",
        f"{statistics['average_occupancy_percentage']:.2f}%"
    )

    # Calculate slot-level utilization.
    slot_analyzer = SlotOccupancyAnalyzer()

    slot_analytics = slot_analyzer.analyze(
        summaries
    )

    print("\nSLOT-LEVEL ANALYTICS")
    print("--------------------")

    for item in slot_analytics:
        print(
            f"Slot {item['slot_id']} -> "
            f"Busy: {item['busy_percentage']:.1f}% | "
            f"State changes: {item['state_changes']} | "
            f"Estimated busy time: "
            f"{item['estimated_busy_minutes']:.0f} min"
        )

    most_used = slot_analyzer.get_most_used_slots(
        slot_analytics,
        count=5
    )

    least_used = slot_analyzer.get_least_used_slots(
        slot_analytics,
        count=5
    )

    most_changing = slot_analyzer.get_most_changing_slots(
        slot_analytics,
        count=5
    )

    print("\nMOST-USED PARKING SPACES")
    print("------------------------")

    for item in most_used:
        print(
            f"Slot {item['slot_id']} -> "
            f"{item['busy_percentage']:.1f}% busy"
        )

    print("\nLEAST-USED PARKING SPACES")
    print("-------------------------")

    for item in least_used:
        print(
            f"Slot {item['slot_id']} -> "
            f"{item['busy_percentage']:.1f}% busy"
        )

    print("\nMOST-CHANGING PARKING SPACES")
    print("----------------------------")

    for item in most_changing:
        print(
            f"Slot {item['slot_id']} -> "
            f"{item['state_changes']} state changes"
        )

    # Export dashboard-ready analytics.
    exporter = ParkingAnalyticsExporter()

    occupancy_csv_path = (
        output_folder
        / "parking_occupancy.csv"
    )

    slots_csv_path = (
        output_folder
        / "parking_slot_utilization.csv"
    )

    exporter.export_occupancy(
        summaries=enriched_summaries,
        output_path=occupancy_csv_path
    )

    exporter.export_slot_utilization(
        slot_analytics=slot_analytics,
        output_path=slots_csv_path
    )

    print("\nPARKING ANALYTICS EXPORTED")
    print("--------------------------")
    print(f"Occupancy CSV: {occupancy_csv_path}")
    print(f"Slot utilization CSV: {slots_csv_path}")

    # Generate the parking-space utilization heatmap.
    utilization_visualizer = SlotUtilizationVisualizer()
    space_loader = ParkingSpaceLoader()

    parking_spaces = space_loader.load_spaces(
        csv_path
    )

    image_paths = sorted(image_folder.glob("*.jpg"))

    if not image_paths:
        print("No JPG images available for the heatmap.")
        return

    background_image = cv2.imread(
        str(image_paths[0])
    )

    if background_image is None:
        print("Unable to load the heatmap background image.")
        return

    heatmap_image = utilization_visualizer.draw_utilization(
        image=background_image,
        parking_spaces=parking_spaces,
        slot_analytics=slot_analytics
    )

    heatmap_output_path = (
        output_folder
        / "parking_slot_utilization_heatmap.jpg"
    )

    if not cv2.imwrite(
        str(heatmap_output_path),
        heatmap_image
    ):
        raise RuntimeError("Failed to save utilization heatmap.")

    print(
        f"\nUtilization heatmap saved to: {heatmap_output_path}"
    )

    cv2.imshow(
        "Parking Slot Utilization Heatmap",
        heatmap_image
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
