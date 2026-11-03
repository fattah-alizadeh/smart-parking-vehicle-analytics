import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics.vehicle_count_validator import VehicleCountValidator


# Input: Validation results.
# Output: Formatted comparison printed to the terminal.
# Purpose: Display directional vehicle counting errors.
def print_validation_results(results):

    print("\nVEHICLE COUNT VALIDATION")
    print("------------------------")

    for direction, result in results.items():

        percentage_error = result["absolute_percentage_error"]

        error_text = (
            f"{percentage_error:.2f}%"
            if percentage_error is not None
            else "N/A"
        )

        print(
            f"{direction.upper()} | "
            f"Manual: {result['manual_count']} | "
            f"AI: {result['ai_count']} | "
            f"Difference: {result['difference']:+d} | "
            f"Error: {error_text}"
        )


# Input: Manual and AI directional counts entered by the user.
# Output: Vehicle counting validation report.
# Purpose: Compare manually observed vehicle crossings with model predictions.
def main():

    validator = VehicleCountValidator()

    manual_a_to_b = int(input("Manual A -> B count: "))
    manual_b_to_a = int(input("Manual B -> A count: "))

    ai_a_to_b = int(input("AI A -> B count: "))
    ai_b_to_a = int(input("AI B -> A count: "))

    results = validator.validate(
        manual_a_to_b=manual_a_to_b,
        manual_b_to_a=manual_b_to_a,
        ai_a_to_b=ai_a_to_b,
        ai_b_to_a=ai_b_to_a
    )

    print_validation_results(results)


if __name__ == "__main__":
    main()