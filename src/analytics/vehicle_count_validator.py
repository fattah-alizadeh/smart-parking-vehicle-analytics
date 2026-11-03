class VehicleCountValidator:

    # Input: Manual vehicle count and AI vehicle count.
    # Output: Dictionary containing count difference and absolute percentage error.
    # Purpose: Measure how closely an AI vehicle count matches a manually verified count.
    def compare_counts(self, manual_count, ai_count):

        if manual_count < 0 or ai_count < 0:
            raise ValueError("Vehicle counts cannot be negative.")

        difference = ai_count - manual_count
        absolute_error = abs(difference)

        if manual_count == 0:
            percentage_error = 0.0 if ai_count == 0 else None
        else:
            percentage_error = (
                absolute_error / manual_count
            ) * 100

        return {
            "manual_count": manual_count,
            "ai_count": ai_count,
            "difference": difference,
            "absolute_error": absolute_error,
            "absolute_percentage_error": percentage_error
        }

    # Input: Manual and AI counts for both crossing directions.
    # Output: Validation results for each direction and the combined total.
    # Purpose: Summarize directional vehicle counting errors.
    def validate(self, manual_a_to_b, manual_b_to_a, ai_a_to_b, ai_b_to_a):

        return {
            "a_to_b": self.compare_counts(
                manual_a_to_b,
                ai_a_to_b
            ),
            "b_to_a": self.compare_counts(
                manual_b_to_a,
                ai_b_to_a
            ),
            "total": self.compare_counts(
                manual_a_to_b + manual_b_to_a,
                ai_a_to_b + ai_b_to_a
            )
        }