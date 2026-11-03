from datetime import datetime


class SlotOccupancyAnalyzer:

    # Input: Folder-processing summaries containing slot_states for each timestamp.
    # Output: Dictionary mapping each slot ID to its occupancy history.
    # Purpose: Build a time-ordered Free/Busy history for every parking space.
    def build_slot_histories(self, summaries):
        histories = {}

        for summary in summaries:
            timestamp = datetime.strptime(
                summary["image_name"],
                "%Y-%m-%d_%H%M.jpg"
            )

            for slot_id, state in summary["slot_states"].items():

                if slot_id not in histories:
                    histories[slot_id] = []

                histories[slot_id].append(
                    {
                        "timestamp": timestamp,
                        "label": state["label"],
                        "confidence": state["confidence"]
                    }
                )

        return histories


    # Input: One slot's time-ordered occupancy history.
    # Output: Number of times the slot changes between Free and Busy.
    # Purpose: Measure how often occupancy state changes for one parking space.
    def count_state_changes(self, history):
        if len(history) < 2:
            return 0

        changes = 0

        previous_label = history[0]["label"]

        for item in history[1:]:
            current_label = item["label"]

            if current_label != previous_label:
                changes += 1

            previous_label = current_label

        return changes


    # Input: One slot's occupancy history.
    # Output: Percentage of observed timestamps where the slot was Busy.
    # Purpose: Measure how frequently one parking space is occupied.
    def calculate_busy_percentage(self, history):
        if not history:
            return 0.0

        busy_count = sum(
            1
            for item in history
            if item["label"] == "Busy"
        )

        return busy_count / len(history) * 100


    # Input: One slot's occupancy history.
    # Output: Approximate total occupied duration in minutes.
    # Purpose: Estimate how long the parking space was occupied across sampled images.
    def estimate_busy_duration_minutes(self, history):
        if len(history) < 2:
            return 0.0

        busy_minutes = 0.0

        for current, next_item in zip(
            history[:-1],
            history[1:]
        ):
            if current["label"] == "Busy":
                time_difference = (
                    next_item["timestamp"]
                    - current["timestamp"]
                )

                busy_minutes += (
                    time_difference.total_seconds()
                    / 60
                )

        return busy_minutes


    # Input: All folder-processing summaries.
    # Output: List of analytics dictionaries, one for each parking slot.
    # Purpose: Produce slot-level occupancy statistics for the full observation period.
    def analyze(self, summaries):
        histories = self.build_slot_histories(
            summaries
        )

        analytics = []

        for slot_id, history in histories.items():

            analytics.append(
                {
                    "slot_id": slot_id,
                    "observations": len(history),

                    "busy_percentage":
                        self.calculate_busy_percentage(
                            history
                        ),

                    "state_changes":
                        self.count_state_changes(
                            history
                        ),

                    "estimated_busy_minutes":
                        self.estimate_busy_duration_minutes(
                            history
                        )
                }
            )

        return analytics

    # Input: Slot analytics list and number of slots to return.
    # Output: Most frequently occupied parking spaces.
    # Purpose: Identify the parking spaces with the highest occupancy percentage.
    def get_most_used_slots(self, analytics, count=5):
        return sorted(
            analytics,
            key=lambda item: item["busy_percentage"],
            reverse=True
        )[:count]


    # Input: Slot analytics list and number of slots to return.
    # Output: Least frequently occupied parking spaces.
    # Purpose: Identify the parking spaces with the lowest occupancy percentage.
    def get_least_used_slots(self, analytics, count=5):
        return sorted(
            analytics,
            key=lambda item: item["busy_percentage"]
        )[:count]


    # Input: Slot analytics list and number of slots to return.
    # Output: Parking spaces with the most Free/Busy state changes.
    # Purpose: Identify spaces with the highest apparent turnover/activity.
    def get_most_changing_slots(self, analytics, count=5):
        return sorted(
            analytics,
            key=lambda item: item["state_changes"],
            reverse=True
        )[:count]    