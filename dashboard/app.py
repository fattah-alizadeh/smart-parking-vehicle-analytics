
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYTICS_DIR = PROJECT_ROOT / "results" / "analytics"

VEHICLE_VIDEO_PATH = (
    PROJECT_ROOT / "results" / "videos" / "cam_1_browser.mp4"
)

VEHICLE_CSV_PATH = ANALYTICS_DIR / "cam_1_vehicle_movements.csv"
PARKING_CSV_PATH = ANALYTICS_DIR / "parking_occupancy.csv"
SLOT_CSV_PATH = ANALYTICS_DIR / "parking_slot_utilization.csv"
PARKING_HEATMAP_PATH = (
    ANALYTICS_DIR / "parking_slot_utilization_heatmap.jpg"
)


# Input: Path to an analytics CSV file.
# Output: Pandas DataFrame.
# Purpose: Load analytics data for interactive dashboard visualizations.
@st.cache_data
def load_csv(csv_path):
    return pd.read_csv(csv_path)


# Input: DataFrame, download filename, and button label.
# Output: Downloadable CSV button.
# Purpose: Allow users to export dashboard data.
def display_csv_download(df, filename, label):
    st.download_button(
        label=label,
        data=df.to_csv(index=False),
        file_name=filename,
        mime="text/csv"
    )


# Input: Vehicle analytics DataFrame.
# Output: Summary metrics for vehicle tracking.
# Purpose: Display track counts and directional crossing statistics.
def display_vehicle_summary(df):

    crossed = df[
        df["crossing_direction"].isin(["A_TO_B", "B_TO_A"])
    ]

    a_to_b = int((crossed["crossing_direction"] == "A_TO_B").sum())
    b_to_a = int((crossed["crossing_direction"] == "B_TO_A").sum())

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Reliable Tracks", len(df))
    col2.metric("Crossings in CSV", len(crossed))
    col3.metric("A → B", a_to_b)
    col4.metric("B → A", b_to_a)

    st.caption(
        "Vehicle statistics are based on filtered track records. "
        "Tracks are not necessarily distinct physical vehicles."
    )


# Input: Vehicle analytics DataFrame.
# Output: Interactive vehicle analytics charts.
# Purpose: Visualize classes, crossing directions, durations, and distances.
def display_vehicle_charts(df):

    st.subheader("Vehicle Analytics")

    col1, col2 = st.columns(2)

    with col1:
        counts = (
            df["class_name"]
            .value_counts()
            .rename_axis("Vehicle Class")
            .reset_index(name="Track Count")
        )

        fig = px.bar(
            counts,
            x="Vehicle Class",
            y="Track Count",
            title="Tracks by Vehicle Class"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        counts = (
            df["crossing_direction"]
            .value_counts()
            .rename_axis("Direction")
            .reset_index(name="Track Count")
        )

        fig = px.bar(
            counts,
            x="Direction",
            y="Track Count",
            title="Tracks by Crossing Direction"
        )
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        fig = px.histogram(
            df,
            x="duration_seconds",
            nbins=30,
            title="Track Duration Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = px.histogram(
            df,
            x="distance_pixels",
            nbins=30,
            title="Trajectory Distance Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)


# Input: Vehicle analytics DataFrame.
# Output: Filterable vehicle records table.
# Purpose: Explore individual vehicle movement records.
def display_vehicle_table(df):

    st.subheader("Vehicle Movement Records")

    classes = sorted(df["class_name"].dropna().unique().tolist())
    directions = sorted(
        df["crossing_direction"].dropna().unique().tolist()
    )

    selected_classes = st.multiselect(
        "Vehicle class",
        options=classes,
        default=classes
    )

    selected_directions = st.multiselect(
        "Crossing direction",
        options=directions,
        default=directions
    )

    filtered = df[
        df["class_name"].isin(selected_classes)
        & df["crossing_direction"].isin(selected_directions)
    ]

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True
    )

    display_csv_download(
        filtered,
        "filtered_vehicle_movements.csv",
        "Download Vehicle Analytics CSV"
    )


# Input: None.
# Output: Vehicle traffic dashboard.
# Purpose: Display annotated video, vehicle counts, and movement analytics.
def display_vehicle_tab():

    st.header("Vehicle Traffic Analytics")

    if not VEHICLE_CSV_PATH.is_file():
        st.warning("Vehicle analytics CSV is missing.")
        return

    df = load_csv(str(VEHICLE_CSV_PATH))

    if df.empty:
        st.warning("Vehicle analytics CSV is empty.")
        return

    display_vehicle_summary(df)

    st.divider()

    st.subheader("Annotated Traffic Video")

    if VEHICLE_VIDEO_PATH.is_file():
        st.video(str(VEHICLE_VIDEO_PATH), format="video/mp4")
    else:
        st.warning("Browser-compatible annotated video is missing.")

    st.divider()

    display_vehicle_charts(df)

    st.divider()

    display_vehicle_table(df)


# Input: Parking occupancy DataFrame.
# Output: Summary metrics.
# Purpose: Display parking capacity, average occupancy, and peak utilization.
def display_parking_summary(df):

    total_spaces = int(df["total"].max())
    average_occupancy = df["occupancy_percentage"].mean()

    peak_row = df.loc[df["busy"].idxmax()]
    minimum_row = df.loc[df["busy"].idxmin()]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Parking Spaces", total_spaces)
    col2.metric("Average Occupancy", f"{average_occupancy:.1f}%")
    col3.metric("Peak Busy Spaces", int(peak_row["busy"]))
    col4.metric("Minimum Busy Spaces", int(minimum_row["busy"]))

    st.caption(
        f"Peak occupancy: {peak_row['timestamp']:%Y-%m-%d %H:%M} | "
        f"Minimum occupancy: {minimum_row['timestamp']:%Y-%m-%d %H:%M}"
    )


# Input: Parking occupancy DataFrame.
# Output: Interactive occupancy charts.
# Purpose: Visualize parking occupancy changes across the image sequence.
def display_parking_charts(df):

    st.subheader("Parking Occupancy Over Time")

    chart_df = df.melt(
        id_vars=["timestamp"],
        value_vars=["busy", "free"],
        var_name="Parking Status",
        value_name="Spaces"
    )

    fig = px.line(
        chart_df,
        x="timestamp",
        y="Spaces",
        color="Parking Status",
        markers=True,
        title="Busy and Free Parking Spaces"
    )

    st.plotly_chart(fig, use_container_width=True)

    fig_percentage = px.line(
        df,
        x="timestamp",
        y="occupancy_percentage",
        markers=True,
        title="Parking Occupancy Percentage",
        labels={
            "occupancy_percentage": "Occupancy (%)",
            "timestamp": "Time"
        }
    )

    fig_percentage.update_yaxes(range=[0, 100])

    st.plotly_chart(fig_percentage, use_container_width=True)


# Input: Parking slot utilization DataFrame.
# Output: Interactive utilization charts and slot analytics table.
# Purpose: Identify frequently occupied and frequently changing spaces.
def display_slot_analytics(df):

    st.subheader("Parking Space Utilization")

    top_used = df.nlargest(10, "busy_percentage")

    fig = px.bar(
        top_used,
        x="slot_id",
        y="busy_percentage",
        title="Top 10 Most Utilized Parking Spaces",
        labels={
            "slot_id": "Parking Space",
            "busy_percentage": "Busy Percentage (%)"
        }
    )

    fig.update_yaxes(range=[0, 100])

    st.plotly_chart(fig, use_container_width=True)

    most_changing = df.nlargest(10, "state_changes")

    fig_changes = px.bar(
        most_changing,
        x="slot_id",
        y="state_changes",
        title="Top 10 Parking Spaces by State Changes"
    )

    st.plotly_chart(fig_changes, use_container_width=True)

    st.subheader("Individual Parking Space Records")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    display_csv_download(
        df,
        "parking_slot_utilization.csv",
        "Download Slot Utilization CSV"
    )


# Input: None.
# Output: Interactive parking occupancy dashboard.
# Purpose: Display occupancy trends, parking statistics, and slot utilization.
def display_parking_tab():

    st.header("Parking Occupancy Analytics")

    st.markdown(
        "Parking-space occupancy analysis using "
        "**MobileNetV3 and CNR-EXT parking images**."
    )

    if not PARKING_CSV_PATH.is_file():
        st.warning("Parking occupancy CSV is missing.")
        return

    if not SLOT_CSV_PATH.is_file():
        st.warning("Parking slot utilization CSV is missing.")
        return

    occupancy_df = load_csv(str(PARKING_CSV_PATH))
    slot_df = load_csv(str(SLOT_CSV_PATH))

    if occupancy_df.empty or slot_df.empty:
        st.warning("Parking analytics data is empty.")
        return

    occupancy_df["timestamp"] = pd.to_datetime(
        occupancy_df["timestamp"]
    )

    occupancy_df = occupancy_df.sort_values("timestamp")

    display_parking_summary(occupancy_df)

    st.divider()

    display_parking_charts(occupancy_df)

    st.divider()

    display_slot_analytics(slot_df)

    st.divider()

    st.subheader("Parking Utilization Heatmap")

    if PARKING_HEATMAP_PATH.is_file():
        st.image(
            str(PARKING_HEATMAP_PATH),
            caption="Parking space utilization across the analyzed image sequence.",
            use_container_width=True
        )
    else:
        st.warning("Parking utilization heatmap is missing.")

    st.divider()

    st.subheader("Parking Occupancy Records")

    st.dataframe(
        occupancy_df,
        use_container_width=True,
        hide_index=True
    )

    display_csv_download(
        occupancy_df,
        "parking_occupancy.csv",
        "Download Parking Occupancy CSV"
    )


# Input: None.
# Output: Complete Streamlit dashboard.
# Purpose: Combine vehicle traffic and parking occupancy analytics.
def main():

    st.set_page_config(
        page_title="Smart Parking & Vehicle Analytics",
        page_icon="🚗",
        layout="wide"
    )

    st.title("Smart Parking & Vehicle Analytics")

    st.markdown(
        "AI-powered computer vision system for "
        "**vehicle traffic analysis** and "
        "**parking occupancy monitoring**."
    )

    vehicle_tab, parking_tab = st.tabs(
        [
            "Vehicle Traffic Analytics",
            "Parking Occupancy Analytics"
        ]
    )

    with vehicle_tab:
        display_vehicle_tab()

    with parking_tab:
        display_parking_tab()


if __name__ == "__main__":
    main()
