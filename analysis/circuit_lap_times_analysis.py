# Average Lap Time Analysis
# Author: Siddhant Gaikwad
# Date: 26 December 2024
# Description: This script analyzes circuits with the longest and shortest average lap times.

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def analyze_average_lap_times(lap_times_file, races_file):
    """
    Analyze and visualize the circuits with the longest and shortest average lap times.

    Parameters:
        lap_times_file (str): Path to the lap times CSV file.
        races_file (str): Path to the races CSV file.

    Returns:
        None
    """
    # Load datasets
    lap_times_df = pd.read_csv(lap_times_file)
    race_df = pd.read_csv(races_file)

    # Step 1: Calculate average lap time for each race
    average_lap_times = lap_times_df.groupby('raceId')['milliseconds'].mean().reset_index()
    average_lap_times['average_lap_seconds'] = average_lap_times['milliseconds'] / 1000  # Convert to seconds

    # Merge with races to get circuit names and years
    average_lap_times_with_names = average_lap_times.merge(
        race_df[['raceId', 'name', 'year']], on='raceId', how='left'
    )

    # Step 2: Calculate the average lap time for each circuit across all years
    average_lap_times_by_circuit = (
        average_lap_times_with_names.groupby('name')['average_lap_seconds'].mean().reset_index()
    )
    average_lap_times_by_circuit = average_lap_times_by_circuit.sort_values(by='average_lap_seconds', ascending=False)

    # Convert to MM:SS.xxx format
    def format_time(seconds):
        minutes = int(seconds // 60)
        seconds = seconds % 60
        milliseconds = int((seconds - int(seconds)) * 1000)
        return f"{minutes}:{int(seconds):02}.{milliseconds:03}"

    average_lap_times_by_circuit['formatted_time'] = average_lap_times_by_circuit['average_lap_seconds'].apply(format_time)

    # Find the circuit with the longest and shortest average lap time
    slowest_circuit = average_lap_times_by_circuit.iloc[0]
    fastest_circuit = average_lap_times_by_circuit.iloc[-1]

    print("Circuit with the Longest Average Lap Time:")
    print(f"Circuit: {slowest_circuit['name']}")
    print(f"Average Lap Time: {slowest_circuit['formatted_time']}")

    print("\nCircuit with the Shortest Average Lap Time:")
    print(f"Circuit: {fastest_circuit['name']}")
    print(f"Average Lap Time: {fastest_circuit['formatted_time']}")

    # Scatter plot
    fig_scatter = px.scatter(
        average_lap_times_by_circuit,
        x='name',
        y='average_lap_seconds',
        size='average_lap_seconds',
        color='average_lap_seconds',
        hover_data={'formatted_time': True},
        title='Average Lap Times by Circuit (Scatter Plot)',
        labels={'average_lap_seconds': 'Average Lap Time (Seconds)', 'name': 'Circuit'},
        height=800,
        color_continuous_scale='Viridis'
    )

    fig_scatter.update_traces(marker=dict(opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
    fig_scatter.update_layout(
        xaxis=dict(title="Circuit", showgrid=False, tickangle=45),
        yaxis=dict(title="Average Lap Time (Seconds)", showgrid=True),
        title=dict(font=dict(size=24), x=0.5),
        template="plotly"
    )

    fig_scatter.write_image("average_lap_times_scatter.png")
    fig_scatter.show()

    # Table
    table = go.Table(
        header=dict(
            values=["<b>Circuit</b>", "<b>Average Lap Time</b>"],
            fill_color='paleturquoise',
            align='center',
            font=dict(size=14)
        ),
        cells=dict(
            values=[
                average_lap_times_by_circuit['name'],
                average_lap_times_by_circuit['formatted_time']
            ],
            fill_color='lavender',
            align='center',
            font=dict(size=12)
        )
    )

    fig_table = go.Figure(data=[table])
    fig_table.update_layout(
        title="Average Lap Times Table",
        height=600,
        width=800,
        template="plotly"
    )

    fig_table.write_image("average_lap_times_table.png")
    fig_table.show()

# Main script execution
if __name__ == "__main__":
    analyze_average_lap_times(
        '/Users/sid/Downloads/Formula1_Analysis/F1_dataset/lap_times.csv',
        '/Users/sid/Downloads/Formula1_Analysis/F1_dataset/races_cleaned_1.csv'
    )
