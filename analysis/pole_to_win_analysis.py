# Pole Position to Race Win Conversion Analysis
# Author: Siddhant Gaikwad
# Date: 27 December 2024
# Description: Analyzes which drivers converted pole positions to race wins.


import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def analyze_pole_to_win_conversion(results_file, qualifying_file, drivers_file):
    """
    Analyzes which drivers could convert pole positions to race wins.
    """
    # Load datasets
    results_df = pd.read_csv(results_file)
    qualifying_df = pd.read_csv(qualifying_file)
    drivers_df = pd.read_csv(drivers_file)

    # Calculate grid poles (pole positions)
    grid_poles = qualifying_df[qualifying_df['position'] == 1].groupby('driverId').size().reset_index(name='grid_poles')

    # Calculate race wins
    race_wins = results_df[results_df['positionOrder'] == 1].groupby('driverId').size().reset_index(name='race_wins')

    # Merge and calculate pole-to-win ratio
    driver_stats = grid_poles.merge(race_wins, on='driverId', how='left').fillna(0)
    driver_stats['pole_to_win_ratio'] = (driver_stats['race_wins'] / driver_stats['grid_poles']).round(2)

    # Add driver names
    driver_stats = driver_stats.merge(drivers_df[['driverId', 'forename', 'surname']], on='driverId', how='left')
    driver_stats['driver_name'] = driver_stats['forename'] + ' ' + driver_stats['surname']

    # Filter for drivers with more than 10 poles
    filtered_stats = driver_stats[driver_stats['grid_poles'] > 10].sort_values('pole_to_win_ratio', ascending=False)

    # Create combined plot
    fig_combined = make_subplots(
        rows=2, cols=1,
        row_heights=[0.6, 0.4],
        specs=[[{"type": "xy"}], [{"type": "table"}]],
        vertical_spacing=0.1,
        subplot_titles=("Pole to Win Conversion Ratio", "Driver Pole to Win Details")
    )

    # Add bar chart
    fig_combined.add_trace(
        go.Bar(
            x=filtered_stats['driver_name'],
            y=filtered_stats['pole_to_win_ratio'],
            text=filtered_stats['pole_to_win_ratio'],
            textposition='outside',
            marker=dict(color='blue'),
            name="Pole to Win Ratio"
        ),
        row=1, col=1
    )

    # Add table
    fig_combined.add_trace(
        go.Table(
            header=dict(
                values=["<b>Driver</b>", "<b>Grid Poles</b>", "<b>Race Wins</b>", "<b>Pole to Win Ratio</b>"],
                fill_color="lightblue",
                align="center",
                font=dict(size=14)
            ),
            cells=dict(
                values=[
                    filtered_stats['driver_name'],
                    filtered_stats['grid_poles'],
                    filtered_stats['race_wins'],
                    filtered_stats['pole_to_win_ratio']
                ],
                fill_color="white",
                align="center",
                font=dict(size=12)
            )
        ),
        row=2, col=1
    )

    # Update layout
    fig_combined.update_layout(
        title="Pole to Win Conversion Analysis",
        xaxis_title="Driver",
        yaxis_title="Pole to Win Ratio",
        height=900,
        showlegend=False
    )

    # Save and display
    fig_combined.write_image("pole_to_win_analysis.png")
    fig_combined.show()

# Execute the function
analyze_pole_to_win_conversion(
    results_file='/Users/sid/Downloads/Formula1_Analysis/F1_dataset/results.csv',
    qualifying_file='/Users/sid/Downloads/Formula1_Analysis/F1_dataset/qualifying.csv',
    drivers_file='/Users/sid/Downloads/Formula1_Analysis/F1_dataset/drivers.csv'
)
