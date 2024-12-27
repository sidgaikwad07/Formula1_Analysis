# Teams Pole-to-Win Efficiency Analysis
# Author: Siddhant Gaikwad
# Date: 27 December 2024
# Description: This script analyzes teams' pole-to-win efficiency in Formula 1.

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def analyze_teams_pole_to_win(results_file, qualifying_file, constructors_file):
    """
    Analyze pole-to-win efficiency for teams.

    Parameters:
        results_file (str): Path to the results CSV file.
        qualifying_file (str): Path to the qualifying CSV file.
        constructors_file (str): Path to the constructors CSV file.

    Returns:
        None
    """
    # Load datasets
    results_df = pd.read_csv(results_file)
    qualifying_df = pd.read_csv(qualifying_file)
    constructors_df = pd.read_csv(constructors_file)

    # Calculate team poles
    team_poles = qualifying_df[qualifying_df['position'] == 1].groupby('constructorId').size().reset_index(name='team_poles')

    # Calculate team wins
    team_wins = results_df[results_df['positionOrder'] == 1].groupby('constructorId').size().reset_index(name='team_wins')

    # Merge data and calculate pole-to-win ratio
    team_stats = team_poles.merge(team_wins, on='constructorId', how='left').fillna(0)
    team_stats['team_poles'] = team_stats['team_poles'].astype(int)
    team_stats['team_wins'] = team_stats['team_wins'].astype(int)
    team_stats['pole_to_win_ratio'] = (team_stats['team_wins'] / team_stats['team_poles']).round(2)

    # Add team names
    team_stats = team_stats.merge(constructors_df[['constructorId', 'name']], on='constructorId', how='left')

    # Filter and sort teams by pole-to-win ratio
    filtered_stats = team_stats[team_stats['team_poles'] > 10].sort_values('pole_to_win_ratio', ascending=False)

    # Create a combined plot
    fig_combined = make_subplots(
        rows=2, cols=1,
        row_heights=[0.6, 0.4],
        specs=[[{"type": "xy"}], [{"type": "table"}]],
        vertical_spacing=0.1,
        subplot_titles=("Teams' Pole-to-Win Efficiency", "Team Pole-to-Win Details")
    )

    # Add bar chart for pole-to-win ratio
    fig_combined.add_trace(
        go.Bar(
            x=filtered_stats['name'],
            y=filtered_stats['pole_to_win_ratio'],
            text=filtered_stats['pole_to_win_ratio'],
            textposition='outside',
            marker=dict(color='blue'),
            name="Pole-to-Win Ratio"
        ),
        row=1, col=1
    )

    # Add table for detailed statistics
    fig_combined.add_trace(
        go.Table(
            header=dict(
                values=["<b>Team</b>", "<b>Poles</b>", "<b>Wins</b>", "<b>Pole-to-Win Ratio</b>"],
                fill_color="lightblue",
                align="center",
                font=dict(size=14)
            ),
            cells=dict(
                values=[
                    filtered_stats['name'],
                    filtered_stats['team_poles'],
                    filtered_stats['team_wins'],
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
        title="Teams' Pole-to-Win Efficiency Analysis",
        xaxis_title="Team",
        yaxis_title="Pole-to-Win Ratio",
        height=900,
        showlegend=False
    )

    # Save and display the figure
    fig_combined.write_image("teams_pole_to_win_analysis.png")
    fig_combined.show()

# Main script execution
if __name__ == "__main__":
    analyze_teams_pole_to_win(
        results_file='/Users/sid/Downloads/Formula1_Analysis/F1_dataset/results.csv',
        qualifying_file='/Users/sid/Downloads/Formula1_Analysis/F1_dataset/qualifying.csv',
        constructors_file='/Users/sid/Downloads/Formula1_Analysis/F1_dataset/constructors.csv'
    )
