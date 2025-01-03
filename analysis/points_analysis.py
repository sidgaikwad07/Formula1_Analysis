# Maximum Points Analysis
# Author: Siddhant Gaikwad
# Date: 26 December 2024
# Description: This script analyzes the maximum number of points earned by drivers and teams.

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def load_and_prepare_data():
    """
    Loads and merges necessary datasets for analysis.
    Returns:
        results_with_names (DataFrame): Merged DataFrame with race, driver, and team information.
    """
    drivers_df = pd.read_csv('/Users/sid/Downloads/Formula1_Analysis/F1_dataset/drivers.csv')
    results_df = pd.read_csv('/Users/sid/Downloads/Formula1_Analysis/F1_dataset/results.csv')
    race_df = pd.read_csv('/Users/sid/Downloads/Formula1_Analysis/F1_dataset/races_cleaned_1.csv')
    constructors_df = pd.read_csv('/Users/sid/Downloads/Formula1_Analysis/F1_dataset/constructors.csv')

    # Merge datasets
    results_with_year = results_df.merge(race_df[['raceId', 'year']], on='raceId', how='left')
    results_with_names = results_with_year.merge(
        drivers_df[['driverId', 'forename', 'surname']], on='driverId', how='left'
    ).merge(
        constructors_df[['constructorId', 'name']], on='constructorId', how='left'
    )
    return results_with_names

def analyze_driver_points(results_with_names):
    """
    Analyze and visualize the maximum points earned by drivers.
    """
    # Aggregate total points by drivers
    driver_points = results_with_names.groupby(['driverId', 'forename', 'surname'])['points'].sum().reset_index()
    driver_points = driver_points.sort_values(by='points', ascending=False)

    # Prepare data for visualization
    top_drivers = driver_points.head(10)
    top_drivers['Driver'] = top_drivers['forename'] + ' ' + top_drivers['surname']

    # Create table
    driver_table = go.Figure(data=[
        go.Table(
            header=dict(
                values=["<b>Driver</b>", "<b>Total Points</b>"],
                fill_color='paleturquoise',
                align='center',
                font=dict(size=14)
            ),
            cells=dict(
                values=[top_drivers['Driver'], top_drivers['points']],
                fill_color='lavender',
                align='center',
                font=dict(size=12)
            )
        )
    ])
    driver_table.update_layout(title="Driver Points Table")

    # Create bar chart
    fig_driver = px.bar(
        top_drivers,
        x='points',
        y='Driver',
        orientation='h',
        title='Top 10 Drivers by Total Points',
        labels={'points': 'Total Points', 'y': 'Driver Name'},
        height=700
    )
    fig_driver.update_layout(template="plotly")

    # Save and show separately
    fig_driver.write_image("driver_points_chart.png")
    driver_table.write_image("driver_points_table.png")
    fig_driver.show()
    driver_table.show()

def analyze_team_points(results_with_names):
    """
    Analyze and visualize the maximum points earned by teams.
    """
    # Aggregate total points by teams
    team_points = results_with_names.groupby(['constructorId', 'name'])['points'].sum().reset_index()
    team_points = team_points.sort_values(by='points', ascending=False)

    # Prepare data for visualization
    top_teams = team_points.head(10)

    # Create table
    team_table = go.Figure(data=[
        go.Table(
            header=dict(
                values=["<b>Team</b>", "<b>Total Points</b>"],
                fill_color='paleturquoise',
                align='center',
                font=dict(size=14)
            ),
            cells=dict(
                values=[top_teams['name'], top_teams['points']],
                fill_color='lavender',
                align='center',
                font=dict(size=12)
            )
        )
    ])
    team_table.update_layout(title="Team Points Table")

    # Create bar chart
    fig_team = px.bar(
        top_teams,
        x='points',
        y='name',
        orientation='h',
        title='Top 10 Teams by Total Points',
        labels={'points': 'Total Points', 'y': 'Team Name'},
        height=600
    )
    fig_team.update_layout(template="plotly")

    # Save and show separately
    fig_team.write_image("team_points_chart.png")
    team_table.write_image("team_points_table.png")
    fig_team.show()
    team_table.show()

# Main script execution
if __name__ == "__main__":
    results_with_names = load_and_prepare_data()
    analyze_driver_points(results_with_names)
    analyze_team_points(results_with_names)
