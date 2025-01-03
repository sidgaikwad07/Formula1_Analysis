# Fastest Pit Stops Analysis
# Author: Siddhant Gaikwad
# Date: 25 December 2024
# Description: This script analyzes and visualizes the fastest pit stops for Formula 1 teams. 

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load datasets
drivers_df = pd.read_csv('/Users/sid/Downloads/Formula1_Analysis/F1_dataset/drivers.csv')
results_df = pd.read_csv('/Users/sid/Downloads/Formula1_Analysis/F1_dataset/results.csv')
race_df = pd.read_csv('/Users/sid/Downloads/Formula1_Analysis/F1_dataset/races_cleaned_1.csv')
pit_stops_df = pd.read_csv('/Users/sid/Downloads/Formula1_Analysis/F1_dataset/pit_stops.csv')
constructors_df = pd.read_csv('/Users/sid/Downloads/Formula1_Analysis/F1_dataset/constructors.csv')

# Merge pit stops with results to link drivers to team
pit_stops_with_teams = pit_stops_df.merge(results_df[['raceId','driverId','constructorId']], on=['raceId','driverId'], how='left')

# Merge with races to get the year
pit_stops_with_years = pit_stops_with_teams.merge(race_df[['raceId','year']], on='raceId', how='left')

# Merge with the constructors to get the team names
pit_stops_with_teams = pit_stops_with_years.merge(constructors_df[['constructorId', 'name']], on='constructorId', how='left')
fastest_pitstops = pit_stops_with_teams.loc[pit_stops_with_teams.groupby(['year', 'name'])['milliseconds'].idxmin()]
fastest_pitstops['seconds'] = fastest_pitstops['milliseconds'] / 1000

# Filter for years 2012 to 2024
filtered_fastest_pitstops = fastest_pitstops[(fastest_pitstops['year'] >= 2012) & (fastest_pitstops['year'] <= 2024)]

print("Fastest Pitstops for each team by Year")
print(filtered_fastest_pitstops[['year', 'name', 'seconds']])

# Create a table for fastest pit stops
fastest_pitstop_table = go.Figure(data=[
    go.Table(
        header=dict(
            values=["<b>Year</b>", "<b>Team</b>", "<b>Fastest Pit Stop (s)</b>"],
            fill_color='paleturquoise',
            align='center',
            font=dict(size=14)
        ),
        cells=dict(
            values=[
                filtered_fastest_pitstops['year'],
                filtered_fastest_pitstops['name'],
                filtered_fastest_pitstops['seconds'].round(3)
            ],
            fill_color='lavender',
            align='center',
            font=dict(size=12)
        )
    )
])
fastest_pitstop_table.update_layout(title="Fastest Pit Stops Table (2012-2024)", height=600, width=800)

# Heatmap data
heatmap_data = filtered_fastest_pitstops.pivot_table(values='seconds', index='name', columns='year', aggfunc='min')

# Plotting the heatmap
fig_heatmap = px.imshow(
    heatmap_data, 
    color_continuous_scale='Viridis',
    title='Fastest Pitstops Heatmap (seconds)',
    labels={'x': 'Year', 'y': 'Team', 'color': 'Pitstop Duration (s)'},
    height=800
)
fig_heatmap.update_layout(
    xaxis_title='Year', 
    yaxis_title='Team', 
    coloraxis_colorbar=dict(title="Pitstop Duration (s)"), 
    template="plotly"
)

# Save images
fig_heatmap.write_image("fastest_pitstops_heatmap.png")
fastest_pitstop_table.write_image("fastest_pitstops_table.png")

# Show plots
fig_heatmap.show()
fastest_pitstop_table.show()
