# Formula 1 FIA Championship Analysis
# Author: Siddhant Gaikwad
# Date: 23 December 2024
# Description: This script analyzes the number of FIA championships won by teams using the constructor standings dataset.

import pandas as pd
import plotly.express as px
import os

# Load datasets
constructor_standings = pd.read_csv("/Users/sid/Downloads/Formula1_data_analysis/F1_dataset/constructor_standings.csv")
constructors = pd.read_csv("/Users/sid/Downloads/Formula1_data_analysis/F1_dataset/constructors.csv")

# Analysis: Number of times the teams won the FIA championship
def fia_championship_analysis(constructor_standings, constructors):
    """
   Analyze the number of FIA championships won by each team.

    Parameters:
    constructor_standings (DataFrame): Dataset containing constructor standings.
    constructors (DataFrame): Dataset containing constructor details.

    Returns:
    DataFrame: A DataFrame with team names and their respective championship wins, 
    sorted in descending order.
    """
    championship_winners = constructor_standings[constructor_standings['position'] == 1]
    team_wins = championship_winners['constructorId'].value_counts().reset_index()
    team_wins.columns = ['constructorId', 'championship_wins']
    team_wins = team_wins.merge(constructors, on='constructorId', how='left')
    team_wins = team_wins.sort_values(by='championship_wins', ascending=False)
    return team_wins[['name', 'championship_wins']]

# Perform the analysis
team_wins = fia_championship_analysis(constructor_standings, constructors)

# Plotting using Plotly
fig = px.bar(
    team_wins,
    x='name',
    y='championship_wins',
    title='Number of FIA Championships won by the Teams',
    labels={'name': 'Team Name', 'championship_wins': 'Number of Championships'},
    color='championship_wins',
    color_continuous_scale='Viridis',
    width=1000,
    height=600
)
fig.update_layout(
    xaxis_title="Team Name",
    yaxis_title="Number of Championships",
    template="plotly",
    xaxis_tickangle=-45
)

# Save the plot as a PNG file
try:
    fig.write_image("fia_championships.png")
    print("Image saved as fia_championships.png")
except ValueError as e:
    print("Error saving image. Ensure 'kaleido' is installed correctly:", e)

# Display the plot
fig.show()
