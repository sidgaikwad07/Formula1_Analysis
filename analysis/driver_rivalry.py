# Formula 1 Driver Rivalry Analysis
# Author: Siddhant Gaikwad
# Date: 24 December 2024
# Description: This script analyzes the biggest driver rivalries in a given year using driver standings, results, and circuit data.

import pandas as pd
import plotly.express as px

def rivalry_analysis(year, driver_position):
    """
    Analyze the biggest rivalry in a given year and visualize the results.

    Parameters:
        year (int): The year to analyze.
        driver_position (DataFrame): Comprehensive DataFrame with driver standings, 
        results, and circuit data.
    """
    # Step 1: Filter for the given year and get top 5 drivers based on points
    competition = driver_position[driver_position['year'] == year] \
        .groupby(['surname', 'year'])['points'].max() \
        .sort_values(ascending=False).reset_index().head(5)

    # Check for rivalry intensity
    if competition.iloc[0, 2] - competition.iloc[1, 2] <= 10:
        print('\033[1m' + 'BIGGEST RIVALRY IN THE MAKING!')
    elif competition.iloc[0, 2] - competition.iloc[1, 2] <= 20:
        print('\033[1m' + 'Spicy Rivalry!')
    elif competition.iloc[0, 2] - competition.iloc[1, 2] < 30:
        print('\033[1m' + 'Moderate Rivalry!')
    else:
        print('\033[1m' + 'Dominant Lead - Less Competition')

    # Step 2: Prepare visualization data for top 5 drivers
    top_five = driver_position[driver_position['year'] == year] \
        .groupby(['surname'])[['points', 'wins']].max() \
        .sort_values('points', ascending=False).head(5).reset_index()

    # Step 3: Create the bar plot for visualization
    fig = px.bar(
        top_five,
        x='surname',
        y='points',
        hover_data=['wins'],
        color='points',
        height=400,
        color_continuous_scale='turbo',
        title=f"Biggest Rivalry in {year}"
    )

    fig.update_layout(
        xaxis=dict(title="Driver Surname", showgrid=False),
        yaxis=dict(title="Points", showgrid=False),
        template="plotly"
    )

    fig.update_traces(
        textfont_size=20,
        marker=dict(line=dict(color='#000000', width=2))
    )

    # Save the plot as a PNG file
    try:
        fig.write_image(f"driver_rivalry_{year}.png")
        print(f"Image saved as driver_rivalry_{year}.png")
    except ValueError as e:
        print("Error saving image. Ensure 'kaleido' is installed correctly:", e)

    print('----------------------------------')
    fig.show()

driver_standings = pd.read_csv('/Users/sid/Downloads/Formula1_Analysis/F1_dataset/driver_standings.csv')
drivers = pd.read_csv('/Users/sid/Downloads/Formula1_Analysis/F1_dataset/drivers.csv')  
results = pd.read_csv('/Users/sid/Downloads/Formula1_Analysis/F1_dataset/results.csv')  
circuits = pd.read_csv('/Users/sid/Downloads/Formula1_Analysis/F1_dataset/circuits.csv')  
races = pd.read_csv('/Users/sid/Downloads/Formula1_Analysis/F1_dataset/races_cleaned_1.csv') 

# Merge datasets to create a comprehensive `driver_position` DataFrame
driver_position = driver_standings.merge(
    drivers, left_on='driverId', right_on='driverId', how='left'
).merge(
    results, on=['raceId', 'driverId'], how='left'
).merge(
    races, on='raceId', how='left'
).merge(
    circuits, left_on='circuitId', right_on='circuitId', how='left'
)

# Ensure all columns are lowercase for consistency
driver_position.columns = map(str.lower, driver_position.columns)

# Convert dates and relevant fields
driver_position['year'] = pd.to_datetime(driver_position['date']).dt.year
driver_position['date'] = pd.to_datetime(driver_position['date'])

# Ensure the 'points' column exists
if 'points' not in driver_position.columns:
    driver_position['points'] = driver_position['points_x']  # Use points from driver_standings or results if necessary

# Example usage of the rivalry analysis function
rivalry_analysis(2021, driver_position)
