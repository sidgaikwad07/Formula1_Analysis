# Constructor Rivalry Analysis
# Author: Siddhant Gaikwad
# Date: 24 December 2024
# Description: Analyze the biggest constructor rivalry in a given year using constructor standings, race, and constructor details datasets.

import pandas as pd
import plotly.express as px

def load_constructor_data():
    """
    Load and merge constructor-related datasets.

    Returns:
        DataFrame: Merged constructor standings, constructors, and races data.
    """
    constructor_standings = pd.read_csv('/Users/sid/Downloads/Formula1_Analysis/F1_dataset/constructor_standings.csv')
    constructors = pd.read_csv('/Users/sid/Downloads/Formula1_Analysis/F1_dataset/constructors.csv')
    races = pd.read_csv('/Users/sid/Downloads/Formula1_Analysis/F1_dataset/races_cleaned_1.csv')

    # Merge datasets
    constructor_position = constructor_standings.merge(
        constructors.rename(columns={'name': 'constructor_name'}), on='constructorId', how='left'
    ).merge(
        races.rename(columns={'name': 'race_name'}), on='raceId', how='left'
    )

    # Standardize column names and add year column
    constructor_position.columns = map(str.lower, constructor_position.columns)
    constructor_position['year'] = pd.to_datetime(constructor_position['date']).dt.year

    return constructor_position

def constructor_rivalry_analysis(year, constructor_position):
    """
    Analyze the biggest constructor rivalry in a given year and visualize the results.

    Parameters:
        year (int): The year to analyze.
        constructor_position (DataFrame): Comprehensive DataFrame with constructor standings and race data.
    """
    # Step 1: Filter for the given year and get top 5 constructors based on points
    constructor_competition = constructor_position[constructor_position['year'] == year] \
        .groupby(['constructor_name', 'year'])['points'].max() \
        .sort_values(ascending=False).reset_index().head(5)

    # Rivalry Intensity
    if constructor_competition.iloc[0, 2] - constructor_competition.iloc[1, 2] <= 10:
        print('\033[1m' + 'BIGGEST CONSTRUCTOR RIVALRY IN THE MAKING!')
    elif constructor_competition.iloc[0, 2] - constructor_competition.iloc[1, 2] <= 20:
        print('\033[1m' + 'Spicy Constructor Rivalry!')
    elif constructor_competition.iloc[0, 2] - constructor_competition.iloc[1, 2] < 30:
        print('\033[1m' + 'Moderate Constructor Rivalry!')
    else:
        print('\033[1m' + 'Dominant Constructor Lead - Less Competition')

    # Step 2: Prepare visualization data for top 5 constructors
    constructor_top_five = constructor_position[constructor_position['year'] == year] \
        .groupby(['constructor_name'])[['points', 'wins']].max() \
        .sort_values('points', ascending=False).head(5).reset_index()

    # Step 3: Create the bar plot for visualization
    fig = px.bar(
        constructor_top_five,
        x='constructor_name',
        y='points',
        hover_data=['wins'],
        color='points',
        height=400,
        color_continuous_scale='viridis',
        title=f"Constructor Rivalries in {year}"
    )

    fig.update_layout(
        xaxis=dict(title="Constructor Name", showgrid=False),
        yaxis=dict(title="Points", showgrid=False),
        template="plotly"
    )

    fig.update_traces(
        textfont_size=20,
        marker=dict(line=dict(color='#000000', width=2))
    )

    # Save the plot as a PNG file
    try:
        fig.write_image(f"constructor_rivalry_{year}.png")
        print(f"Image saved as constructor_rivalry_{year}.png")
    except ValueError as e:
        print("Error saving image. Ensure 'kaleido' is installed correctly:", e)

    print('----------------------------------')
    fig.show()

if __name__ == "__main__":
    # Load data
    constructor_position = load_constructor_data()

    # Analyze constructor rivalry for a specific year
    year_to_analyze = 2021
    constructor_rivalry_analysis(year_to_analyze, constructor_position)
