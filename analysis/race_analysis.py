# Race Analysis and Fastest Lap Records
# Author: Siddhant Gaikwad
# Date: 25 December 2024
# Description: This script analyzes race statistics and fastest lap records.

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load datasets
drivers_df = pd.read_csv('/Users/sid/Downloads/Formula1_Analysis/F1_dataset/drivers.csv')
results_df = pd.read_csv('/Users/sid/Downloads/Formula1_Analysis/F1_dataset/results.csv')
races_df = pd.read_csv('/Users/sid/Downloads/Formula1_Analysis/F1_dataset/races.csv')
lap_times_df = pd.read_csv('/Users/sid/Downloads/Formula1_Analysis/F1_dataset/lap_times.csv')

# Data Cleaning and Preprocessing
def clean_race_data(race_data):
    """
    Cleans and imputes missing values in race data.
    """
    race_data.replace('\\N', pd.NA, inplace=True)
    race_data.fillna({
        'fp1_date': 'Not Available',
        'time': '00:00:00'
    }, inplace=True)
    return race_data.drop(columns=['sprint_date', 'sprint_time'], errors='ignore')

# Clean the races data
races_df_cleaned = clean_race_data(races_df)

# Save the cleaned data for future analysis
races_df_cleaned.to_csv('/Users/sid/Downloads/Formula1_Analysis/F1_dataset/races_cleaned_1.csv', index=False)

# Analysis 1: Number of Races per Year
races_per_year = races_df_cleaned['year'].value_counts().sort_index()
fig1 = px.line(
    x=races_per_year.index,
    y=races_per_year.values,
    title='Number of Races per Year',
    labels={'x': 'Year', 'y': 'Number of Races'},
    markers=True
)
fig1.update_layout(template='plotly')
fig1.write_image("races_per_year.png")
fig1.show()

# Analysis 2: Top Circuits with Most Races
races_by_circuit = races_df_cleaned['name'].value_counts().head(10)
fig2 = px.bar(
    x=races_by_circuit.index,
    y=races_by_circuit.values,
    title='Top 10 Circuits with Most Races',
    labels={'x': 'Circuit Name', 'y': 'Number of Races'},
    color=races_by_circuit.values,
    color_continuous_scale='Blues'
)
fig2.update_layout(template='plotly')
fig2.write_image("top_circuits.png")
fig2.show()

# Analysis 3: Upcoming Races
upcoming_races = races_df_cleaned[races_df_cleaned['year'] >= 2024]
fig3 = px.bar(
    x=upcoming_races['name'],
    y=pd.to_datetime(upcoming_races['date']).dt.month,
    title='Upcoming Races and Their Scheduled Months',
    labels={'x': 'Race Name', 'y': 'Month'},
    color=pd.to_datetime(upcoming_races['date']).dt.month,
    color_continuous_scale='Greens'
)
fig3.update_layout(template='plotly')
fig3.write_image("upcoming_races.png")
fig3.show()

# Analysis 4: Fastest Lap Records by Circuit
lap_times_with_drivers = lap_times_df.merge(drivers_df[['driverId', 'forename', 'surname']], on='driverId', how='left')
lap_times_with_races = lap_times_with_drivers.merge(races_df_cleaned[['raceId', 'name']], on='raceId', how='left')

fastest_lap_per_circuit = lap_times_with_races.loc[lap_times_with_races.groupby('name')['milliseconds'].idxmin()]
fastest_lap_per_circuit['Driver'] = (
    fastest_lap_per_circuit['forename'] + ' ' + fastest_lap_per_circuit['surname']
)

# Convert milliseconds to a readable lap time format
def milliseconds_to_laptime(ms):
    minutes = ms // 60000
    seconds = (ms % 60000) // 1000
    milliseconds = ms % 1000
    return f"{minutes}:{seconds:02}.{milliseconds:03}"

fastest_lap_per_circuit['Lap Time'] = fastest_lap_per_circuit['milliseconds'].apply(milliseconds_to_laptime)

# Sort fastest lap records in descending order of time
fastest_lap_per_circuit = fastest_lap_per_circuit.sort_values(by='milliseconds', ascending=False)

# Create the bar chart
bar_chart = go.Figure(
    go.Bar(
        x=fastest_lap_per_circuit['milliseconds'],
        y=fastest_lap_per_circuit['name'],
        orientation='h',
        text=fastest_lap_per_circuit['Lap Time'],
        marker=dict(color='skyblue'),
        hovertemplate=(
            '<b>Circuit:</b> %{y}<br>'
            '<b>Driver:</b> %{customdata[0]}<br>'
            '<b>Lap Time:</b> %{text}<extra></extra>'
        ),
        customdata=fastest_lap_per_circuit[['Driver']]
    )
)
bar_chart.update_layout(
    title='Fastest Lap Record by Circuit',
    xaxis_title='Fastest Lap Time (Milliseconds)',
    yaxis_title='Circuit Name',
    yaxis=dict(categoryorder='total ascending'),
    height=800,
    template='plotly'
)

# Create the table
table = go.Figure(
    go.Table(
        header=dict(
            values=["<b>Circuit</b>", "<b>Driver</b>", "<b>Lap Time</b>"],
            fill_color='paleturquoise',
            align='center',
            font=dict(size=14)
        ),
        cells=dict(
            values=[
                fastest_lap_per_circuit['name'],
                fastest_lap_per_circuit['Driver'],
                fastest_lap_per_circuit['Lap Time']
            ],
            fill_color='lavender',
            align='center',
            font=dict(size=12)
        )
    )
)

# Save images
bar_chart.write_image("fastest_lap_bar_chart.png")
table.write_image("fastest_lap_table.png")

# Show plots
bar_chart.show()
table.show()
