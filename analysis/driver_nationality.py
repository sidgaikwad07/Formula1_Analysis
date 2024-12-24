# Nationality of Drivers Analysis with Pie Chart and Table
# Author: Siddhant Gaikwad
# Date: 24 December 2024
# Description: This script analyzes the distribution of drivers by nationality, generates a pie chart, and includes a table of nationalities.

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Load the dataset
drivers = pd.read_csv("/Users/sid/Downloads/Formula1_Analysis/F1_dataset/drivers.csv")

# Convert 'dob' to datetime and calculate age
drivers['dob'] = pd.to_datetime(drivers['dob'], errors='coerce')
today = pd.Timestamp.now()
drivers['age'] = drivers['dob'].apply(lambda x: (today - x).days // 365 if pd.notnull(x) else None)

# Group by nationality and count drivers
driver_by_nationality = (
    drivers.groupby('nationality')['driverId']
    .count()
    .reset_index(name='Number of Drivers')
)

driver_by_nationality['Percentage'] = (driver_by_nationality['Number of Drivers'] / driver_by_nationality['Number of Drivers'].sum() * 100).round(2)

# Sort by the number of drivers for better visibility
driver_by_nationality = driver_by_nationality.sort_values(by='Number of Drivers', ascending=False)

# Create a subplot with a treemap and a detailed table
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{"type": "treemap"}, {"type": "table"}]],
    column_widths=[0.6, 0.4],
)

# Add treemap
fig.add_trace(
    go.Treemap(
        labels=driver_by_nationality['nationality'],
        parents=[""] * len(driver_by_nationality),
        values=driver_by_nationality['Number of Drivers'],
        textinfo="label+value+percent entry",
        marker=dict(colors=px.colors.sequential.Viridis)
    ),
    row=1, col=1
)

# Add detailed table
fig.add_trace(
    go.Table(
        header=dict(
            values=["<b>Nationality</b>", "<b>Number of Drivers</b>", "<b>Percentage (%)</b>"],
            fill_color="lightgrey",
            align="left",
            font=dict(size=12)
        ),
        cells=dict(
            values=[
                driver_by_nationality['nationality'],
                driver_by_nationality['Number of Drivers'],
                driver_by_nationality['Percentage']
            ],
            fill_color="white",
            align="left",
            font=dict(size=10)
        )
    ),
    row=1, col=2
)

# Update layout
fig.update_layout(
    title_text="Drivers by Nationality: Treemap and Detailed Table",
    template="plotly",
    height=800,
    width=1400  # Increased width for better spacing
)

# Save the chart as a PNG file
try:
    fig.write_image("drivers_nationality_treemap_table.png")
    print("Image saved as drivers_nationality_treemap_table.png")
except ValueError as e:
    print("Error saving image. Ensure 'kaleido' is installed correctly:", e)

# Show the figure
fig.show()