# Pit Stop Strategy Optimization
# Author: Siddhant Gaikwad
# Date: 02 January 2025
# Description: This script predicts the optimal pit stop strategy using machine learning based on historical data.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


def load_pit_stop_data():
    """
    Load and preprocess data for pit stop strategy optimization.

    Returns:
        pd.DataFrame: Merged dataset containing pit stop details and race information.
    """
    # Load datasets
    pit_stops = pd.read_csv('/Users/sid/Downloads/Formula1_Analysis/F1_dataset/pit_stops.csv')
    results = pd.read_csv('/Users/sid/Downloads/Formula1_Analysis/F1_dataset/results.csv')
    races = pd.read_csv('/Users/sid/Downloads/Formula1_Analysis/F1_dataset/races_cleaned_1.csv')
    circuits = pd.read_csv('/Users/sid/Downloads/Formula1_Analysis/F1_dataset/circuits.csv')

    # Check and print columns for debugging
    print("Circuits columns:", circuits.columns)
    print("Races columns:", races.columns)

    # Merge datasets
    pit_stop_data = pit_stops.merge(
        results[['raceId', 'driverId', 'constructorId', 'positionOrder', 'points']], on=['raceId', 'driverId'], how='left'
    ).merge(
        races[['raceId', 'year', 'name', 'date', 'circuitId']], on='raceId', how='left'
    )

    # Check if 'circuitId' exists in races before merging with circuits
    if 'circuitId' in races.columns and 'circuitId' in circuits.columns:
        pit_stop_data = pit_stop_data.merge(
            circuits[['circuitId', 'name', 'lat', 'lng', 'alt']], on='circuitId', how='left'
        )
    else:
        print("circuitId missing in races or circuits; skipping this merge.")
        pit_stop_data['lat'] = None
        pit_stop_data['lng'] = None
        pit_stop_data['alt'] = None

    # Clean data
    pit_stop_data.dropna(subset=['milliseconds', 'positionOrder'], inplace=True)
    pit_stop_data['pit_duration'] = pit_stop_data['milliseconds'] / 1000

    return pit_stop_data


def preprocess_pit_stop_data(data):
    """
    Preprocess the dataset for machine learning.

    Parameters:
        data (pd.DataFrame): Raw pit stop data.

    Returns:
        tuple: Features and target variable.
    """
    # Add target variable: Optimal pit stop strategy (1 = Top 5 Finish, 0 = Not Top 5)
    data['optimal_strategy'] = data['positionOrder'].apply(lambda x: 1 if x <= 5 else 0)

    # Features and target
    features = data[['pit_duration', 'lat', 'lng', 'alt', 'year']]
    target = data['optimal_strategy']

    # Normalize numerical features
    features[['pit_duration', 'lat', 'lng', 'alt']] = features[['pit_duration', 'lat', 'lng', 'alt']].apply(
        lambda x: (x - x.mean()) / x.std()
    )

    return features, target


def train_pit_stop_model(features, target):
    """
    Train a machine learning model to predict optimal pit stop strategies.

    Parameters:
        features (pd.DataFrame): Feature matrix.
        target (pd.Series): Target variable.

    Returns:
        RandomForestClassifier: Trained model.
    """
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Train a Random Forest model with class weights
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    # Evaluate the model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, predictions))

    return model


def feature_importance_plot(model, features):
    """
    Plot the feature importance of the model and save it as a PNG file.

    Parameters:
        model (RandomForestClassifier): Trained Random Forest Model.
        features (pd.DataFrame): Feature matrix used in modeling.

    Returns:
        None.
    """
    importance = model.feature_importances_
    feature_names = features.columns
    sns.barplot(x=importance, y=feature_names)
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig("/Users/sid/Downloads/Formula1_Analysis/F1_dataset/feature_importance.png")
    plt.show()


def main():
    """
    Main function to execute the pit stop strategy optimization workflow.

    Returns:
        None
    """
    # Load and preprocess data
    data = load_pit_stop_data()
    features, target = preprocess_pit_stop_data(data)

    # Train model
    model = train_pit_stop_model(features, target)

    # Plot feature importance
    feature_importance_plot(model, features)


if __name__ == "__main__":
    main()
