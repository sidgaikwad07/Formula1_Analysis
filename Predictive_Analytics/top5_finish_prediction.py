# Predictive Modeling: Top 5 Finish Prediction
# Author: Siddhant Gaikwad
# Date: 07 January 2025
# Description: This script predicts whether a driver will finish in the top 5 based on qualifying results, driver/team form, track characteristics, and pit stop data.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """
    Load and preprocess datasets for predictive modeling.

    Returns:
        pd.DataFrame: Merged datasets containing qualifying results, driver details,
                      circuit characteristics, race years, and pit stop details.
    """
    drivers = pd.read_csv('/Users/sid/Downloads/Formula1_Analysis/F1_dataset/drivers.csv')
    results = pd.read_csv('/Users/sid/Downloads/Formula1_Analysis/F1_dataset/results.csv')
    qualifying = pd.read_csv('/Users/sid/Downloads/Formula1_Analysis/F1_dataset/qualifying.csv')
    circuits = pd.read_csv('/Users/sid/Downloads/Formula1_Analysis/F1_dataset/circuits.csv')
    races = pd.read_csv('/Users/sid/Downloads/Formula1_Analysis/F1_dataset/races_cleaned_1.csv')
    pit_stops = pd.read_csv('/Users/sid/Downloads/Formula1_Analysis/F1_dataset/pit_stops.csv')

    # Merge datasets for modeling
    results = results.merge(races[['raceId', 'year', 'circuitId']], on='raceId', how='left')
    data = results.merge(
        qualifying[['raceId', 'driverId', 'position']].rename(columns={'position': 'qualifying_position'}),
        on=['raceId', 'driverId'],
        how='left'
    ).merge(
        drivers[['driverId', 'forename', 'surname']],
        on='driverId',
        how='left'
    ).merge(
        circuits[['circuitId', 'name', 'lat', 'lng', 'alt']],
        on='circuitId',
        how='left'
    ).merge(
        pit_stops[['raceId', 'driverId', 'milliseconds']].groupby(['raceId', 'driverId']).mean().reset_index().rename(
            columns={'milliseconds': 'avg_pit_stop_time'}),
        on=['raceId', 'driverId'],
        how='left'
    )

    # Ensure all columns are lowercase
    data.columns = map(str.lower, data.columns)
    return data

def preprocess_data(data):
    """
    Preprocess the data for binary classification (top 5 finish or not).

    Parameters:
        data (pd.DataFrame): Merged dataset containing race and qualifying details.

    Returns:
        tuple: Processed features and target variables for modeling.
    """
    # Drop rows with missing 'positionorder', 'qualifying_position', or 'avg_pit_stop_time'
    data = data.dropna(subset=['positionorder', 'qualifying_position', 'avg_pit_stop_time'])

    # Create a binary target variable: Top 5 finish
    data['is_top_5'] = (data['positionorder'] <= 5).astype(int)

    # Create features
    features = data[['qualifying_position', 'lat', 'lng', 'alt', 'year', 'avg_pit_stop_time']]
    features[['lat', 'lng', 'alt', 'avg_pit_stop_time']] = features[['lat', 'lng', 'alt', 'avg_pit_stop_time']].apply(lambda x: (x - x.mean()) / x.std())

    return features, data['is_top_5']

def train_model(features, target):
    """
    Train a binary classifier to predict top 5 finish.

    Parameters:
        features (pd.DataFrame): Feature matrix for modeling.
        target (pd.Series): Binary target variable (top 5 finish or not).

    Returns:
        tuple: Trained model, test features, test target, and predictions.
    """
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"Accuracy: {accuracy:.2f}")
    print(f"ROC AUC: {roc_auc:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, predictions))

    # Plot and save the ROC-AUC curve
    roc_display = RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title("ROC-AUC Curve")
    plt.tight_layout()
    plt.savefig("/Users/sid/Downloads/Formula1_Analysis/F1_dataset/roc_auc_curve.png")
    plt.show()

    return model, X_test, y_test, predictions

def feature_importance_plot(model, features):
    """
    Plot the feature importance of the model.

    Parameters:
        model (RandomForestClassifier): Trained Random Forest Model.
        features (pd.DataFrame): Feature matrix used in modeling.

    Returns:
        None
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
    Main function to execute the predictive modeling workflow.

    Returns:
        None
    """
    data = load_data()
    features, target = preprocess_data(data)
    model, X_test, y_test, predictions = train_model(features, target)

    # Plot feature importance
    feature_importance_plot(model, features)

if __name__ == "__main__":
    main()
