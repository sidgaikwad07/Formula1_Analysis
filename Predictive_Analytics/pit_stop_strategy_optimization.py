# Pit Stop Strategy Optimization with Imbalance Handling and Advanced Tuning
# Author: Siddhant Gaikwad
# Date: 02 January 2025
# Description: This script predicts the optimal pit stop strategy using machine learning with advanced techniques to handle class imbalance and improve model performance.

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
import shap

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

def train_optimized_lgbm(features, target):
    """
    Train and tune a LightGBM model for pit stop strategy optimization.

    Parameters:
        features (pd.DataFrame): Feature matrix.
        target (pd.Series): Target variable.

    Returns:
        LGBMClassifier: Trained model.
    """
    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    features_resampled, target_resampled = smote.fit_resample(features, target)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(features_resampled, target_resampled, test_size=0.2, random_state=42)

    # Hyperparameter tuning
    param_grid = {
        'num_leaves': [31, 50],
        'learning_rate': [0.01, 0.1],
        'n_estimators': [100, 200],
        'boosting_type': ['gbdt', 'dart']
    }

    lgbm = LGBMClassifier(random_state=42, class_weight='balanced')
    grid_search = GridSearchCV(estimator=lgbm, param_grid=param_grid, scoring='accuracy', cv=3, verbose=1)
    grid_search.fit(X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_

    # Evaluate the model
    predictions = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Tuned LightGBM Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, predictions))

    return best_model, X_test, y_test

def feature_importance_plot(model, features):
    """
    Plot the feature importance of the model and save it as a PNG file.

    Parameters:
        model (LGBMClassifier): Trained LightGBM Model.
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
    plt.savefig("/Users/sid/Downloads/Formula1_Analysis/F1_dataset/feature_importance_optimized.png")
    plt.show()

def explain_model(model, X_test):
    """
    Use SHAP to explain the model's predictions.

    Parameters:
        model (LGBMClassifier): Trained LightGBM Model.
        X_test (pd.DataFrame): Test features.

    Returns:
        None.
    """
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig("/Users/sid/Downloads/Formula1_Analysis/F1_dataset/shap_summary_plot.png")
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

    # Train and tune model
    model, X_test, y_test = train_optimized_lgbm(features, target)

    # Plot feature importance
    feature_importance_plot(model, features)

    # Explain model predictions
    explain_model(model, X_test)

if __name__ == "__main__":
    main()
