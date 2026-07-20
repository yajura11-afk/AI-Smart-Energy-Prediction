"""Train and save machine learning models."""

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from data_preprocessing import load_dataset, remove_duplicates, encode_household_id
from feature_engineering import prepare_model_features, scale_features


def train_models(X_train, y_train):
    """Train all project models."""
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42),
        "XGBoost": XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
            objective="reg:squarederror"
        ),
    }

    for model in models.values():
        model.fit(X_train, y_train)

    return models


def train_and_save_final_model(
    dataset_path="data/raw/energy_consumption.csv",
     model_path="models/Smart_Energy_Model.pkl",
    scaler_path="models/scaler.pkl",
):
    """Train the final XGBoost model and save both model and scaler."""
    df = load_dataset(dataset_path)
    df = remove_duplicates(df)
    df = encode_household_id(df)

    X, y = df.drop(columns=["consumption_kwh"]), df["consumption_kwh"]
    X = prepare_model_features(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    final_model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        objective="reg:squarederror"
    )
    final_model.fit(X_train_scaled, y_train)

    joblib.dump(final_model, model_path)
    joblib.dump(scaler, scaler_path)

    return final_model, scaler, X_test_scaled, y_test


if __name__ == "__main__":
    train_and_save_final_model()
