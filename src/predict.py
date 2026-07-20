"""Prediction utilities for the smart energy application."""

import joblib
import pandas as pd
from feature_engineering import prepare_model_features
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def load_model_and_scaler(
    model_path=BASE_DIR / "models" / "Smart_Energy_Model.pkl",
    scaler_path=BASE_DIR / "models" / "scaler.pkl",
):

    """Load the trained model and scaler."""
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def make_prediction(input_data, model, scaler):
    """Prepare input data and predict energy consumption."""
    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])

    input_data = prepare_model_features(input_data)
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    return float(prediction[0])
