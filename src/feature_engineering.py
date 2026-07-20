"""Feature engineering utilities for smart energy prediction."""

import pandas as pd
from sklearn.preprocessing import StandardScaler


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert timestamp and extract useful time-based features."""
    df = df.copy()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["year"] = df["timestamp"].dt.year
        df["month"] = df["timestamp"].dt.month
        df["day"] = df["timestamp"].dt.day
        df["minute"] = df["timestamp"].dt.minute

    return df


def prepare_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare the final feature set used by the model."""
    df = create_time_features(df)

    if "timestamp" in df.columns:
        df = df.drop(columns=["timestamp"])

    return df


def scale_features(X_train, X_test=None, scaler=None):
    """Scale features using StandardScaler."""
    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        X_train_scaled = scaler.transform(X_train)

    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler

    return X_train_scaled, scaler
