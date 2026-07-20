"""Data loading and preprocessing utilities for smart energy prediction."""

import pandas as pd


def load_dataset(file_path: str) -> pd.DataFrame:
    """Load the energy consumption dataset."""
    return pd.read_csv(file_path)


def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """Return the number of missing values in each column."""
    return df.isnull().sum()


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows."""
    return df.drop_duplicates().copy()


def encode_household_id(df: pd.DataFrame) -> pd.DataFrame:
    """Encode household_id as numerical values."""
    df = df.copy()
    if "household_id" in df.columns and not pd.api.types.is_numeric_dtype(df["household_id"]):
        df["household_id"] = df["household_id"].astype("category").cat.codes
    return df


def prepare_features_and_target(df: pd.DataFrame, target_column: str = "consumption_kwh"):
    """Separate input features and target variable."""
    df = df.copy()
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y
