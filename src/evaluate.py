"""Model evaluation utilities."""

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


def evaluate_model(model, X_test, y_test):
    """Calculate MAE, RMSE, and R² score."""
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2,
    }


def print_evaluation_results(results):
    """Print evaluation metrics."""
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
