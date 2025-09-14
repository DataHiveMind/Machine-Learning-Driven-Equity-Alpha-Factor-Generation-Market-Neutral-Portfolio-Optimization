"""  
src/modeling/training.py

Purpose: Training utilities for machine learning models, including hyperparameter tuning and model saving/loading.
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer, mean_squared_error
from scipy.stats import uniform, randint

def hyperparameter_tuning(
    model: BaseEstimator, param_grid: dict, 
    X_train: pd.DataFrame, y_train: pd.Series,
    scoring: str = 'neg_mean_squared_error', 
    cv: int = 5, n_iter: int = 10, 
    random_search: bool = False, random_state: int = 42
) -> BaseEstimator:
    """
    Performs hyperparameter tuning using GridSearchCV or RandomizedSearchCV.

    Parameters:
    model (BaseEstimator): The machine learning model to tune.
    param_grid (dict): The hyperparameter grid for tuning.
    X_train (pd.DataFrame): Training features.
    y_train (pd.Series): Training target.
    scoring (str): Scoring metric for evaluation.
    cv (int): Number of cross-validation folds.
    n_iter (int): Number of parameter settings that are sampled in RandomizedSearchCV.
    random_search (bool): If True, use RandomizedSearchCV; otherwise, use GridSearchCV.
    random_state (int): Random seed for reproducibility.

    Returns:
    BaseEstimator: The best model found during tuning.
    """
    if random_search:
        search = RandomizedSearchCV(
            model, param_distributions=param_grid, n_iter=n_iter,
            scoring=scoring, cv=cv, random_state=random_state, n_jobs=-1
        )
    else:
        search = GridSearchCV(
            model, param_grid=param_grid,
            scoring=scoring, cv=cv, n_jobs=-1
        )
    
    search.fit(X_train, y_train)
    return search.best_estimator_

def save_model(model: BaseEstimator, filepath: str) -> None:
    """
    Saves the trained model to a file using joblib.

    Parameters:
    model (BaseEstimator): The trained machine learning model.
    filepath (str): The file path to save the model.
    """
    import joblib
    import os
    # Ensure the artifacts directory exists
    artifacts_dir = os.path.join("models", "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    # If filepath is not in artifacts_dir, move it there
    if not os.path.dirname(filepath) or os.path.dirname(filepath) != artifacts_dir:
        filename = os.path.basename(filepath)
        filepath = os.path.join(artifacts_dir, filename)
    joblib.dump(model, filepath)

def load_model(filepath: str) -> BaseEstimator:
    """
    Loads a trained model from a file using joblib.

    Parameters:
    filepath (str): The file path to load the model from.

    Returns:
    BaseEstimator: The loaded machine learning model.
    """
    import joblib
    return joblib.load(filepath)

if __name__ == "__main__":
    # Example usage
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split

    # Generate synthetic regression data
    X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define model and hyperparameter grid
    model = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }

    # Perform hyperparameter tuning
    best_model = hyperparameter_tuning(model, param_grid, pd.DataFrame(X_train), pd.Series(y_train))

    # Evaluate the best model
    y_pred = best_model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f"Validation RMSE: {rmse}")

    # Save the best model in models/artifacts
    save_model(best_model, "best_model.joblib")

    # Load the model back from models/artifacts
    loaded_model = load_model(os.path.join("models", "artifacts", "best_model.joblib"))
    loaded_y_pred = loaded_model.predict(X_val)
    loaded_rmse = np.sqrt(mean_squared_error(y_val, loaded_y_pred))
    print(f"Loaded Model Validation RMSE: {loaded_rmse}")