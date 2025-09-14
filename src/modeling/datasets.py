"""
src/modeling/datasets.py

Purpose: Use to train/test split datasets for modeling, and purged CV.
"""
import pandas as pd
import numpy as np
from typing import Any

from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit

def train_test_split_time_series(df: pd.DataFrame, target_column: str, 
                                 test_size: float = 0.2, random_state: int = 42)-> tuple[Any, Any, Any, Any]:
    """
    Splits the DataFrame into training and testing sets based on time series data.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing features and target.
    target_column (str): The name of the target column.
    test_size (float): The proportion of the dataset to include in the test split.
    random_state (int): Random seed for reproducibility.

    Returns:
    X_train, X_test, y_train, y_test: Split datasets.
    """
    # Sort by date if a date column exists
    if 'date' in df.columns:
        df = df.sort_values(by='date')

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

    return X_train, X_test, y_train, y_test

from typing import Callable, Generator, Tuple
import numpy as np

def purged_time_series_cv(
    df: pd.DataFrame, n_splits: int = 5, purge_period: int = 1
) -> Callable[[pd.DataFrame], Generator[Tuple[np.ndarray, np.ndarray], None, None]]:
    """
    Creates a purged time series cross-validation splitter.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing features and target.
    n_splits (int): Number of splits for cross-validation.
    purge_period (int): Number of periods to purge between training and testing sets.

    Returns:
    Callable: A function that yields train/test indices with purging.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Custom generator to yield train/test indices with purging
    def purged_split(X: pd.DataFrame) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        for train_index, test_index in tscv.split(X):
            if purge_period > 0:
                # Purge the last 'purge_period' samples from the training set
                train_index = train_index[train_index < test_index[0] - purge_period]
            yield train_index, test_index

    return purged_split

if __name__ == "__main__":
    # Example usage
    data = {
        'date': pd.date_range(start='2022-01-01', periods=100, freq='D'),
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'target': np.random.randint(0, 2, size=100)
    }
    df = pd.DataFrame(data)

    X_train, X_test, y_train, y_test = train_test_split_time_series(df, target_column='target')
    print("Train features shape:", X_train.shape)
    print("Test features shape:", X_test.shape)

    purged_cv = purged_time_series_cv(df, n_splits=5, purge_period=2)
    for fold, (train_idx, test_idx) in enumerate(purged_cv(df)):
        print(f"Fold {fold}: Train indices {train_idx}, Test indices {test_idx}")
