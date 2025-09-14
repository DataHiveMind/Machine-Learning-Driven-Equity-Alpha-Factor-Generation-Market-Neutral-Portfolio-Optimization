"""
src/data/preprocess.py

Purpose: Data preprocessing utilities for the project.
"""
import pandas as pd
import numpy as np
import scipy.stats as stats

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the input DataFrame by handling missing values and normalizing numerical columns, and outliners if any.

    Parameters:
    df (pd.DataFrame): Input DataFrame to preprocess.

    Returns:
    pd.DataFrame: Preprocessed DataFrame.
    """
    # Handle missing values by filling with the mean of numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # Normalize numerical columns to a 0-1 range
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    df[numerical_cols] = (df[numerical_cols] - df[numerical_cols].min()) / (df[numerical_cols].max() - df[numerical_cols].min())

    # Remove outliers using Z-score method
    df = df[(np.abs(stats.zscore(df[numerical_cols])) < 3).all(axis=1)]

    # Save processed data to data/processed folder
    import os
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    processed_path = os.path.join(processed_dir, "processed_data.csv")
    df.to_csv(processed_path, index=False)

    return df


if __name__ == "__main__":
    # Example usage
    example_data = {
        'A': [1, 2, np.nan, 4, 1000],
        'B': [5, np.nan, 7, 8, 9],
        'C': ['foo', 'bar', 'baz', 'qux', 'quux']
    }
    df = pd.DataFrame(example_data)
    print("Original DataFrame:")
    print(df)

    preprocessed_df = preprocess_data(df)
    print("\nPreprocessed DataFrame:")
    print(preprocessed_df)

