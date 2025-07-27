import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def cross_sectional_zscore(df: pd.DataFrame, group_col: str, feature_cols: list) -> pd.DataFrame:
    """
    Z-score normalization across all assets at each time point.
    """
    for col in feature_cols:
        df[f'{col}_zscore'] = df.groupby(group_col)[col].transform(lambda x: (x - x.mean()) / x.std())
    return df

def cross_sectional_minmax(df: pd.DataFrame, group_col: str, feature_cols: list) -> pd.DataFrame:
    """
    Min-max scaling across all assets at each time point.
    """
    for col in feature_cols:
        df[f'{col}_minmax'] = df.groupby(group_col)[col].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
    return df

def time_series_zscore(df: pd.DataFrame, id_col: str, feature_cols: list) -> pd.DataFrame:
    """
    Z-score normalization over time for each asset.
    """
    for col in feature_cols:
        df[f'{col}_ts_zscore'] = df.groupby(id_col)[col].transform(lambda x: (x - x.mean()) / x.std())
    return df

def winsorize_series(series: pd.Series, limits: float = 0.01) -> pd.Series:
    """
    Winsorize a pandas Series to limit the impact of outliers.
    """
    lower = series.quantile(limits)
    upper = series.quantile(1 - limits)
    return series.clip(lower, upper)

def winsorize_features(df: pd.DataFrame, feature_cols: list, limits: float = 0.01) -> pd.DataFrame:
    """
    Winsorize multiple features in a DataFrame.
    """
    for col in feature_cols:
        df[col] = winsorize_series(df[col], limits)
    return df

def impute_missing(df: pd.DataFrame, feature_cols: list, strategy: str = 'mean') -> pd.DataFrame:
    """
    Impute missing values using a specified strategy (mean, median, most_frequent).
    """
    imputer = SimpleImputer(strategy=strategy)
    df[feature_cols] = imputer.fit_transform(df[feature_cols])
    return df

def dynamic_neutralization(df: pd.DataFrame, feature_col: str, neutralize_col: str, group_col: str) -> pd.DataFrame:
    """
    Neutralize a feature against a market or sector effect within each group (e.g., date).
    """
    def neutralize(x):
        y = x[feature_col]
        z = x[neutralize_col]
        beta = np.cov(y, z)[0, 1] / np.var(z) if np.var(z) > 0 else 0
        return y - beta * z
    df[f'{feature_col}_neutralized'] = df.groupby(group_col).apply(neutralize).reset_index(level=0, drop=True)
    return df