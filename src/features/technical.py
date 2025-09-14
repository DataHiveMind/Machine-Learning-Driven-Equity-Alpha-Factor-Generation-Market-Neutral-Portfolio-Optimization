"""
src/features/technical.py

Purpose: Technical analysis feature engineering utilities for the project.
"""
import pandas as pd
import numpy as np
import ta

def add_returns_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add daily returns to the DataFrame.

    Parameters:
    df (pd.DataFrame): Input DataFrame with a 'Close' column.

    Returns:
    pd.DataFrame: DataFrame with an additional 'Returns' column.
    """
    df['Returns'] = df['Close'].pct_change()
    return df

def add_moving_averages(df:pd.DataFrame , windows : list = [5, 10, 20])-> pd.DataFrame:
    """
    Add simple and exponetail moving averages to the DataFrame.

    Parameters:
    df (pd.DataFrame): Input DataFrame with a 'Close' column.
    windows (list): List of window sizes for moving averages.

    Returns:
    pd.DataFrame: DataFrame with additional moving average columns.
    """
    for window in windows:
        df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'EMA_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
    return df

def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add momentum indicators to the DataFrame.

    Parameters:
    df (pd.DataFrame): Input DataFrame with a 'Close' column.

    Returns:
    pd.DataFrame: DataFrame with additional momentum indicator columns.
    """
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['MACD'] = ta.trend.macd(df['Close'])
    df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
    return df

def add_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volatility indicators to the DataFrame.

    Parameters:
    df (pd.DataFrame): Input DataFrame with 'High', 'Low', and 'Close' columns.

    Returns:
    pd.DataFrame: DataFrame with additional volatility indicator columns.
    """
    df['Bollinger_High'] = ta.volatility.bollinger_hband(df['Close'], window=20, window_dev=2)
    df['Bollinger_Low'] = ta.volatility.bollinger_lband(df['Close'], window=20, window_dev=2)
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    return df

if __name__ == "__main__":
    # Example usage
    data = {
        'Close': [100, 102, 101, 105, 107, 106, 108, 110, 111, 115],
        'High': [101, 103, 102, 106, 108, 107, 109, 111, 112, 116],
        'Low': [99, 101, 100, 104, 106, 105, 107, 109, 110, 114]
    }
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)

    df = add_returns_column(df)
    df = add_moving_averages(df)
    df = add_momentum_indicators(df)
    df = add_volatility_indicators(df)

    print("\nDataFrame with Technical Indicators:")
    print(df)