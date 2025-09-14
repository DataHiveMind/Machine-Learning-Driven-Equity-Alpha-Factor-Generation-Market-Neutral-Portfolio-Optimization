"""
src/features/technical.py

Purpose: Technical analysis feature engineering utilities for the project.
"""
import pandas as pd
import numpy as np


def add_returns_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add daily returns to the DataFrame.
    """
    df['Returns'] = df['Close'].pct_change()
    return df

def add_moving_averages(df: pd.DataFrame, windows: list = [5, 10, 20]) -> pd.DataFrame:
    """
    Add simple and exponential moving averages to the DataFrame.
    """
    for window in windows:
        df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'EMA_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
    return df

def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Add Relative Strength Index (RSI) to the DataFrame.
    """
    delta = df['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=window).mean()
    avg_loss = pd.Series(loss).rolling(window=window).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Add MACD and MACD Signal to the DataFrame.
    """
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    df['MACD'] = ema_fast - ema_slow
    df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    return df

def add_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: int = 2) -> pd.DataFrame:
    """
    Add Bollinger Bands to the DataFrame.
    """
    rolling_mean = df['Close'].rolling(window=window).mean()
    rolling_std = df['Close'].rolling(window=window).std()
    df['Bollinger_High'] = rolling_mean + (rolling_std * num_std)
    df['Bollinger_Low'] = rolling_mean - (rolling_std * num_std)
    return df

def add_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Add Average True Range (ATR) to the DataFrame.
    """
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=window).mean()
    return df

def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = add_rsi(df)
    df = add_macd(df)
    return df

def add_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = add_bollinger_bands(df)
    df = add_atr(df)
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