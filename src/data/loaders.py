"""
src/data/loaders.py

Purpose: Data loading utilities for the project.
"""
import pandas as pd
from openbb import obb
from typing import List

# --- Equities ---
def get_equity_data(symbol: str, start_date: str) -> pd.DataFrame:
    """
    Retrieves historical price data for a single equity.

    Args:
        symbol (str): The stock ticker symbol (e.g., 'AAPL').
        start_date (str): The start date for the data in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: A DataFrame with OHLCV data, or empty if an error occurs.
    """
    import os
    try:
        data = obb.equity.price.historical(symbol=symbol, start_date=start_date, provider="yfinance")
        df = data.to_df()
        save_dir = "data/raw/equity"
        os.makedirs(save_dir, exist_ok=True)
        save_path = f"{save_dir}/{symbol}_equity_data.csv"
        df.to_csv(save_path, index=False)
        return df
    except Exception as e:
        print(f"Error fetching equity data for {symbol}: {e}")
        return pd.DataFrame()

# --- Options (Full Chain) ---
def get_options_chain(symbol: str) -> pd.DataFrame:
    """
    Retrieves the full options chain for a given equity symbol.

    Args:
        symbol (str): The stock ticker symbol (e.g., 'TSLA').

    Returns:
        pd.DataFrame: A DataFrame with the options chain, or empty if an error occurs.
    """
    import os
    try:
        data = obb.derivatives.options.chains(symbol=symbol, provider="cboe")
        df = data.to_df()
        save_dir = "data/raw/options"
        os.makedirs(save_dir, exist_ok=True)
        save_path = f"{save_dir}/{symbol}_options_chain.csv"
        df.to_csv(save_path, index=False)
        return df
    except Exception as e:
        print(f"Error fetching options chain for {symbol}: {e}")
        return pd.DataFrame()

# --- Futures ---
def get_futures_data(symbol: str, start_date: str) -> pd.DataFrame:
    """
    Retrieves historical price data for a futures contract.

    Args:
        symbol (str): The futures contract symbol (e.g., 'ES' for S&P 500).
        start_date (str): The start date for the data in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: A DataFrame with historical futures data, or empty if an error occurs.
    """
    import os
    try:
        # Note: Futures symbols can vary by provider. 'ES' is common for S&P 500 futures.
        data = obb.derivatives.futures.historical(symbol=symbol, start_date=start_date, provider="yfinance")
        df = data.to_df()
        save_dir = "data/raw/futures"
        os.makedirs(save_dir, exist_ok=True)
        save_path = f"{save_dir}/{symbol}_futures_data.csv"
        df.to_csv(save_path, index=False)
        return df
    except Exception as e:
        print(f"Error fetching futures data for {symbol}: {e}")
        return pd.DataFrame()

# --- Exchange-Traded Funds (ETF) ---
def get_etf_data(symbol: str, start_date: str) -> pd.DataFrame:
    """
    Retrieves historical price data for an ETF.

    Args:
        symbol (str): The ETF ticker symbol (e.g., 'SPY').
        start_date (str): The start date for the data in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: A DataFrame with OHLCV data, or empty if an error occurs.
    """
    import os
    try:
        # Use the correct method for ETF historical prices
        data = obb.etf.historical(symbol=symbol, start_date=start_date, provider="yfinance")
        df = data.to_df()
        save_dir = "data/raw/etf"
        os.makedirs(save_dir, exist_ok=True)
        save_path = f"{save_dir}/{symbol}_etf_data.csv"
        df.to_csv(save_path, index=False)
        return df
    except Exception as e:
        print(f"Error fetching ETF data for {symbol}: {e}")
        return pd.DataFrame()

# --- Forex (Currency Pairs) ---
def get_forex_data(symbol: str, start_date: str) -> pd.DataFrame:
    """
    Retrieves historical data for a forex currency pair.

    Args:
        symbol (str): The currency pair symbol (e.g., 'EURUSD').
        start_date (str): The start date for the data in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: A DataFrame with OHLCV data, or empty if an error occurs.
    """
    import os
    try:
        data = obb.currency.price.historical(symbol=symbol, start_date=start_date, provider="yfinance")
        df = data.to_df()
        save_dir = "data/raw/forex"
        os.makedirs(save_dir, exist_ok=True)
        save_path = f"{save_dir}/{symbol}_forex_data.csv"
        df.to_csv(save_path, index=False)
        return df
    except Exception as e:
        print(f"Error fetching forex data for {symbol}: {e}")
        return pd.DataFrame()

# --- Macroeconomic (Real GDP) ---
def get_macro_gdp(countries: List[str], start_date: str) -> pd.DataFrame:
    """
    Retrieves quarterly Real GDP data for a list of countries.

    Args:
        countries (List[str]): A list of country names (e.g., ['united_states', 'china']).
        start_date (str): The start date for the data in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: A DataFrame with GDP data, or empty if an error occurs.
    """
    import os
    try:
        # Macro data parameters can be different. Here we get quarterly Real GDP.
        # Use provider 'econdb' or 'oecd' instead of 'fred'
        data = obb.economy.gdp.real(units="billions_of_chained_2017_dollars", start_date=start_date, provider="econdb", countries=countries)
        df = data.to_df()
        save_dir = "data/raw/macro"
        os.makedirs(save_dir, exist_ok=True)
        save_path = f"{save_dir}/{'_'.join(countries)}_gdp_data.csv"
        df.to_csv(save_path, index=False)
        return df
    except Exception as e:
        print(f"Error fetching macro GDP data for {countries}: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    # Example usage
    equity_df = get_equity_data("AAPL", "2022-01-01")
    options_df = get_options_chain("AAPL")
    forex_df = get_forex_data("EURUSD", "2022-01-01")
    macro_df = get_macro_gdp(["united_states"], "2022-01-01")
    etf_df = get_etf_data("SPY", "2022-01-01")
    futures_df = get_futures_data("ES", "2022-01-01")

    print("Sample Data Outputs:")
    print("\nequity_df")
    print(equity_df.head())

    print("\noptions_df")
    print(options_df.head())

    print("\nforex_df")
    print(forex_df.head())

    print("\nmacro_df")
    print(macro_df.head())

    print("\netf_df")
    print(etf_df.head())

    print("\nfutures_df")
    print(futures_df.head())


