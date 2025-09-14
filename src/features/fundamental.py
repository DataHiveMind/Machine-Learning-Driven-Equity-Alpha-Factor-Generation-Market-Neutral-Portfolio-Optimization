"""
src/features/fundamental.py

Purpose: This module provides functions to compute fundamental financial ratios from raw financial data.
Such as Price-to-Earnings (P/E), Price-to-Book (P/B), and Debt-to-Equity (D/E) ratios, value-weighted
averages, and other key financial metrics.
"""

import pandas as pd
import numpy as np

def compute_pe_ratio(price, earnings):
    """Compute Price-to-Earnings (P/E) ratio."""
    with np.errstate(divide='ignore', invalid='ignore'):
        pe_ratio = np.where(earnings != 0, price / earnings, np.nan)
    return pe_ratio

def compute_pb_ratio(price, book_value):
    """Compute Price-to-Book (P/B) ratio."""
    with np.errstate(divide='ignore', invalid='ignore'):
        pb_ratio = np.where(book_value != 0, price / book_value, np.nan)
    return pb_ratio

def compute_de_ratio(debt, equity):
    """Compute Debt-to-Equity (D/E) ratio."""
    with np.errstate(divide='ignore', invalid='ignore'):
        de_ratio = np.where(equity != 0, debt / equity, np.nan)
    return de_ratio

def value_weighted_average(values, weights):
    """Compute value-weighted average."""
    values = np.array(values)
    weights = np.array(weights)
    if np.sum(weights) == 0:
        return np.nan
    return np.sum(values * weights) / np.sum(weights)

def compute_financial_ratios(df):
    """Compute a set of fundamental financial ratios from a DataFrame."""
    df = df.copy()
    df['P/E'] = compute_pe_ratio(df['Price'], df['Earnings'])
    df['P/B'] = compute_pb_ratio(df['Price'], df['BookValue'])
    df['D/E'] = compute_de_ratio(df['Debt'], df['Equity'])
    
    # Example of value-weighted average P/E ratio
    df['VWAP/E'] = value_weighted_average(df['P/E'], df['MarketCap'])
    
    return df

if __name__ == "__main__":
    # Example usage
    data = {
        'Price': [100, 150, 200],
        'Earnings': [10, 15, 0],
        'BookValue': [50, 75, 100],
        'Debt': [30, 50, 70],
        'Equity': [70, 100, 130],
        'MarketCap': [1000, 1500, 2000]
    }
    df = pd.DataFrame(data)
    ratios_df = compute_financial_ratios(df)
    print(ratios_df)