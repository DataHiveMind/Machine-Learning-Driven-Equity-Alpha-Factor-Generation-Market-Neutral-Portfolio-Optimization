"""
src/backtest/monte_carlo.py

Purpose: Monte Carlo return generators for backtesting trading strategies.
"""

import pandas as pd
import numpy as np

import scipy.stats as stats
from typing import Tuple

def monte_carlo_resample(returns: pd.Series, n_samples: int) -> pd.Series:
    """
    Generates Monte Carlo resampled returns based on the empirical distribution of the original returns.

    Parameters:
    returns (pd.Series): Original return series.
    n_samples (int): Number of samples to generate.

    Returns:
    pd.Series: Monte Carlo resampled returns.
    """
    # Fit a kernel density estimate to the returns
    kde = stats.gaussian_kde(returns)
    # Generate samples from the fitted distribution
    samples = kde.resample(n_samples).flatten()
    return pd.Series(samples)

def fit_distribution(returns: pd.Series) -> Tuple[float, float]:
    """
    Fits a normal distribution to the returns and returns the mean and standard deviation.

    Parameters:
    returns (pd.Series): Original return series.

    Returns:
    Tuple[float, float]: Mean and standard deviation of the fitted normal distribution.
    """
    mu, sigma = stats.norm.fit(returns)
    return mu, sigma

if __name__ == "__main__":
    # Example usage
    example_returns = pd.Series(np.random.normal(0.001, 0.02, 1000))  # Simulated daily returns
    monte_carlo_samples = monte_carlo_resample(example_returns, 1000)
    mu, sigma = fit_distribution(example_returns)
    print(f"Fitted Normal Distribution: mu={mu}, sigma={sigma}")