""" 
src/backtest/simulators.py

Purpose: Bootstrap / Gaussian return generators for backtesting trading strategies.
"""

import pandas as pd
import numpy as np

import scipy.stats as stats
from typing import Tuple

def bootstrap_resample(returns: pd.Series, n_samples: int) -> pd.Series:
    """
    Generates bootstrap resampled returns.

    Parameters:
    returns (pd.Series): Original return series.
    n_samples (int): Number of samples to generate.

    Returns:
    pd.Series: Bootstrap resampled returns.
    """
    return pd.Series(np.random.choice(returns, size=n_samples, replace=True))

def gaussian_resample(returns: pd.Series, n_samples: int) -> pd.Series:
    """
    Generates Gaussian resampled returns based on the mean and standard deviation of the original returns.

    Parameters:
    returns (pd.Series): Original return series.
    n_samples (int): Number of samples to generate.

    Returns:
    pd.Series: Gaussian resampled returns.
    """
    mu = returns.mean()
    sigma = returns.std()
    return pd.Series(np.random.normal(mu, sigma, n_samples))

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
    bootstrap_samples = bootstrap_resample(example_returns, 1000)
    gaussian_samples = gaussian_resample(example_returns, 1000)
    mu, sigma = fit_distribution(example_returns)

    print("Bootstrap Samples Head:")
    print(bootstrap_samples.head())
    print("\nGaussian Samples Head:")
    print(gaussian_samples.head())
    print(f"\nFitted Distribution: mu={mu}, sigma={sigma}")

