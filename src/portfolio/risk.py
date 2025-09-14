"""
src/portfolio/risk.py

Purpose: Risk management and evaluation metrics for portfolio performance by Covariance estimation, shrinkage
"""
import pandas as pd
import numpy as np

from sklearn.covariance import LedoitWolf, EmpiricalCovariance
from sklearn.metrics import mean_squared_error
from scipy.stats import zscore

def calculate_portfolio_variance(returns: pd.DataFrame, weights: np.ndarray, method: str = 'ledoit_wolf') -> float:
    """
    Calculate the portfolio variance using the specified covariance estimation method.

    Parameters:
    returns (pd.DataFrame): DataFrame of asset returns.
    weights (np.ndarray): Array of portfolio weights.
    method (str): Covariance estimation method ('ledoit_wolf' or 'empirical').

    Returns:
    float: The calculated portfolio variance.
    """
    if method == 'ledoit_wolf':
        cov_estimator = LedoitWolf()
    elif method == 'empirical':
        cov_estimator = EmpiricalCovariance()
    else:
        raise ValueError("Method must be 'ledoit_wolf' or 'empirical'.")

    cov_estimator.fit(returns)
    cov_matrix = cov_estimator.covariance_
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    
    return portfolio_variance

def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Calculate the Sharpe Ratio of a return series.

    Parameters:
    returns (pd.Series): Series of asset returns.
    risk_free_rate (float): The risk-free rate for the Sharpe ratio calculation.

    Returns:
    float: The calculated Sharpe Ratio.
    """
    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / np.std(excess_returns)

def max_drawdown(returns: pd.Series) -> float:
    """
    Calculate the Maximum Drawdown of a return series.

    Parameters:
    returns (pd.Series): Series of asset returns.

    Returns:
    float: The calculated Maximum Drawdown.
    """
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_dd = drawdown.min()
    
    return max_dd

def value_at_risk(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Calculate the Value at Risk (VaR) of a return series.

    Parameters:
    returns (pd.Series): Series of asset returns.
    confidence_level (float): The confidence level for VaR calculation.
    
    Returns:
    float: The calculated Value at Risk.
    """
    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be between 0 and 1.")
    
    var = np.percentile(returns, (1 - confidence_level) * 100)
    return var

def conditional_value_at_risk(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Calculate the Conditional Value at Risk (CVaR) of a return series.

    Parameters:
    returns (pd.Series): Series of asset returns.
    confidence_level (float): The confidence level for CVaR calculation.

    Returns:
    float: The calculated Conditional Value at Risk.
    """
    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be between 0 and 1.")
    
    var = value_at_risk(returns, confidence_level)
    cvar = returns[returns <= var].mean()
    
    return cvar

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    returns = pd.DataFrame(np.random.randn(1000, 5) / 100, columns=[f'Asset_{i}' for i in range(5)])
    weights = np.array([0.2] * 5)

    portfolio_var = calculate_portfolio_variance(returns, weights, method='ledoit_wolf')
    print(f"Portfolio Variance: {portfolio_var}")

    asset_returns = returns['Asset_0']
    sr = sharpe_ratio(asset_returns)
    print(f"Sharpe Ratio: {sr}")

    mdd = max_drawdown(asset_returns)
    print(f"Maximum Drawdown: {mdd}")

    var = value_at_risk(asset_returns)
    print(f"Value at Risk (95%): {var}")

    cvar = conditional_value_at_risk(asset_returns)
    print(f"Conditional Value at Risk (95%): {cvar}")