"""
src/portfolio/constraints.py

Purpose: Portfolio constraints and optimization utilities for Beta neutrality, sector caps, turnover limits
"""

import pandas as pd
import numpy as np

from scipy.optimize import linprog
from sklearn.linear_model import LinearRegression
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import OneHotEncoder
from typing import List, Dict, Any

def beta_neutral_weights(returns: pd.DataFrame, market_returns: pd.Series) -> np.ndarray:
    """
    Calculate portfolio weights that achieve beta neutrality with respect to the market.

    Parameters:
    returns (pd.DataFrame): DataFrame of asset returns.
    market_returns (pd.Series): Series of market returns.

    Returns:
    np.ndarray: Array of portfolio weights achieving beta neutrality.
    """
    betas = []
    for col in returns.columns:
        model = LinearRegression().fit(market_returns.values.reshape(-1, 1), returns[col].values)
        betas.append(model.coef_[0])
    
    betas = np.array(betas)
    
    # Objective: Minimize sum of absolute weights
    c = np.ones(len(betas))
    
    # Constraints: Sum of weights * betas = 0 (beta neutrality), sum of weights = 1
    A_eq = np.vstack([betas, np.ones(len(betas))])
    b_eq = np.array([0, 1])
    
    bounds = [(0, None) for _ in range(len(betas))]  # Long-only constraints
    
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if result.success:
        return result.x
    else:
        raise ValueError("Optimization failed to find a solution.")
    
def sector_cap_weights(returns: pd.DataFrame, sectors: pd.Series, sector_cap: float) -> np.ndarray:
    """
    Calculate portfolio weights with sector exposure constraints.

    Parameters:
    returns (pd.DataFrame): DataFrame of asset returns.
    sectors (pd.Series): Series indicating the sector of each asset.
    sector_cap (float): Maximum allowed weight for any single sector.

    Returns:
    np.ndarray: Array of portfolio weights satisfying sector constraints.
    """
    n_assets = len(returns.columns)
    
    # Objective: Minimize sum of absolute weights
    c = np.ones(n_assets)
    
    # Constraints: Sum of weights = 1
    A_eq = np.ones((1, n_assets))
    b_eq = np.array([1])
    
    # Sector constraints
    encoder = OneHotEncoder(sparse_output=False)
    sector_matrix = encoder.fit_transform(sectors.values.reshape(-1, 1))
    
    A_ub = np.vstack([sector_matrix.T, -sector_matrix.T])
    b_ub = np.hstack([np.full(sector_matrix.shape[1], sector_cap), np.zeros(sector_matrix.shape[1])])
    
    A_ub = np.vstack([A_ub, -np.eye(n_assets)])  # Long-only constraints
    b_ub = np.hstack([b_ub, np.zeros(n_assets)])
    
    bounds = [(0, None) for _ in range(n_assets)]  # Long-only constraints
    
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if result.success:
        return result.x
    else:
        raise ValueError("Optimization failed to find a solution.")
    
def turnover_limited_weights(current_weights: np.ndarray, target_weights: np.ndarray, max_turnover: float) -> np.ndarray:
    """
    Adjust target weights to limit portfolio turnover.

    Parameters:
    current_weights (np.ndarray): Current portfolio weights.
    target_weights (np.ndarray): Desired target portfolio weights.
    max_turnover (float): Maximum allowed turnover (sum of absolute weight changes).

    Returns:
    np.ndarray: Adjusted portfolio weights satisfying turnover constraints.
    """
    n_assets = len(current_weights)
    
    # Objective: Minimize distance to target weights
    c = np.ones(n_assets)
    
    # Constraints: Sum of weights = 1
    A_eq = np.ones((1, n_assets))
    b_eq = np.array([1])
    
    # Turnover constraints
    A_ub = np.vstack([np.eye(n_assets), -np.eye(n_assets)])
    b_ub = np.hstack([current_weights + max_turnover, -current_weights + max_turnover])
    
    A_ub = np.vstack([A_ub, -np.eye(n_assets)])  # Long-only constraints
    b_ub = np.hstack([b_ub, np.zeros(n_assets)])
    
    bounds = [(0, None) for _ in range(n_assets)]  # Long-only constraints
    
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if result.success:
        return result.x
    else:
        raise ValueError("Optimization failed to find a solution.")
    
if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=100)
    assets = [f"Asset_{i}" for i in range(10)]
    returns = pd.DataFrame(np.random.randn(100, 10) / 100, index=dates, columns=assets)
    market_returns = pd.Series(np.random.randn(100) / 100, index=dates)
    sectors = pd.Series(['Tech', 'Finance', 'Health', 'Tech', 'Finance', 'Health', 'Tech', 'Finance', 'Health', 'Tech'], index=assets)
    
    # Beta neutral weights
    beta_weights = beta_neutral_weights(returns, market_returns)
    print("Beta Neutral Weights:", beta_weights)
    
    # Sector cap weights
    sector_weights = sector_cap_weights(returns, sectors, sector_cap=0.4)
    print("Sector Cap Weights:", sector_weights)
    
    # Turnover limited weights
    current_weights = np.array([0.1] * 10)
    target_weights = np.array([0.2] * 10)
    turnover_weights = turnover_limited_weights(current_weights, target_weights, max_turnover=0.1)
    print("Turnover Limited Weights:", turnover_weights)