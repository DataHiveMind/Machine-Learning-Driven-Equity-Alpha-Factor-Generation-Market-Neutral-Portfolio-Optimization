"""
src/potfolio/optimization.py

Purpose: Portfolio optimization techniques including Mean-Variance Optimization, Black-Litterman model,
and Risk Parity strategies.
"""

import pandas as pd
import numpy as np

from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from typing import Optional

def mean_variance_optimization(returns: pd.DataFrame, target_return: float) -> np.ndarray:
    """
    Perform Mean-Variance Optimization to find optimal portfolio weights.

    Parameters:
    returns (pd.DataFrame): DataFrame of asset returns.
    target_return (float): Desired target return for the portfolio.

    Returns:
    np.ndarray: Array of optimal portfolio weights.
    """
    mean_returns = returns.mean()
    cov_matrix = LedoitWolf().fit(returns).covariance_
    num_assets = len(mean_returns)

    def portfolio_variance(weights):
        return np.dot(weights.T, np.dot(cov_matrix, weights))

    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
        {'type': 'eq', 'fun': lambda x: np.dot(x, mean_returns) - target_return}  # Target return
    )
    bounds = tuple((0, 1) for _ in range(num_assets))  # Long-only constraints
    initial_guess = num_assets * [1. / num_assets]

    result = minimize(portfolio_variance, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    if result.success:
        return result.x
    else:
        raise ValueError("Optimization failed to find a solution.")
    
def black_litterman(returns: pd.DataFrame, P: np.ndarray, Q: np.ndarray, tau: float = 0.05) -> np.ndarray:
    """
    Implement the Black-Litterman model to adjust expected returns based on investor views.

    Parameters:
    returns (pd.DataFrame): DataFrame of asset returns.
    P (np.ndarray): Pick matrix representing views.
    Q (np.ndarray): View returns.
    tau (float): Scaling factor for the uncertainty in the prior estimate of the mean.

    Returns:
    np.ndarray: Adjusted expected returns.
    """
    mean_returns = returns.mean()
    cov_matrix = LedoitWolf().fit(returns).covariance_
    pi = np.dot(cov_matrix, np.linalg.inv(cov_matrix + np.eye(len(cov_matrix)) * tau)).dot(mean_returns)

    # Calculate the posterior estimate of the mean
    omega = np.diag(np.diag(np.dot(P, np.dot(tau * cov_matrix, P.T))))
    middle_term = np.linalg.inv(np.linalg.inv(tau * cov_matrix) + np.dot(P.T, np.linalg.inv(omega)).dot(P))
    adjusted_returns = middle_term.dot(np.linalg.inv(tau * cov_matrix).dot(pi) + np.dot(P.T, np.linalg.inv(omega)).dot(Q))

    return adjusted_returns

def risk_parity(returns: pd.DataFrame) -> np.ndarray:
    """
    Calculate portfolio weights using the Risk Parity approach.

    Parameters:
    returns (pd.DataFrame): DataFrame of asset returns.

    Returns:
    np.ndarray: Array of portfolio weights satisfying risk parity.
    """
    cov_matrix = LedoitWolf().fit(returns).covariance_
    num_assets = len(cov_matrix)

    def risk_contribution(weights):
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        marginal_contrib = np.dot(cov_matrix, weights)
        risk_contrib = weights * marginal_contrib / np.sqrt(portfolio_variance)
        return risk_contrib

    def objective(weights):
        rc = risk_contribution(weights)
        return np.sum((rc - np.mean(rc))**2)

    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
    bounds = tuple((0, 1) for _ in range(num_assets))  # Long-only constraints
    initial_guess = num_assets * [1. / num_assets]

    result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    if result.success:
        return result.x
    else:
        raise ValueError("Optimization failed to find a solution.")
    
if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_spd_matrix

    # Generate synthetic returns data
    np.random.seed(42)
    num_assets = 5
    num_samples = 1000
    cov_matrix = make_spd_matrix(num_assets)
    mean_returns = np.random.rand(num_assets) * 0.1
    returns = np.random.multivariate_normal(mean_returns, cov_matrix, size=num_samples)
    returns_df = pd.DataFrame(returns, columns=[f'Asset_{i+1}' for i in range(num_assets)])

    # Mean-Variance Optimization
    target_return = 0.05
    mv_weights = mean_variance_optimization(returns_df, target_return)
    print("Mean-Variance Optimal Weights:", mv_weights)

    # Black-Litterman Model
    P = np.array([[1, -1, 0, 0, 0], [0, 1, -1, 0, 0]])
    Q = np.array([0.02, 0.01])
    bl_adjusted_returns = black_litterman(returns_df, P, Q)
    print("Black-Litterman Adjusted Returns:", bl_adjusted_returns)

    # Risk Parity
    rp_weights = risk_parity(returns_df)
    print("Risk Parity Weights:", rp_weights)