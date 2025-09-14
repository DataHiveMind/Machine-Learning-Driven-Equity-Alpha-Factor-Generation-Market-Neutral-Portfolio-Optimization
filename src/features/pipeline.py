""" 
src/features/pipeline.py

Purpose: This module defines the data processing pipeline for feature engineering by factor standardization,
winsorization, and neutralization.
"""

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scipy.stats import mstats


class Winsorizer(BaseEstimator, TransformerMixin):
    """Custom transformer for winsorization."""
    def __init__(self, limits=(0.05, 0.05)):
        self.limits = limits

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Ensure X is a DataFrame for compatibility
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X_winsorized = X.copy()
        for col in X.columns:
            X_winsorized[col] = mstats.winsorize(X[col], limits=self.limits)
        return X_winsorized
    
class Neutralizer(BaseEstimator, TransformerMixin):
    """Custom transformer for neutralization."""
    def __init__(self, factors):
        self.factors = factors

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Ensure X is a DataFrame for compatibility
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X_neutralized = X.copy()
        for factor in self.factors:
            if factor in X.columns:
                factor_values = X[factor].values.reshape(-1, 1)
                for col in X.columns:
                    if col != factor:
                        coef = np.linalg.lstsq(factor_values, X[col].values, rcond=None)[0]
                        X_neutralized[col] -= coef * factor_values.flatten()
        return X_neutralized
    
def create_feature_pipeline(winsor_limits=(0.05, 0.05), neutralize_factors=None):
    """Creates a data processing pipeline for feature engineering."""
    steps = [
        ('standardize', StandardScaler()),
        ('winsorize', Winsorizer(limits=winsor_limits))
    ]
    
    if neutralize_factors:
        steps.append(('neutralize', Neutralizer(factors=neutralize_factors)))
    
    pipeline = Pipeline(steps)
    return pipeline

if __name__ == "__main__":
    # Example usage
    data = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100) * 10,
        'factor1': np.random.randn(100)
    })
    
    pipeline = create_feature_pipeline(neutralize_factors=['factor1'])
    processed_data = pipeline.fit_transform(data)
    print(processed_data)