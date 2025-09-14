"""
src/modeling/models.py

Purpose: Modeling utilities for training and evaluating machine learning models, using tensorflow for LightGBM, XGBoost,  and pyTorch for linear models
"""
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, r2_score

from tensorflow import keras
import lightgbm as lgb
import xgboost as xgb

import torch
import torch.nn as nn

def evaluate_regression_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Evaluates regression model performance using common metrics.

    Parameters:
    y_true (np.ndarray): True target values.
    y_pred (np.ndarray): Predicted target values.

    Returns:
    dict: Dictionary containing RMSE and R² scores.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "R2": r2}

class SimpleNN(nn.Module):
    """A simple feedforward neural network for regression tasks."""
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
def train_lightgbm(X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: pd.DataFrame, y_val: pd.Series,
                   params: dict = None, num_boost_round: int = 1000,
                   early_stopping_rounds: int = 50) -> lgb.Booster:
    """
    Trains a LightGBM model.

    Parameters:
    X_train (pd.DataFrame): Training features.
    y_train (pd.Series): Training target.
    X_val (pd.DataFrame): Validation features.
    y_val (pd.Series): Validation target.
    params (dict): LightGBM parameters.
    num_boost_round (int): Number of boosting rounds.
    early_stopping_rounds (int): Early stopping rounds.

    Returns:
    lgb.Booster: Trained LightGBM model.
    """
    if params is None:
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1
        }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(params, train_data, num_boost_round=num_boost_round,
                      valid_sets=[train_data, val_data],
                      callbacks=[lgb.early_stopping(early_stopping_rounds)])
    
    return model
from typing import Optional

def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series,
                  X_val: pd.DataFrame, y_val: pd.Series,
                  params: Optional[dict] = None, num_boost_round: int = 1000,
                  early_stopping_rounds: int = 50) -> xgb.Booster:
    """
    Trains an XGBoost model.
    Parameters:
    X_train (pd.DataFrame): Training features.
    y_train (pd.Series): Training target.
    X_val (pd.DataFrame): Validation features.
    y_val (pd.Series): Validation target.
    params (dict): XGBoost parameters.
    num_boost_round (int): Number of boosting rounds.
    early_stopping_rounds (int): Early stopping rounds.

    Returns:
    xgb.Booster: Trained XGBoost model.
    """
    if params is None:

        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'verbosity': 0
        }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    evals = [(dtrain, 'train'), (dval, 'eval')]
    model = xgb.train(params, dtrain, num_boost_round=num_boost_round,
                      evals=evals, early_stopping_rounds=early_stopping_rounds) 
    return model

def train_simple_nn(X_train: pd.DataFrame, y_train: pd.Series,
                    X_val: pd.DataFrame, y_val: pd.Series,
                    input_dim: int, hidden_dim: int = 64,
                    epochs: int = 100, batch_size: int = 32,
                    learning_rate: float = 0.001) -> nn.Module:
    """
    Trains a simple feedforward neural network.

    Parameters:
    X_train (pd.DataFrame): Training features.
    y_train (pd.Series): Training target.
    X_val (pd.DataFrame): Validation features.
    y_val (pd.Series): Validation target.
    input_dim (int): Number of input features.
    hidden_dim (int): Number of hidden layer units.
    epochs (int): Number of training epochs.
    batch_size (int): Batch size for training.
    learning_rate (float): Learning rate for optimizer.

    Returns:
    nn.Module: Trained neural network model.
    """
    model = SimpleNN(input_dim, hidden_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

    return model


if __name__ == "__main__":
    # Example usage
    from sklearn.model_selection import train_test_split

    # Generate some synthetic data
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(1000, 10), columns=[f'feature_{i}' for i in range(10)])
    y = pd.Series(np.random.randn(1000), name='target')

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate LightGBM
    lgb_model = train_lightgbm(X_train, y_train, X_val, y_val)
    lgb_preds = lgb_model.predict(X_val)
    lgb_metrics = evaluate_regression_model(y_val, lgb_preds)
    print("LightGBM Metrics:", lgb_metrics)

    # Train and evaluate XGBoost
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val)
    xgb_dval = xgb.DMatrix(X_val)
    xgb_preds = xgb_model.predict(xgb_dval)
    xgb_metrics = evaluate_regression_model(y_val, xgb_preds)
    print("XGBoost Metrics:", xgb_metrics)

    # Train and evaluate SimpleNN
    nn_model = train_simple_nn(X_train, y_train, X_val, y_val, input_dim=X.shape[1])
    nn_model.eval()
    with torch.no_grad():
        X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
        nn_preds = nn_model(X_val_tensor).numpy().flatten()
    nn_metrics = evaluate_regression_model(y_val, nn_preds)
    print("SimpleNN Metrics:", nn_metrics)
