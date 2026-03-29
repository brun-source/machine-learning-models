# utils.py
import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Load data from a CSV file."""
    import pandas as pd
    return pd.read_csv(file_path)

def split_data(X, y, n_splits=5, shuffle=True, random_state=42):
    """Split data into training and validation sets using KFold."""
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    return kf.split(X)

def evaluate_model(y_pred, y_true):
    """Evaluate model performance using accuracy score."""
    return accuracy_score(y_true, y_pred)

def scale_data(X_train, X_test):
    """Scale data using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled