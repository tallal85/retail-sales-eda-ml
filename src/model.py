"""
model.py
--------
Trains and evaluates a Random Forest model to predict sales profit.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import logging

logger = logging.getLogger(__name__)


def prepare_data(df: pd.DataFrame, target: str = "profit") -> tuple:
    """Split features and target, returning train/test sets."""
    df = df.copy()
    drop_cols = [target, "order_id", "customer_id", "customer_name",
                 "product_id", "product_name", "order_date", "ship_date"]
    drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=drop_cols)
    X = X.select_dtypes(include=[np.number]).fillna(0)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info(f"Train size: {len(X_train):,}  Test size: {len(X_test):,}")
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, n_estimators: int = 100) -> RandomForestRegressor:
    """Train a Random Forest regressor."""
    logger.info(f"Training Random Forest (n_estimators={n_estimators})...")
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    logger.info("Training complete.")
    return model


def evaluate_model(model, X_test, y_test) -> dict:
    """Return MAE and R2 on the test set."""
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2  = r2_score(y_test, preds)
    metrics = {"MAE": round(mae, 2), "R2": round(r2, 4)}
    logger.info(f"Model evaluation -> MAE: {mae:.2f}  R2: {r2:.4f}")
    return metrics


def feature_importance(model, X_train) -> pd.DataFrame:
    """Return a sorted DataFrame of feature importances."""
    fi = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return fi
