"""
features.py
-----------
Feature engineering for the retail sales ML model.
"""

import pandas as pd
import numpy as np


def add_time_features(df: pd.DataFrame, date_col: str = "order_date") -> pd.DataFrame:
    """Extract temporal features from a date column."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["year"]        = df[date_col].dt.year
    df["month"]       = df[date_col].dt.month
    df["quarter"]     = df[date_col].dt.quarter
    df["day_of_week"] = df[date_col].dt.dayofweek
    df["is_weekend"]  = df["day_of_week"].isin([5, 6]).astype(int)
    return df


def add_revenue_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute revenue and profitability metrics."""
    df = df.copy()
    df["revenue_per_unit"] = df["sales"] / df["quantity"].replace(0, np.nan)
    df["profit_margin"]    = df["profit"] / df["sales"].replace(0, np.nan)
    df["discount_impact"]  = df["sales"] * df["discount"]
    return df


def encode_categoricals(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """One-hot encode categorical columns (drop first to avoid multicollinearity)."""
    return pd.get_dummies(df, columns=cols, drop_first=True)


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline."""
    df = add_time_features(df)
    df = add_revenue_features(df)
    cat_cols = [c for c in ["category", "segment", "region", "ship_mode"] if c in df.columns]
    df = encode_categoricals(df, cat_cols)
    return df
