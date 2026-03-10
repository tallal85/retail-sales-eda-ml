"""
main.py
-------
End-to-end runner: load data -> EDA -> feature engineering -> train -> evaluate.

Usage:
    python main.py --data data/sample_sales.csv
"""

import argparse
import logging
import pandas as pd

from src.eda import data_overview, plot_sales_by_category, plot_profit_margin_by_segment, plot_monthly_sales_trend
from src.features import build_feature_matrix
from src.model import prepare_data, train_model, evaluate_model, feature_importance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run(data_path: str) -> None:
    logger.info("Loading data...")
    df = pd.read_csv(data_path)

    # --- EDA ---
    logger.info("Running EDA...")
    data_overview(df)
    plot_sales_by_category(df)
    plot_profit_margin_by_segment(df)
    plot_monthly_sales_trend(df)

    # --- Feature Engineering ---
    logger.info("Building feature matrix...")
    df_features = build_feature_matrix(df)

    # --- Modelling ---
    X_train, X_test, y_train, y_test = prepare_data(df_features, target="profit")
    model = train_model(X_train, y_train)

    # --- Evaluation ---
    metrics = evaluate_model(model, X_test, y_test)
    print(f"\nModel Results:\n  MAE  : ${metrics['MAE']:.2f}\n  R2   : {metrics['R2']:.4f}")

    # --- Feature Importance ---
    fi = feature_importance(model, X_train)
    print(f"\nTop 10 Features:\n{fi.head(10).to_string(index=False)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run retail EDA and ML pipeline.")
    parser.add_argument("--data", default="data/sample_sales.csv", help="Path to sales CSV")
    args = parser.parse_args()
    run(args.data)
