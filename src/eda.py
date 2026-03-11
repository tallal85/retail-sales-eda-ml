"""
eda.py
------
Exploratory Data Analysis helpers for the retail sales dataset.
Produces console summaries and saves charts to /outputs/.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from pathlib import Path

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

PALETTE = "Blues_d"
sns.set_theme(style="whitegrid", palette=PALETTE)


# ---------- summary helpers ----------

def data_overview(df: pd.DataFrame) -> None:
    print("\n=== Dataset Overview ===")
    print(f"Rows: {len(df):,}   Columns: {df.shape[1]}")
    print(f"\nNull counts:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nNumeric summary:\n{df.describe().round(2)}")


def sales_by_category(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("category")[["sales", "profit"]]
        .sum()
        .sort_values("sales", ascending=False)
        .round(2)
    )


def top_products(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return (
        df.groupby("product_name")["sales"]
        .nlargest(n)
        .reset_index()
    )


def monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["month"] = df["order_date"].dt.to_period("M")
    return df.groupby("month")["sales"].sum().reset_index()


# ---------- chart helpers ----------

def plot_sales_by_category(df: pd.DataFrame, save: bool = True) -> None:
    summary = sales_by_category(df).reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=summary, x="category", y="sales", ax=ax, palette=PALETTE)
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
    ax.set_title("Total Sales by Category", fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Sales ($)")
    plt.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "sales_by_category.png", dpi=150)
    plt.show()


def plot_profit_margin_by_segment(df: pd.DataFrame, save: bool = True) -> None:
    df = df.copy()
    df["profit_margin"] = df["profit"] / df["sales"]
    summary = df.groupby("segment")["profit_margin"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=summary, x="segment", y="profit_margin", ax=ax, palette="Greens_d")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.set_title("Avg. Profit Margin by Customer Segment", fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Profit Margin (%)")
    plt.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "profit_margin_by_segment.png", dpi=150)
    plt.show()


def plot_monthly_sales_trend(df: pd.DataFrame, save: bool = True) -> None:
    trend = monthly_trend(df)
    trend["month_str"] = trend["month"].astype(str)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(trend["month_str"], trend["sales"], marker="o", color="#2E86AB", linewidth=2)
    ax.fill_between(range(len(trend)), trend["sales"], alpha=0.15, color="#2E86AB")
    ax.set_xticks(range(len(trend)))
    ax.set_xticklabels(trend["month_str"], rotation=45, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
    ax.set_title("Monthly Sales Trend", fontsize=14, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Sales ($)")
    plt.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "monthly_sales_trend.png", dpi=150)
    plt.show()
