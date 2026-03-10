# 📊 Retail Sales EDA & ML Insights

End-to-end data analysis and machine learning project on retail sales data.
Covers exploratory data analysis (EDA), feature engineering, and a Random Forest regression model to predict order profit.

---

## 📁 Project Structure

```
retail-sales-eda-ml/
├── src/
│   ├── eda.py          # EDA summaries and chart helpers
│   ├── features.py     # Feature engineering pipeline
│   └── model.py        # ML training, evaluation, feature importance
├── data/
│   └── sample_sales.csv
├── outputs/            # Generated charts (auto-created on run)
├── main.py             # End-to-end runner
├── requirements.txt
└── README.md
```

---

## 🔍 What This Project Covers

### 1. Exploratory Data Analysis
- Dataset overview (shape, nulls, dtypes)
- Sales and profit breakdown by **category**, **segment**, and **region**
- **Monthly sales trend** with time series visualisation
- Profit margin analysis per customer segment

### 2. Feature Engineering
| Feature | Description |
|---------|-------------|
| `year`, `month`, `quarter` | Temporal features from `order_date` |
| `day_of_week`, `is_weekend` | Order timing patterns |
| `revenue_per_unit` | `sales / quantity` |
| `profit_margin` | `profit / sales` |
| `discount_impact` | `sales × discount` |
| One-hot encoded | `category`, `segment`, `region`, `ship_mode` |

### 3. ML Model — Profit Prediction
- **Algorithm:** Random Forest Regressor (scikit-learn)
- **Target:** `profit` (continuous)
- **Split:** 80/20 train/test
- **Metrics:** MAE, R²
- **Output:** Feature importance ranking

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/tallal85/retail-sales-eda-ml.git
cd retail-sales-eda-ml

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run everything
python main.py --data data/sample_sales.csv
```

Charts are saved automatically to the `outputs/` folder.

---

## 📈 Sample Insights

- **Technology** drives the highest total sales but **Office Supplies** yields more consistent profit margins
- **Corporate** segment orders have higher average order value than **Consumer** or **Home Office**
- Discount rates above 20% are strongly correlated with negative profit — a key business risk signal
- Top predictors of profit: `sales`, `revenue_per_unit`, `discount_impact`

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **pandas / NumPy** — data wrangling
- **matplotlib / seaborn** — visualisation
- **scikit-learn** — ML modelling

---

## 👤 Author

**Tallal Moshrif** — Data Engineer & BI Developer
[LinkedIn](https://linkedin.com/in/tallalmoshrif) · [GitHub](https://github.com/tallal85)
