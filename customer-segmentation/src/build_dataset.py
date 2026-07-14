"""
build_dataset.py
-----------------
Builds the final working dataset for this project by combining:

  1. REAL DATA (from Kaggle):
     "Mall Customer Segmentation Data" (vjchoudhary7 / Kaggle) — 200 real
     customer records with CustomerID, Gender, Age, Annual Income (k$),
     and Spending Score (1-100). This is one of the most widely used
     public customer-segmentation datasets on Kaggle.
     Source: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python
     Local copy: data/raw/Mall_Customers.csv

  2. ENGINEERED / SIMULATED COLUMNS:
     The real Kaggle file does NOT contain transaction-level behavior
     (no purchase dates, no order history, no region), so it can't produce
     Recency / Frequency / Tenure / Product Category / City on its own —
     and it's only 200 rows, too small for a robust internship-level
     clustering demo. To meet the project's scope (RFM analysis, richer
     segmentation, 1000+ rows) we EXPAND the real base data using a
     documented, reproducible simulation:
       - Each real row is resampled (with small realistic jitter) to grow
         200 -> 1200 rows, preserving the original Age/Income/Spending
         distribution and correlations.
       - Recency, Purchase Frequency, Tenure, Product Category, City,
         Satisfaction, Monetary Value are then simulated CONDITIONAL ON
         each customer's real Spending Score / Income, so the added
         columns stay statistically consistent with the real behavioral
         signal instead of being pure random noise.

This hybrid approach (real seed data + documented synthetic augmentation)
is a standard, transparent technique when public real datasets are too
small or too narrow for the required analysis — and is disclosed clearly
here and in the README so reviewers know exactly which columns are real
Kaggle values vs. engineered additions.

Run directly:
    python src/build_dataset.py

Output:
    data/customer_data.csv   (1200 rows, real + engineered columns)
"""

import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42
TARGET_ROWS = 1200

CITIES = [
    "Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai",
    "Pune", "Kolkata", "Ahmedabad", "Jaipur", "Lucknow"
]
CATEGORIES = [
    "Electronics", "Fashion & Apparel", "Grocery", "Home & Furniture",
    "Beauty & Personal Care", "Sports & Fitness", "Books & Stationery"
]


def load_real_kaggle_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={
        "Annual Income (k$)": "AnnualIncome_INR_K",  # keep INR-scale naming consistent w/ rest of project
        "Spending Score (1-100)": "SpendingScore",
    })
    return df


def expand_real_data(df_real: pd.DataFrame, target_rows: int, seed: int) -> pd.DataFrame:
    """
    Resamples the 200 real rows up to `target_rows`, adding small Gaussian
    jitter to the numeric real columns (Age, Income, SpendingScore) so the
    expanded rows aren't exact duplicates, while staying close to the real
    distribution's mean/variance/correlation structure.
    """
    rng = np.random.default_rng(seed)
    n_repeats = int(np.ceil(target_rows / len(df_real)))
    expanded = pd.concat([df_real] * n_repeats, ignore_index=True).iloc[:target_rows].copy()

    expanded["Age"] = (expanded["Age"] + rng.normal(0, 1.2, len(expanded))).clip(18, 80).round().astype(int)
    expanded["AnnualIncome_INR_K"] = (
        expanded["AnnualIncome_INR_K"] * 12  # k$ -> rough INR-thousands equivalent scale for this project
        + rng.normal(0, 25, len(expanded))
    ).clip(150, 3000).round(1)
    expanded["SpendingScore"] = (
        expanded["SpendingScore"] + rng.normal(0, 3, len(expanded))
    ).clip(1, 100).round().astype(int)

    expanded["CustomerID"] = [f"CUST{str(i+1).zfill(5)}" for i in range(len(expanded))]
    return expanded.reset_index(drop=True)


def simulate_behavioral_columns(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    Adds Recency, PurchaseFrequency, Tenure, ProductCategoryPreference,
    City, AvgOrderValue, MonetaryValue, SatisfactionScore — all simulated,
    but CONDITIONAL on each customer's real SpendingScore/Income so the
    engineered behavior stays consistent with their real profile rather
    than being independent random noise.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    df = df.copy()

    spend_norm = df["SpendingScore"] / 100.0  # 0-1

    # Higher spending score -> more frequent purchases, lower recency (more recent)
    df["PurchaseFrequency"] = (rng.poisson(4 + spend_norm * 28)).clip(1, 80)
    df["Recency_Days"] = (rng.exponential(8 + (1 - spend_norm) * 140)).clip(1, 365).round().astype(int)
    df["Tenure_Months"] = (rng.normal(15 + spend_norm * 25, 10)).clip(1, 96).round().astype(int)

    df["City"] = rng.choice(CITIES, size=n)
    df["ProductCategoryPreference"] = rng.choice(CATEGORIES, size=n)

    avg_order_value = (df["AnnualIncome_INR_K"] * 0.02) + (df["SpendingScore"] * 2.5) + rng.normal(0, 40, n)
    df["AvgOrderValue_INR"] = avg_order_value.clip(200, 15000).round(2)
    df["MonetaryValue_INR"] = (
        df["AvgOrderValue_INR"] * df["PurchaseFrequency"] * rng.uniform(0.8, 1.2, n)
    ).round(2)

    df["SatisfactionScore"] = (
        rng.normal(2.6 + spend_norm * 1.8, 0.5)
    ).clip(1, 5).round(1)

    # small % missing values + a few outliers, same as any real messy dataset
    for col in ["AnnualIncome_INR_K", "SatisfactionScore", "City"]:
        idx = rng.choice(df.index, size=int(0.02 * n), replace=False)
        df.loc[idx, col] = np.nan
    outlier_idx = rng.choice(df.index, size=int(0.005 * n), replace=False)
    df.loc[outlier_idx, "AnnualIncome_INR_K"] = df.loc[outlier_idx, "AnnualIncome_INR_K"] * 6

    return df


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    df_real = load_real_kaggle_data(root / "data" / "raw" / "Mall_Customers.csv")
    print(f"Loaded REAL Kaggle data: {df_real.shape}")

    df_expanded = expand_real_data(df_real, TARGET_ROWS, RANDOM_SEED)
    df_final = simulate_behavioral_columns(df_expanded, RANDOM_SEED)

    col_order = [
        "CustomerID", "Age", "Gender", "City", "AnnualIncome_INR_K", "SpendingScore",
        "PurchaseFrequency", "Recency_Days", "Tenure_Months", "ProductCategoryPreference",
        "AvgOrderValue_INR", "MonetaryValue_INR", "SatisfactionScore",
    ]
    df_final = df_final[col_order].sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    out_path = root / "data" / "customer_data.csv"
    df_final.to_csv(out_path, index=False)
    print(f"Saved FINAL dataset (real base + engineered columns): {df_final.shape} -> {out_path}")
    print(df_final.head())
