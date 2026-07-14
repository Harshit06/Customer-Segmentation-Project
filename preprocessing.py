"""
preprocessing.py
-----------------
Data cleaning, RFM feature engineering, encoding, and scaling utilities
for the Customer Segmentation project.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute numeric columns with median, categorical with mode."""
    df = df.copy()
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include="object").columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])
    return df


def cap_outliers_iqr(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Cap outliers to the 1.5*IQR fences (winsorization) for given columns."""
    df = df.copy()
    for col in cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        df[col] = df[col].clip(lower, upper)
    return df


def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds the classic RFM table:
      Recency   -> Recency_Days (lower = better, already in dataset)
      Frequency -> PurchaseFrequency
      Monetary  -> MonetaryValue_INR

    Adds RFM quartile scores (1-4, 4=best) and a combined RFM segment code,
    which is the industry-standard way analysts turn RFM into a label.
    """
    rfm = df[["CustomerID", "Recency_Days", "PurchaseFrequency", "MonetaryValue_INR"]].copy()
    rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]

    # Recency: lower is better -> reverse-score it (use rank to avoid duplicate bin edges)
    rfm["R_Score"] = pd.qcut(rfm["Recency"].rank(method="first"), 4, labels=[4, 3, 2, 1]).astype(int)
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    rfm["M_Score"] = pd.qcut(rfm["Monetary"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)

    rfm["RFM_Score"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]
    rfm["RFM_Segment"] = (
        rfm["R_Score"].astype(str) + rfm["F_Score"].astype(str) + rfm["M_Score"].astype(str)
    )
    return rfm


def encode_categoricals(df: pd.DataFrame, cols: list) -> tuple[pd.DataFrame, dict]:
    """Label-encodes categorical columns; returns encoded df + fitted encoders (for inverse_transform)."""
    df = df.copy()
    encoders = {}
    for col in cols:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col])
        encoders[col] = le
    return df, encoders


def scale_features(df: pd.DataFrame, cols: list) -> tuple[np.ndarray, StandardScaler]:
    """Standardizes given numeric columns to mean 0 / std 1 -- required before K-Means/PCA."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[cols])
    return X_scaled, scaler
