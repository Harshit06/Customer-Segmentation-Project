"""
app.py
------
Interactive Streamlit dashboard for the Customer Segmentation project.

Run with:
    streamlit run app.py

Features:
  - KPI cards (total customers, avg spend, avg income, segment count)
  - Filters: Age range, Income range, City/Region, Segment
  - Segment distribution chart
  - Segment profile comparison (radar-style bar chart)
  - PCA cluster scatter plot
  - Raw filtered data table + CSV download
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "src"))
from preprocessing import load_data, handle_missing_values, cap_outliers_iqr, encode_categoricals, scale_features
from clustering import run_kmeans, reduce_dimensions

st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide", page_icon="📊")

SEGMENT_NAMES = {
    0: "Steady Mature Regulars",
    1: "New Bargain Hunters",
    2: "Premium Spenders",
    3: "At-Risk Customers",
}
SEGMENT_COLORS = {
    "Premium Spenders": "#2E7D32",
    "Steady Mature Regulars": "#1565C0",
    "New Bargain Hunters": "#F9A825",
    "At-Risk Customers": "#C62828",
}


@st.cache_data
def get_data():
    root = Path(__file__).resolve().parent
    df = load_data(str(root / "data" / "customer_data.csv"))
    df = handle_missing_values(df)
    df = cap_outliers_iqr(df, ["AnnualIncome_INR_K", "MonetaryValue_INR", "AvgOrderValue_INR"])

    df_enc, _ = encode_categoricals(df, ["Gender", "City", "ProductCategoryPreference"])
    cluster_features = ["Age", "AnnualIncome_INR_K", "SpendingScore", "PurchaseFrequency",
                         "Recency_Days", "Tenure_Months", "MonetaryValue_INR"]
    X_scaled, _ = scale_features(df_enc, cluster_features)
    labels, _ = run_kmeans(X_scaled, n_clusters=4, random_state=42)
    X_pca, _ = reduce_dimensions(X_scaled, n_components=2, random_state=42)

    df["Cluster"] = labels
    df["Segment"] = df["Cluster"].map(SEGMENT_NAMES)
    df["PC1"] = X_pca[:, 0]
    df["PC2"] = X_pca[:, 1]
    return df


df = get_data()

st.title("📊 Customer Segmentation Dashboard")
st.caption(
    "K-Means segmentation (k=4) over real Kaggle customer data (Mall Customer Segmentation Dataset) "
    "expanded with engineered behavioral features. See README for the real-vs-engineered column breakdown."
)

# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.header("🔎 Filters")

age_min, age_max = int(df["Age"].min()), int(df["Age"].max())
age_range = st.sidebar.slider("Age range", age_min, age_max, (age_min, age_max))

inc_min, inc_max = float(df["AnnualIncome_INR_K"].min()), float(df["AnnualIncome_INR_K"].max())
income_range = st.sidebar.slider("Annual Income (₹ thousands)", inc_min, inc_max, (inc_min, inc_max))

cities = sorted(df["City"].dropna().unique().tolist())
selected_cities = st.sidebar.multiselect("City / Region", cities, default=cities)

segments = sorted(df["Segment"].unique().tolist())
selected_segments = st.sidebar.multiselect("Segment", segments, default=segments)

filtered = df[
    (df["Age"].between(*age_range)) &
    (df["AnnualIncome_INR_K"].between(*income_range)) &
    (df["City"].isin(selected_cities)) &
    (df["Segment"].isin(selected_segments))
]

st.sidebar.markdown("---")
st.sidebar.metric("Customers matching filters", len(filtered))

# ---------------- KPI ROW ----------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Customers", f"{len(filtered):,}")
c2.metric("Avg Spending Score", f"{filtered['SpendingScore'].mean():.1f}" if len(filtered) else "—")
c3.metric("Avg Annual Income (₹K)", f"{filtered['AnnualIncome_INR_K'].mean():,.0f}" if len(filtered) else "—")
c4.metric("Avg Monetary Value (₹)", f"{filtered['MonetaryValue_INR'].mean():,.0f}" if len(filtered) else "—")

st.markdown("---")

if len(filtered) == 0:
    st.warning("No customers match the current filters. Try widening the range.")
    st.stop()

# ---------------- SEGMENT DISTRIBUTION ----------------
left, right = st.columns([1, 1])

with left:
    st.subheader("Segment Distribution")
    seg_counts = filtered["Segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Count"]
    fig_pie = px.pie(
        seg_counts, names="Segment", values="Count", hole=0.45,
        color="Segment", color_discrete_map=SEGMENT_COLORS,
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with right:
    st.subheader("Customer Segments — PCA Projection")
    fig_scatter = px.scatter(
        filtered, x="PC1", y="PC2", color="Segment",
        color_discrete_map=SEGMENT_COLORS,
        hover_data=["CustomerID", "Age", "AnnualIncome_INR_K", "SpendingScore"],
        opacity=0.7,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ---------------- SEGMENT PROFILE COMPARISON ----------------
st.subheader("Average Behavior by Segment")
profile_cols = ["Age", "AnnualIncome_INR_K", "SpendingScore", "PurchaseFrequency", "Recency_Days", "Tenure_Months", "MonetaryValue_INR"]
profile = filtered.groupby("Segment")[profile_cols].mean().round(1).reset_index()
st.dataframe(profile, use_container_width=True)

metric_choice = st.selectbox("Compare segments by metric", profile_cols, index=2)
fig_bar = px.bar(
    profile, x="Segment", y=metric_choice, color="Segment",
    color_discrete_map=SEGMENT_COLORS, text_auto=".1f",
)
st.plotly_chart(fig_bar, use_container_width=True)

# ---------------- CITY BREAKDOWN ----------------
st.subheader("Segment Mix by City")
city_seg = filtered.groupby(["City", "Segment"]).size().reset_index(name="Count")
fig_city = px.bar(
    city_seg, x="City", y="Count", color="Segment",
    color_discrete_map=SEGMENT_COLORS, barmode="stack",
)
st.plotly_chart(fig_city, use_container_width=True)

# ---------------- RAW DATA + DOWNLOAD ----------------
st.subheader("Filtered Customer Data")
st.dataframe(filtered.drop(columns=["PC1", "PC2"]), use_container_width=True, height=300)

csv = filtered.drop(columns=["PC1", "PC2"]).to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download filtered data as CSV", csv, "filtered_customers.csv", "text/csv")

st.markdown("---")
st.caption("Built with Streamlit + scikit-learn + Plotly | Customer Segmentation Project")
