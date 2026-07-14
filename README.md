# 🛍️ Customer Segmentation for E-Commerce

**Unsupervised Machine Learning project that segments retail customers into actionable personas using RFM analysis and clustering (K-Means, Hierarchical, DBSCAN).**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Viz-3F4F75?logo=plotly&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📌 Project Overview & Problem Statement

An e-commerce company is running the same marketing strategy for every customer — same emails,
same discounts, same ads — regardless of whether the customer is a high-spending loyalist or
someone who hasn't ordered in months. This is expensive and ineffective.

**Goal:** Use unsupervised machine learning to segment the customer base into a small number of
statistically distinct, business-interpretable groups based on income, spending behavior,
purchase frequency, and recency — so marketing can design a tailored strategy per segment
instead of one blurry strategy for everyone.

This project covers the full data science lifecycle: data sourcing, cleaning, EDA, feature
engineering (RFM), clustering, model comparison, evaluation, business storytelling, and an
interactive dashboard.

---

## 📊 Dataset

This project uses a **hybrid real + engineered dataset**, which is disclosed transparently below.

**Real data (from Kaggle):** [Mall Customer Segmentation Data](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
by vjchoudhary7 — one of the most widely used public customer-segmentation datasets. It provides
**200 real customer records** with `CustomerID`, `Gender`, `Age`, `Annual Income`, and `Spending Score`.
A local copy is stored at `data/raw/Mall_Customers.csv`.

**Why augment it:** The real Kaggle file has no transaction history (no order dates, no region,
no category data), so it cannot support RFM analysis or a richer segmentation on its own, and at
200 rows it's small for a robust clustering demo. `src/build_dataset.py` expands the 200 real
rows to **1,200** (resampling + small realistic jitter that preserves the original distribution)
and adds behavioral columns that are simulated **conditionally on each customer's real Spending
Score / Income** — not independent random noise — so the added behavior stays statistically
consistent with real signal. The full logic is documented in that file.

| Column | Source | Description |
|---|---|---|
| `CustomerID` | Real (Kaggle) | Unique customer identifier |
| `Age` | Real (Kaggle, jittered) | Customer age |
| `Gender` | Real (Kaggle) | Male / Female |
| `AnnualIncome_INR_K` | Real (Kaggle, jittered) | Annual income, thousands |
| `SpendingScore` | Real (Kaggle, jittered) | Store-assigned spending score (1-100) |
| `City` | **Engineered** | Simulated region across 10 major Indian cities |
| `PurchaseFrequency` | **Engineered** | Orders/year, conditional on Spending Score |
| `Recency_Days` | **Engineered** | Days since last order, conditional on Spending Score |
| `Tenure_Months` | **Engineered** | Months as a customer, conditional on Spending Score |
| `ProductCategoryPreference` | **Engineered** | Preferred product category |
| `AvgOrderValue_INR` | **Engineered** | Derived from income + spending score |
| `MonetaryValue_INR` | **Engineered** | Total spend, used as the "M" in RFM |
| `SatisfactionScore` | **Engineered** | 1-5 satisfaction rating, conditional on Spending Score |

**Rows:** 1,200 · **Columns:** 13 · Includes realistic missing values (~2%) and outliers, same as
you'd find in a real CRM export.

> Want to regenerate it? Run `python src/build_dataset.py`.

---

## 🛠️ Tools & Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Data handling | pandas, NumPy |
| Visualization | matplotlib, seaborn, Plotly |
| Machine Learning | scikit-learn (K-Means, Agglomerative Clustering, DBSCAN, PCA) |
| Dashboard | Streamlit |
| Notebook | Jupyter |
| Reporting | Markdown / PDF executive summary |

---

## 🧭 Methodology

1. **Data Sourcing** — Load real Kaggle base data, expand + enrich into the working dataset (`src/build_dataset.py`).
2. **Data Cleaning** — Handle missing values (median/mode imputation), cap outliers (IQR method).
3. **EDA** — Univariate & bivariate visualizations, correlation heatmap, pairplots, insight write-ups.
4. **Feature Engineering** — RFM (Recency, Frequency, Monetary) scoring, categorical encoding, `StandardScaler` feature scaling.
5. **Optimal k Selection** — Elbow method (inertia) + Silhouette score across k=2 to 10.
6. **Clustering** — K-Means, Hierarchical (Ward linkage), and DBSCAN, compared head-to-head.
7. **Dimensionality Reduction** — PCA to 2D/3D for visualization.
8. **Cluster Profiling** — Named, business-interpretable personas with average-behavior tables.
9. **Model Evaluation** — Silhouette Score, Davies-Bouldin Index, cluster stability check across resamples.
10. **Dashboard** — Interactive Streamlit app with filters, KPIs, and segment visualizations.


---

## 🖼️ Screenshots

> Generated automatically by `src/generate_images.py` into `images/`. Replace with your own run's
> output if you regenerate the data.

| | |
|---|---|
| ![Distributions](images/01_univariate_distributions.png) | ![Correlation Heatmap](images/03_correlation_heatmap.png) |
| ![Elbow + Silhouette](images/06_elbow_silhouette.png) | ![PCA Clusters](images/07_pca_clusters_2d.png) |
| ![Algorithm Comparison](images/09_algorithm_comparison.png) | ![Segment Distribution](images/11_segment_distribution_pie.png) |

**Streamlit dashboard:** `![Dashboard Screenshot](images/dashboard_screenshot.png)` — *add your
own screenshot here after running `streamlit run app.py` locally.*

---

## 💡 Key Insights & Business Impact

- **4 clean, stable customer segments** were found: **Premium Spenders** (21%), **Steady Mature
  Regulars** (24%), **New Bargain Hunters** (28%), and **At-Risk Customers** (27%).
- **K-Means (k=4)** was selected as the production model — it matched Hierarchical clustering's
  structure closely (validating the segments are real) and produced clean, balanced,
  business-usable groups, unlike DBSCAN, which fragmented the data into one dominant cluster plus
  many tiny outlier groups despite a higher raw silhouette score.
- **At-Risk Customers** have relatively high income but the lowest spending score and highest
  recency (~120 days since last order) — a high-value win-back opportunity, since low income
  isn't the barrier.
- **Cluster stability check** (10 reruns on 80% resamples) showed low variance in silhouette
  score, confirming the segments are robust and not a one-off artifact.
- **Business impact:** replacing one blanket marketing strategy with four targeted ones
  (VIP tier, cross-sell nudges, price-sensitive promos, and a win-back campaign) directly targets
  retention and revenue growth on the highest-opportunity segments.

Full recommendations per segment are in the notebook (Section 15) and `reports/executive_summary.md` / `.pdf`.

---

## 🚀 How to Run This Project

### 1. Clone & install dependencies
```bash
git clone <your-repo-url>
cd customer-segmentation
pip install -r requirements.txt
```

### 2. (Re)generate the dataset
```bash
python src/build_dataset.py
```

### 3. Run the analysis notebook
```bash
jupyter notebook notebooks/Customer_Segmentation_Analysis.ipynb
```

### 4. Regenerate all chart images (optional)
```bash
python src/generate_images.py
```

### 5. Launch the interactive dashboard
```bash
streamlit run app.py
```

---

## 📁 Folder Structure

```
customer-segmentation/
│
├── data/
│   ├── raw/
│   │   └── Mall_Customers.csv          # Real Kaggle source data (200 rows)
│   ├── customer_data.csv               # Final working dataset (1,200 rows, real + engineered)
│   ├── customer_data_with_segments.csv # Output: customers labeled with their cluster/segment
│   ├── rfm_table.csv                   # RFM scores per customer
│   ├── cluster_profile.csv             # Avg behavior per cluster
│   └── algorithm_comparison.csv        # K-Means vs Hierarchical vs DBSCAN metrics
│
├── notebooks/
│   └── Customer_Segmentation_Analysis.ipynb   # Full, commented, story-driven analysis
│
├── src/
│   ├── build_dataset.py     # Loads real Kaggle data + builds the enriched working dataset
│   ├── preprocessing.py     # Cleaning, RFM, encoding, scaling utilities
│   ├── clustering.py        # K-Means / Hierarchical / DBSCAN / PCA / evaluation utilities
│   └── generate_images.py   # Reproduces every chart used in this README/report
│
├── reports/
│   ├── executive_summary.md   # Business-facing summary (source)
│   └── executive_summary.pdf  # Business-facing summary (client-ready PDF)
│
├── images/                  # All generated charts (EDA, clustering, profiling)
│
├── app.py                   # Streamlit interactive dashboard
├── requirements.txt
└── README.md
```

---

## 🔭 Future Scope

- Swap the engineered behavioral columns for real transaction-level data (order history, timestamps) if/when available, to remove the simulation layer entirely.
- Add a supervised "next best action" model on top of segments (e.g. predict churn probability within the At-Risk segment).
- Track segment migration over time (cohort analysis) to measure whether marketing interventions actually move customers into higher-value segments.
- Deploy the Streamlit dashboard (Streamlit Community Cloud / Docker + cloud hosting) for live stakeholder access.
- Experiment with Gaussian Mixture Models and HDBSCAN as additional clustering baselines.
- Automate the pipeline (Airflow/Prefect) to re-segment customers on a recurring schedule as new data arrives.

---

## 👤 Author / Contact

**Harshit Fartiyal**
📧 fartiyalharshit41@gmail.com · 🔗 [LinkedIn](https://www.linkedin.com/in/harshitfartiyal) · 💻 [GitHub](https://github.com/Harshit06)

*This project was built as part of an internship/portfolio submission to demonstrate an
end-to-end unsupervised machine learning workflow — from raw data to a business-ready dashboard.*

---

## 📄 License

This project is released under the MIT License — free to use, modify, and learn from.
