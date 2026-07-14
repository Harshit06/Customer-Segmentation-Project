"""
generate_images.py
------------------
Reproducible script that runs the full EDA + clustering pipeline and saves
every chart used in the README / reports into images/. Useful for
regenerating screenshots after any data or logic change.

Run:
    python src/generate_images.py
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import load_data, handle_missing_values, cap_outliers_iqr, build_rfm, encode_categoricals, scale_features
from clustering import find_optimal_k, run_kmeans, run_hierarchical, run_dbscan, evaluate_clustering, reduce_dimensions, cluster_stability

sns.set_style("whitegrid")
ROOT = Path(__file__).resolve().parent.parent
IMG = str(ROOT / "images")
DATA = str(ROOT / "data")

df = load_data(f"{DATA}/customer_data.csv")
print("Loaded:", df.shape)
print("Missing values:\n", df.isna().sum())

df = handle_missing_values(df)
num_cols = ["Age", "AnnualIncome_INR_K", "SpendingScore", "PurchaseFrequency", "Recency_Days", "Tenure_Months", "AvgOrderValue_INR", "MonetaryValue_INR", "SatisfactionScore"]
df = cap_outliers_iqr(df, ["AnnualIncome_INR_K", "MonetaryValue_INR", "AvgOrderValue_INR"])

# ---- EDA plots ----
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, col in zip(axes.flat, ["Age", "AnnualIncome_INR_K", "SpendingScore", "PurchaseFrequency", "Recency_Days", "Tenure_Months"]):
    sns.histplot(df[col], kde=True, ax=ax, color="#4C72B0")
    ax.set_title(f"Distribution of {col}")
plt.tight_layout()
plt.savefig(f"{IMG}/01_univariate_distributions.png", dpi=120)
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, col in zip(axes, ["AnnualIncome_INR_K", "SpendingScore", "MonetaryValue_INR"]):
    sns.boxplot(y=df[col], ax=ax, color="#DD8452")
    ax.set_title(f"Boxplot: {col}")
plt.tight_layout()
plt.savefig(f"{IMG}/02_boxplots_outliers.png", dpi=120)
plt.close()

plt.figure(figsize=(10, 8))
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap - Numeric Features")
plt.tight_layout()
plt.savefig(f"{IMG}/03_correlation_heatmap.png", dpi=120)
plt.close()

plt.figure(figsize=(7, 6))
sns.scatterplot(data=df, x="AnnualIncome_INR_K", y="SpendingScore", hue="Gender", alpha=0.6)
plt.title("Annual Income vs Spending Score")
plt.tight_layout()
plt.savefig(f"{IMG}/04_income_vs_spending.png", dpi=120)
plt.close()

pair_cols = ["AnnualIncome_INR_K", "SpendingScore", "PurchaseFrequency", "Recency_Days"]
sns.pairplot(df[pair_cols].sample(400, random_state=42))
plt.savefig(f"{IMG}/05_pairplot.png", dpi=110)
plt.close()

# ---- RFM ----
rfm = build_rfm(df)
print("\nRFM sample:\n", rfm.head())
rfm.to_csv(f"{DATA}/rfm_table.csv", index=False)

# ---- Feature prep for clustering ----
df_enc, encoders = encode_categoricals(df, ["Gender", "City", "ProductCategoryPreference"])
cluster_features = ["Age", "AnnualIncome_INR_K", "SpendingScore", "PurchaseFrequency", "Recency_Days", "Tenure_Months", "MonetaryValue_INR"]
X_scaled, scaler = scale_features(df_enc, cluster_features)
print("\nScaled feature matrix shape:", X_scaled.shape)

# ---- Optimal K ----
opt_df = find_optimal_k(X_scaled, range(2, 11))
print("\nOptimal K search:\n", opt_df)

fig, ax1 = plt.subplots(figsize=(9, 5))
ax1.plot(opt_df["k"], opt_df["inertia"], marker="o", color="#4C72B0", label="Inertia (Elbow)")
ax1.set_xlabel("Number of Clusters (k)")
ax1.set_ylabel("Inertia", color="#4C72B0")
ax2 = ax1.twinx()
ax2.plot(opt_df["k"], opt_df["silhouette"], marker="s", color="#C44E52", label="Silhouette Score")
ax2.set_ylabel("Silhouette Score", color="#C44E52")
plt.title("Elbow Method + Silhouette Score for Optimal k")
fig.tight_layout()
plt.savefig(f"{IMG}/06_elbow_silhouette.png", dpi=120)
plt.close()

best_k = int(opt_df.loc[opt_df["silhouette"].idxmax(), "k"])
print("\nBest k by silhouette:", best_k)
# We'll use k=4 for business interpretability (4 clean personas) unless silhouette strongly disagrees
FINAL_K = 4

# ---- KMeans ----
km_labels, km_model = run_kmeans(X_scaled, FINAL_K)
km_eval = evaluate_clustering(X_scaled, km_labels)
print("\nKMeans eval:", km_eval)

# ---- Hierarchical ----
hc_labels, hc_model = run_hierarchical(X_scaled, FINAL_K)
hc_eval = evaluate_clustering(X_scaled, hc_labels)
print("Hierarchical eval:", hc_eval)

# ---- DBSCAN (tune eps using k-distance heuristic, then try a small grid) ----
from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbors=10).fit(X_scaled)
dists, _ = neigh.kneighbors(X_scaled)
k_dist = np.sort(dists[:, -1])
suggested_eps = float(np.percentile(k_dist, 90))
print(f"\nSuggested DBSCAN eps (90th pct of 10-NN distance): {suggested_eps:.3f}")

best_db = None
for eps_try in [0.8, 0.9, 1.0, 1.1, 1.2, suggested_eps]:
    for ms_try in [5, 8, 10]:
        labels_try, model_try = run_dbscan(X_scaled, eps=eps_try, min_samples=ms_try)
        eval_try = evaluate_clustering(X_scaled, labels_try)
        if eval_try["n_clusters"] >= 2 and not np.isnan(eval_try["silhouette"]):
            if best_db is None or eval_try["silhouette"] > best_db[2]["silhouette"]:
                best_db = (labels_try, model_try, eval_try, eps_try)

if best_db is not None:
    db_labels, db_model, db_eval, chosen_eps = best_db
    print(f"DBSCAN chosen eps={chosen_eps:.3f} ->", db_eval, "unique labels:", np.unique(db_labels))
else:
    db_labels, db_model = run_dbscan(X_scaled, eps=suggested_eps, min_samples=10)
    db_eval = evaluate_clustering(X_scaled, db_labels)
    print("DBSCAN fallback eval:", db_eval, "unique labels:", np.unique(db_labels))

# ---- Stability ----
stability = cluster_stability(X_scaled, FINAL_K)
print("\nCluster stability (KMeans, k=4):", stability)

# ---- PCA 2D viz ----
X_pca, pca_model = reduce_dimensions(X_scaled, 2)
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=km_labels, cmap="viridis", alpha=0.7)
plt.xlabel(f"PC1 ({pca_model.explained_variance_ratio_[0]*100:.1f}% var)")
plt.ylabel(f"PC2 ({pca_model.explained_variance_ratio_[1]*100:.1f}% var)")
plt.title(f"Customer Segments (K-Means, k={FINAL_K}) - PCA 2D Projection")
plt.colorbar(scatter, label="Cluster")
plt.tight_layout()
plt.savefig(f"{IMG}/07_pca_clusters_2d.png", dpi=120)
plt.close()

# ---- PCA 3D viz ----
from mpl_toolkits.mplot3d import Axes3D  # noqa
X_pca3, pca_model3 = reduce_dimensions(X_scaled, 3)
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")
p = ax.scatter(X_pca3[:, 0], X_pca3[:, 1], X_pca3[:, 2], c=km_labels, cmap="viridis", alpha=0.7)
ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
ax.set_title(f"Customer Segments - PCA 3D Projection (k={FINAL_K})")
fig.colorbar(p, label="Cluster", shrink=0.6)
plt.tight_layout()
plt.savefig(f"{IMG}/08_pca_clusters_3d.png", dpi=120)
plt.close()

# ---- Algorithm comparison bar chart ----
comp_df = pd.DataFrame([
    {"Algorithm": "K-Means", **km_eval},
    {"Algorithm": "Hierarchical", **hc_eval},
    {"Algorithm": "DBSCAN", **db_eval},
])
print("\nAlgorithm comparison:\n", comp_df)
comp_df.to_csv(f"{DATA}/algorithm_comparison.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.barplot(data=comp_df, x="Algorithm", y="silhouette", ax=axes[0], palette="Blues_d")
axes[0].set_title("Silhouette Score (higher = better)")
sns.barplot(data=comp_df, x="Algorithm", y="davies_bouldin", ax=axes[1], palette="Oranges_d")
axes[1].set_title("Davies-Bouldin Index (lower = better)")
plt.tight_layout()
plt.savefig(f"{IMG}/09_algorithm_comparison.png", dpi=120)
plt.close()

# ---- Cluster profiling ----
df_profile = df.copy()
df_profile["Cluster"] = km_labels
profile = df_profile.groupby("Cluster")[cluster_features].mean().round(2)
profile["CustomerCount"] = df_profile.groupby("Cluster").size()
profile["PctOfBase"] = (profile["CustomerCount"] / len(df_profile) * 100).round(1)
print("\nCluster profile:\n", profile)
profile.to_csv(f"{DATA}/cluster_profile.csv")

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, col in zip(axes.flat, ["AnnualIncome_INR_K", "SpendingScore", "PurchaseFrequency", "Recency_Days"]):
    sns.barplot(x=profile.index, y=profile[col], ax=ax, palette="viridis")
    ax.set_title(f"Avg {col} by Cluster")
    ax.set_xlabel("Cluster")
plt.tight_layout()
plt.savefig(f"{IMG}/10_cluster_profile_bars.png", dpi=120)
plt.close()

plt.figure(figsize=(7, 6))
sizes = df_profile["Cluster"].value_counts().sort_index()
plt.pie(sizes, labels=[f"Cluster {i}" for i in sizes.index], autopct="%1.1f%%",
        colors=sns.color_palette("viridis", len(sizes)))
plt.title("Customer Segment Distribution")
plt.tight_layout()
plt.savefig(f"{IMG}/11_segment_distribution_pie.png", dpi=120)
plt.close()

print("\nALL DONE - images saved to", IMG)
