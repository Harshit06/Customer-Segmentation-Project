"""
clustering.py
--------------
Clustering utilities: optimal-k search (elbow + silhouette), K-Means,
Hierarchical (Agglomerative), DBSCAN, and PCA for visualization.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA


def find_optimal_k(X, k_range=range(2, 11), random_state=42) -> pd.DataFrame:
    """
    Runs K-Means for each k in k_range and records:
      - inertia (for the elbow method)
      - silhouette score (higher is better, range -1 to 1)
      - Davies-Bouldin index (lower is better)
    Returns a tidy DataFrame so results can be plotted / compared easily.
    """
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=random_state)
        labels = km.fit_predict(X)
        rows.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(X, labels),
            "davies_bouldin": davies_bouldin_score(X, labels),
        })
    return pd.DataFrame(rows)


def run_kmeans(X, n_clusters, random_state=42):
    km = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, random_state=random_state)
    labels = km.fit_predict(X)
    return labels, km


def run_hierarchical(X, n_clusters, linkage="ward"):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(X)
    return labels, model


def run_dbscan(X, eps=0.8, min_samples=10):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    return labels, model


def evaluate_clustering(X, labels) -> dict:
    """
    Returns silhouette + Davies-Bouldin scores.
    DBSCAN can produce a single cluster or all-noise (-1); guard against that
    so evaluation doesn't crash on invalid label counts.
    """
    n_valid_clusters = len(set(labels) - {-1})
    if n_valid_clusters < 2:
        return {"silhouette": np.nan, "davies_bouldin": np.nan, "n_clusters": n_valid_clusters}
    mask = labels != -1
    return {
        "silhouette": silhouette_score(X[mask], labels[mask]),
        "davies_bouldin": davies_bouldin_score(X[mask], labels[mask]),
        "n_clusters": n_valid_clusters,
    }


def reduce_dimensions(X, n_components=2, random_state=42):
    pca = PCA(n_components=n_components, random_state=random_state)
    X_reduced = pca.fit_transform(X)
    return X_reduced, pca


def cluster_stability(X, n_clusters, n_runs=10, random_state=42):
    """
    Simple stability check: re-run K-Means with different random seeds/
    subsamples and measure how much the resulting silhouette score varies.
    A small standard deviation => stable, trustworthy clusters.
    """
    scores = []
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    for i in range(n_runs):
        sample_idx = rng.choice(n, size=int(n * 0.8), replace=False)
        X_sample = X[sample_idx]
        km = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, random_state=int(rng.integers(0, 10000)))
        labels = km.fit_predict(X_sample)
        scores.append(silhouette_score(X_sample, labels))
    return {"mean_silhouette": float(np.mean(scores)), "std_silhouette": float(np.std(scores)), "runs": n_runs}
