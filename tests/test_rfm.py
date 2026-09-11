"""Offline smoke tests for the RFM Customer Segmentation pipeline.

Uses synthetic Online-Retail-like data so tests run fast in CI without
downloading the UCI Excel file. Verifies RFM aggregation, outlier capping,
scaling, K-Means clustering, and segment logic end to end.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def cap_outliers(series, lower_percentile=0.01, upper_percentile=0.99):
    """Mirror of segmentation.ipynb helper."""
    lower = series.quantile(lower_percentile)
    upper = series.quantile(upper_percentile)
    return series.clip(lower, upper)


@pytest.fixture
def synthetic_rfm():
    """Synthetic RFM table mimicking Online Retail aggregates (4339 customers)."""
    rng = np.random.default_rng(RANDOM_STATE)
    n = 500
    data = {
        "CustomerID": np.arange(12346, 12346 + n, dtype=float),
        "Recency": rng.integers(1, 375, n).astype(float),
        "Frequency": rng.integers(1, 50, n).astype(float),
        "Monetary": rng.uniform(0, 30000, n),
    }
    df = pd.DataFrame(data)
    # Inject known outliers to test capping
    df.loc[0, "Monetary"] = 280206.02
    df.loc[1, "Frequency"] = 210
    return df


@pytest.fixture
def synthetic_transactions():
    """Tiny transaction table to test groupby RFM aggregation."""
    rng = np.random.default_rng(RANDOM_STATE)
    # 20 transactions across 5 customers
    df = pd.DataFrame(
        {
            "CustomerID": [1, 1, 2, 2, 2, 3, 4, 4, 5, 5] * 2,
            "InvoiceNo": [f"INV{i:03d}" for i in rng.integers(100, 999, 20)],
            "InvoiceDate": pd.date_range("2011-01-01", periods=20, freq="D"),
            "Quantity": rng.integers(1, 10, 20),
            "UnitPrice": rng.uniform(1, 20, 20),
        }
    )
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    return df


def test_rfm_aggregation_logic(synthetic_transactions):
    """Groupby must produce Recency, Frequency, Monetary per customer."""
    df = synthetic_transactions
    max_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = (
        df.groupby("CustomerID")
        .agg(
            {
                "InvoiceDate": lambda x: (max_date - x.max()).days,
                "InvoiceNo": "nunique",
                "TotalPrice": "sum",
            }
        )
        .reset_index()
    )
    rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]
    assert list(rfm.columns) == ["CustomerID", "Recency", "Frequency", "Monetary"]
    assert len(rfm) == df["CustomerID"].nunique()
    assert (rfm["Recency"] >= 1).all()
    assert (rfm["Frequency"] >= 1).all()
    assert (rfm["Monetary"] > 0).all()


def test_cap_outliers_clips_percentiles(synthetic_rfm):
    """cap_outliers must clip to 1st/99th percentiles."""
    s = synthetic_rfm["Monetary"]
    capped = cap_outliers(s, 0.01, 0.99)
    lower = s.quantile(0.01)
    upper = s.quantile(0.99)
    assert capped.min() >= lower - 1e-9
    assert capped.max() <= upper + 1e-9
    # Outlier injected at index 0 should be clipped
    assert capped.iloc[0] <= upper + 1e-9


def test_scaler_zero_mean_unit_variance(synthetic_rfm):
    """StandardScaler on RFM must yield mean~0 and std~1."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(synthetic_rfm[["Recency", "Frequency", "Monetary"]])
    means = scaled.mean(axis=0)
    stds = scaled.std(axis=0, ddof=0)
    assert np.allclose(means, 0, atol=1e-9)
    assert np.allclose(stds, 1, atol=1e-9)


def test_kmeans_cluster_counts_sum(synthetic_rfm):
    """KMeans(k=4) must assign every row to [0,3] and counts sum to n."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(synthetic_rfm[["Recency", "Frequency", "Monetary"]])
    kmeans = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(scaled)
    assert len(labels) == len(synthetic_rfm)
    assert set(labels).issubset({0, 1, 2, 3})
    assert len(set(labels)) == 4  # all clusters used with n=500
    assert kmeans.cluster_centers_.shape == (4, 3)


def test_elbow_inertia_monotonic_decreasing(synthetic_rfm):
    """Inertia must strictly decrease as k increases (elbow method)."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(synthetic_rfm[["Recency", "Frequency", "Monetary"]])
    inertias = []
    for k in range(1, 7):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km.fit(scaled)
        inertias.append(km.inertia_)
    # Inertia strictly decreasing
    for i in range(1, len(inertias)):
        assert inertias[i] < inertias[i - 1]
    # k=1 inertia should be largest
    assert inertias[0] > inertias[-1]


def test_cluster_summary_means_in_range(synthetic_rfm):
    """Cluster summary means must lie within global min/max."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(synthetic_rfm[["Recency", "Frequency", "Monetary"]])
    # Cap outliers first (as notebook does)
    df = synthetic_rfm.copy()
    for col in ["Recency", "Frequency", "Monetary"]:
        df[col] = cap_outliers(df[col])
    scaled = scaler.fit_transform(df[["Recency", "Frequency", "Monetary"]])
    kmeans = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=10)
    df["Cluster"] = kmeans.fit_predict(scaled)
    summary = df.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
    assert summary.shape == (4, 3)
    for col in ["Recency", "Frequency", "Monetary"]:
        assert summary[col].min() >= df[col].min() - 1e-9
        assert summary[col].max() <= df[col].max() + 1e-9


def test_segment_mapping_covers_all_clusters():
    """Cluster -> business name map must cover 0..3 without gaps."""
    cluster_names = {0: "Champions", 1: "Lost", 2: "New/Promising", 3: "At Risk"}
    # Simulate rfm with Cluster column
    rfm = pd.DataFrame({"Cluster": [0, 1, 2, 3, 0, 1]})
    rfm["Segment"] = rfm["Cluster"].map(cluster_names)
    assert rfm["Segment"].isna().sum() == 0
    assert set(rfm["Segment"].unique()) == set(cluster_names.values())
