"""Smoke tests for the Credit Card Fraud anomaly-detection pipeline.

Uses small, deterministic synthetic data so the tests run fast and need no
download. Verifies the loaders, detectors, ranking metrics, and the supervised
baseline end to end on data with a clear anomaly structure.
"""

import anomaly_analysis as ad
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic():
    """Two blobs: a dense 'normal' cloud and a small, shifted 'fraud' cloud.

    29 features (V1..V28, Time, Amount) to match the real schema shape.
    """
    rng = np.random.default_rng(11)
    n_normal, n_fraud = 900, 40
    n_feat = 30  # matches the real schema: V1..V28 + Time + Amount
    normal = rng.normal(0, 1, size=(n_normal, n_feat))
    fraud = rng.normal(6, 1, size=(n_fraud, n_feat))  # clearly anomalous
    X = np.vstack([normal, fraud])
    y = np.array([0] * n_normal + [1] * n_fraud)
    # Shuffle so the fraud rows are not contiguous.
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]
    cols = [f"V{i + 1}" for i in range(28)] + ["Time", "Amount"]
    return pd.DataFrame(X, columns=cols), pd.Series(y)


def test_split_is_stratified_and_sized(synthetic):
    X, y = synthetic
    X_tr, X_te, y_tr, y_te = ad.split_data(X, y, random_state=1)
    assert len(X_tr) + len(X_te) == len(X)
    assert len(X_te) == pytest.approx(0.2 * len(X), abs=1)
    # Both splits must contain fraud (stratified).
    assert y_tr.sum() > 0 and y_te.sum() > 0


def test_isolation_forest_ranks_anomalies(synthetic):
    X, y = synthetic
    X_tr, X_te, y_tr, y_te = ad.split_data(X, y, random_state=2)
    clf = ad.fit_isolation_forest(X_tr, n_estimators=50, random_state=3)
    scores = ad.anomaly_scores(clf, X_te)
    assert len(scores) == len(X_te)
    # Fraud should score more anomalous (higher) than the normal mean.
    assert scores[y_te == 1].mean() > scores[y_te == 0].mean()


def test_lof_runs_and_scores(synthetic):
    X, y = synthetic
    X_tr, X_te, y_tr, y_te = ad.split_data(X, y, random_state=4)
    clf = ad.fit_lof(X_tr, n_neighbors=15)
    scores = ad.anomaly_scores(clf, X_te)
    assert len(scores) == len(X_te)
    assert np.all(np.isfinite(scores))


def test_ocsvm_runs_and_scores(synthetic):
    X, y = synthetic
    X_tr, X_te, y_tr, y_te = ad.split_data(X, y, random_state=5)
    clf = ad.fit_ocsvm(X_tr, random_state=6)
    scores = ad.anomaly_scores(clf, X_te)
    assert len(scores) == len(X_te)
    assert np.all(np.isfinite(scores))


def test_rank_metrics_bounds_and_perfect():
    y = np.array([0, 0, 0, 1, 1])
    perfect = np.array([0.1, 0.2, 0.3, 0.9, 0.8])  # fraud ranks highest
    m = ad.rank_metrics(y, perfect)
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["precision_at_k"] == pytest.approx(1.0)
    assert m["recall_at_k"] == pytest.approx(1.0)
    assert 0.0 <= m["average_precision"] <= 1.0


def test_rank_metrics_random_baseline_low():
    rng = np.random.default_rng(0)
    y = np.array([0] * 100 + [1] * 5)
    scores = rng.normal(0, 1, len(y))
    m = ad.rank_metrics(y, scores)
    # A random ranker should not beat a near-perfect detector badly.
    assert m["roc_auc"] < 0.9


def test_supervised_lr_separable(synthetic):
    X, y = synthetic
    X_tr, X_te, y_tr, y_te = ad.split_data(X, y, random_state=7)
    pipe = ad.fit_supervised(X_tr, y_tr)
    res = ad.supervised_metrics(pipe, X_te, y_te)
    assert res["roc_auc"] > 0.9
    assert res["recall_fraud"] > 0.5
    assert len(res["scores"]) == len(y_te)


def test_run_all_returns_expected_keys(synthetic):
    X, y = synthetic
    # Cap OCSVM implicitly; synthetic is small so no subsample needed.
    result = ad.run_all(X, y, random_state=8)
    assert set(result["detectors"]) == {"IsolationForest", "LOF", "OneClassSVM"}
    for name in result["detectors"]:
        assert "roc_auc" in result["detectors"][name]["metrics"]
    assert "roc_auc" in result["supervised"]
    assert result["pca_explained"][-1] == pytest.approx(1.0)  # PCA cumulative ends at 1
