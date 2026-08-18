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


# --------------------------------------------------------------------------- #
# Cost-sensitive alert-budget optimization
# --------------------------------------------------------------------------- #
def test_cost_sensitive_threshold_flags_budget_fraction(synthetic):
    X, y = synthetic
    X_tr, X_te, y_tr, y_te = ad.split_data(X, y, random_state=9)
    clf = ad.fit_isolation_forest(X_tr, n_estimators=50, random_state=10)
    scores = ad.anomaly_scores(clf, X_te)
    n = len(y_te)
    pt = ad.cost_sensitive_threshold(y_te, scores, budget_frac=0.1)
    # Review rate should be ~10% of the volume (rounding allows +/- 1 alert).
    assert pt["n_alerts"] / n == pytest.approx(0.1, abs=1 / n)


def test_cost_sensitive_threshold_catches_more_fraud_at_higher_budget(synthetic):
    X, y = synthetic
    X_tr, X_te, y_tr, y_te = ad.split_data(X, y, random_state=11)
    clf = ad.fit_isolation_forest(X_tr, n_estimators=50, random_state=12)
    scores = ad.anomaly_scores(clf, X_te)
    low = ad.cost_sensitive_threshold(y_te, scores, budget_frac=0.02)
    high = ad.cost_sensitive_threshold(y_te, scores, budget_frac=0.2)
    # A larger review budget cannot catch fewer fraud (it's a superset of alerts).
    assert high["n_fraud_caught"] >= low["n_fraud_caught"]
    assert high["recall"] >= low["recall"]
    # Precision can drop as the budget grows — assert it stays within [0,1].
    assert 0.0 <= high["precision"] <= 1.0


def test_cost_sensitive_threshold_perfect_on_separable(synthetic):
    X, y = synthetic
    X_tr, X_te, y_tr, y_te = ad.split_data(X, y, random_state=13)
    clf = ad.fit_isolation_forest(X_tr, n_estimators=50, random_state=14)
    scores = ad.anomaly_scores(clf, X_te)
    # Budget sized to the exact number of fraud -> catching all fraud gives recall 1.
    n_fraud = int(y_te.sum())
    n = len(y_te)
    pt = ad.cost_sensitive_threshold(y_te, scores, budget_frac=n_fraud / n)
    assert pt["n_alerts"] >= n_fraud
    assert pt["n_fraud_caught"] == n_fraud
    assert pt["recall"] == pytest.approx(1.0)


def test_cost_sensitive_threshold_rejects_empty():
    with pytest.raises(ValueError):
        ad.cost_sensitive_threshold(np.array([]), np.array([]), budget_frac=0.1)


def test_alert_budget_sweep_returns_all_budgets(synthetic):
    X, y = synthetic
    X_tr, X_te, y_tr, y_te = ad.split_data(X, y, random_state=15)
    clf = ad.fit_isolation_forest(X_tr, n_estimators=50, random_state=16)
    scores = ad.anomaly_scores(clf, X_te)
    sweep = ad.alert_budget_sweep(y_te, scores, budgets=(0.01, 0.05, 0.1))
    assert [pt["budget"] for pt in sweep] == [0.01, 0.05, 0.1]
    for pt in sweep:
        assert "precision" in pt and "recall" in pt
        assert "threshold" in pt


def test_run_all_includes_alert_budget(synthetic):
    X, y = synthetic
    result = ad.run_all(X, y, random_state=17)
    assert "alert_budget" in result
    assert set(result["alert_budget"]) == {"IsolationForest", "LOF", "OneClassSVM", "LogReg"}
    for name in result["alert_budget"]:
        assert len(result["alert_budget"][name]) > 0
