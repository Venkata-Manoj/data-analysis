"""Split-validation smoke tests for the Wine Quality Classification pipeline.

Uses a small synthetic dataset that mirrors the UCI winequality-red schema so
the tests run fast and work in CI without downloading real data. Verifies the
train/test split, scaling, model training, and metric computation end to end.
"""

import csv

import numpy as np
import pandas as pd
import pytest
import wine_analysis as wine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


@pytest.fixture
def synthetic_df():
    """Deterministic synthetic wine-quality dataset (12 features + quality)."""
    rng = np.random.default_rng(42)
    n = 400
    data = {
        "fixed acidity": rng.uniform(4, 16, n),
        "volatile acidity": rng.uniform(0.1, 1.6, n),
        "citric acid": rng.uniform(0, 1, n),
        "residual sugar": rng.uniform(0.6, 15, n),
        "chlorides": rng.uniform(0.01, 0.6, n),
        "free sulfur dioxide": rng.uniform(1, 72, n),
        "total sulfur dioxide": rng.uniform(6, 289, n),
        "density": rng.uniform(0.99, 1.0, n),
        "pH": rng.uniform(2.7, 4, n),
        "sulphates": rng.uniform(0.3, 2, n),
        "alcohol": rng.uniform(8, 15, n),
        "quality": rng.integers(3, 9, n),
    }
    df = pd.DataFrame(data)
    # Keep both classes present so stratified split always works.
    if (df["quality"] >= 7).sum() == 0:
        df.loc[0, "quality"] = 7
    return df


def test_load_data_adds_binary_target(tmp_path):
    """load_data() must add quality_binary and exclude both targets from features."""
    df = pd.DataFrame(
        {
            "fixed acidity": [7.4, 8.1],
            "volatile acidity": [0.7, 0.2],
            "citric acid": [0.0, 0.3],
            "residual sugar": [1.9, 2.1],
            "chlorides": [0.076, 0.08],
            "free sulfur dioxide": [11, 12],
            "total sulfur dioxide": [34, 35],
            "density": [0.9978, 0.998],
            "pH": [3.51, 3.2],
            "sulphates": [0.56, 0.6],
            "alcohol": [9.4, 10.2],
            "quality": [5, 7],
        }
    )
    path = tmp_path / "winequality-red.csv"
    # Match UCI's unquoted semicolon format: default to_csv would quote the
    # header (it contains ";"), which read_csv(sep=";") would then treat as one
    # giant column.
    df.to_csv(path, sep=";", index=False, quoting=csv.QUOTE_NONE, lineterminator="\n")

    loaded = wine.load_data(path)
    assert "quality_binary" in loaded.columns
    assert list(loaded["quality_binary"]) == [0, 1]  # >=7 → 1, else 0
    assert len(wine.get_features(loaded)) == 11  # excludes quality only


def test_binary_split_shapes_and_stratification(synthetic_df):
    """Split must be 80/20, scaled, and preserve class balance."""
    feats = wine.get_features(synthetic_df)
    X_tr, X_te, y_tr, y_te, X_tr_s, X_te_s, _ = wine.binary_split(synthetic_df, feats)

    assert X_tr.shape[0] == pytest.approx(0.8 * len(synthetic_df), abs=1)
    assert X_te.shape[0] == pytest.approx(0.2 * len(synthetic_df), abs=1)
    assert X_tr_s.shape[1] == len(feats) == X_te_s.shape[1]

    # Stratified: class ratio preserved between train and full set.
    full_ratio = (synthetic_df["quality"] >= 7).mean()
    train_ratio = y_tr.mean()
    assert abs(full_ratio - train_ratio) < 0.05


def test_scaler_produces_zero_mean_unit_variance(synthetic_df):
    """StandardScaler output should be centered and scaled."""
    feats = wine.get_features(synthetic_df)
    _, _, _, _, X_tr_s, _, _ = wine.binary_split(synthetic_df, feats)

    assert np.allclose(X_tr_s.mean(axis=0), 0, atol=1e-9)
    assert np.allclose(X_tr_s.std(axis=0), 1, atol=1e-9)


def test_model_trains_and_metrics_in_range(synthetic_df):
    """A simple logistic regression must train and produce valid metrics."""
    feats = wine.get_features(synthetic_df)
    X_tr, X_te, y_tr, y_te, X_tr_s, X_te_s, _ = wine.binary_split(synthetic_df, feats)

    model = LogisticRegression(max_iter=2000, random_state=42)
    model.fit(X_tr_s, y_tr)
    yp = model.predict(X_te_s)

    acc = accuracy_score(y_te, yp)
    assert 0.0 <= acc <= 1.0
    # On synthetic data even a weak model must beat random guessing.
    assert acc > 0.5


def test_evaluate_models_returns_all_metrics(synthetic_df):
    """evaluate_models must fit the full zoo and return all metric keys."""
    feats = wine.get_features(synthetic_df)
    X_tr, X_te, y_tr, y_te, X_tr_s, X_te_s, _ = wine.binary_split(synthetic_df, feats)
    models = wine.build_models()
    results = wine.evaluate_models(models, X_tr_s, y_tr, X_te_s, y_te)

    assert set(results.keys()) == set(models.keys())
    for name, metrics in results.items():
        assert set(metrics.keys()) == {"acc", "prec", "rec", "f1", "auc"}
        for value in metrics.values():
            assert 0.0 <= value <= 1.0
