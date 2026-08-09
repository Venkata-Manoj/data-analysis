"""Split-validation smoke tests for the PM2.5 Air Quality Forecasting pipeline.

Uses a small synthetic hourly dataset that mirrors the UCI PRSA schema (13 raw
columns: No, year, month, day, hour, pm2.5, DEWP, TEMP, PRES, cbwd, Iws, Is, Ir)
so tests run fast and work in CI without downloading real data. Verifies the
cleaning, feature engineering, temporal split, and regression evaluation.
"""

import numpy as np
import pandas as pd
import pm25_analysis as pm
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


@pytest.fixture
def synthetic_df():
    """Deterministic synthetic Beijing-like PM2.5 dataset covering 2010-2014."""
    rng = np.random.default_rng(7)
    hours = pd.date_range("2010-01-01", "2014-12-31 23:00", freq="h")
    n = len(hours)
    base = 60 + 50 * np.sin(2 * np.pi * hours.hour / 24) + rng.normal(0, 25, n)

    df = pd.DataFrame(
        {
            "No": np.arange(n),
            "year": hours.year,
            "month": hours.month,
            "day": hours.day,
            "hour": hours.hour,
            "pm2.5": np.abs(base),
            "DEWP": rng.uniform(-30, 25, n).round(1),
            "TEMP": rng.uniform(-15, 40, n).round(1),
            "PRES": rng.uniform(1000, 1040, n).round(1),
            "cbwd": rng.choice(["NE", "NW", "SE", "cv"], n),
            "Iws": rng.uniform(0, 20, n).round(1),
            "Is": rng.integers(0, 2, n),
            "Ir": rng.uniform(0, 500, n).round(1),
        }
    )
    # Inject some NaN pm2.5 values so interpolation logic is exercised.
    df.loc[df.index[::50], "pm2.5"] = np.nan
    return df


def test_clean_and_preprocess_builds_datetime_index(synthetic_df):
    """Cleaning must drop 'No', build a sorted datetime index, encode wind."""
    clean = pm.clean_and_preprocess(synthetic_df)

    assert "No" not in clean.columns
    assert isinstance(clean.index, pd.DatetimeIndex)
    assert clean.index.is_monotonic_increasing
    assert "cbwd" not in clean.columns
    # 4 wind directions → 4 dummy columns.
    wind_cols = [c for c in clean.columns if c.startswith("wind_")]
    assert len(wind_cols) == 4


def test_clean_fills_nan_pm25(synthetic_df):
    """NaN pm2.5 values must be gone after interpolation + median fill."""
    clean = pm.clean_and_preprocess(synthetic_df)
    assert clean["pm2.5"].isna().sum() == 0


def test_engineer_features_adds_lag_and_rolling(synthetic_df):
    """Feature engineering must add time, lag, and rolling columns."""
    clean = pm.clean_and_preprocess(synthetic_df)
    feat = pm.engineer_features(clean)

    expected_prefixes = ["hour_sin", "hour_cos", "month_sin", "month_cos", "day_of_week", "is_weekend", "quarter"]
    for col in expected_prefixes:
        assert col in feat.columns

    assert "pm25_lag_1h" in feat.columns and "pm25_lag_72h" in feat.columns
    assert "pm25_roll_mean_24h" in feat.columns and "pm25_roll_std_48h" in feat.columns

    # Lag/rolling creation drops leading NaN rows.
    assert len(feat) < len(clean)
    assert not feat.isna().any().any()


def test_temporal_split_keeps_chronology(synthetic_df):
    """Time-based split: all train rows must precede all test rows."""
    clean = pm.clean_and_preprocess(synthetic_df)
    feat = pm.engineer_features(clean)
    train_df, test_df, feature_cols, target = pm.temporal_split(feat)

    assert target == "pm2.5"
    assert len(feature_cols) == feat.shape[1] - 1
    assert train_df.index.max() < test_df.index.min()
    assert "pm2.5" not in feature_cols


def test_temporal_split_respects_cutoff(synthetic_df):
    """The 2014-01-01 cutoff must split the data as documented."""
    clean = pm.clean_and_preprocess(synthetic_df)
    feat = pm.engineer_features(clean)
    train_df, test_df, _, _ = pm.temporal_split(feat, cutoff="2014-01-01")

    assert train_df.index.max() < pd.Timestamp("2014-01-01")
    assert test_df.index.min() >= pd.Timestamp("2014-01-01")


def test_regression_pipeline_trains_and_evaluates(synthetic_df):
    """A linear regression on the engineered features must produce metrics."""
    clean = pm.clean_and_preprocess(synthetic_df)
    feat = pm.engineer_features(clean)
    train_df, test_df, feature_cols, target = pm.temporal_split(feat)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols])
    X_test = scaler.transform(test_df[feature_cols])

    model = LinearRegression()
    model.fit(X_train, train_df[target])
    preds = model.predict(X_test)

    metrics = pm.evaluate_regression(test_df[target], preds)
    assert set(metrics.keys()) == {"MAE", "RMSE", "R2", "MAPE"}
    assert metrics["MAE"] >= 0
    assert metrics["RMSE"] >= 0
    assert metrics["R2"] <= 1.0
    # Synthetic periodic signal is strongly predictable.
    assert metrics["R2"] > 0.5
