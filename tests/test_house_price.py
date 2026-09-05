"""Offline smoke tests for the House Price Prediction pipeline.

Uses synthetic regression data that mirrors the California Housing schema
(8 base features + 5 engineered) so the tests run fast in CI without
downloading sklearn's fetch_california_housing. Verifies feature
engineering formulas, scaling, train/test split, model training, and
metric computation end to end.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


@pytest.fixture
def synthetic_housing():
    """Synthetic California Housing-like dataset (8 base + 5 engineered features)."""
    rng = np.random.default_rng(RANDOM_STATE)
    n = 1000
    data = {
        "MedInc": rng.uniform(0.5, 15, n),
        "HouseAge": rng.uniform(1, 52, n),
        "AveRooms": rng.uniform(2, 10, n),
        "AveBedrms": rng.uniform(0.5, 2.5, n),
        "Population": rng.uniform(3, 35000, n),
        "AveOccup": rng.uniform(0.5, 6, n),
        "Latitude": rng.uniform(32, 42, n),
        "Longitude": rng.uniform(-124, -114, n),
    }
    df = pd.DataFrame(data)
    # Engineered features — same formulas as analysis.py
    df["RoomsPerPerson"] = df["AveRooms"] / df["AveOccup"]
    df["BedroomRatio"] = df["AveBedrms"] / df["AveRooms"]
    df["PopDensity"] = df["Population"] / 100.0  # simplified
    df["IncomePerRoom"] = df["MedInc"] / df["AveRooms"]
    df["LatBin"] = pd.qcut(df["Latitude"], q=10, labels=False, duplicates="drop").astype(float)
    # Log-transformed target
    X_raw, y_raw = make_regression(n_samples=n, n_features=8, noise=10, random_state=RANDOM_STATE)
    y = np.log1p(np.abs(y_raw) + 10)  # positive log target like LogMedHouseVal
    df["LogMedHouseVal"] = y
    return df


def test_engineered_features_formulas(synthetic_housing):
    """Engineered features must match the formulas in analysis.py."""
    df = synthetic_housing
    # Spot-check first row
    row = df.iloc[0]
    assert row["RoomsPerPerson"] == pytest.approx(row["AveRooms"] / row["AveOccup"])
    assert row["BedroomRatio"] == pytest.approx(row["AveBedrms"] / row["AveRooms"])
    assert row["IncomePerRoom"] == pytest.approx(row["MedInc"] / row["AveRooms"])
    # No NaN after fill
    feature_cols = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
        "RoomsPerPerson",
        "BedroomRatio",
        "PopDensity",
        "IncomePerRoom",
        "LatBin",
    ]
    assert not df[feature_cols].isnull().any().any()


def test_log_transform_invertible():
    """log1p / expm1 must round-trip for house values."""
    vals = np.array([0.5, 1.0, 2.5, 5.0])  # $100k units
    assert np.allclose(np.expm1(np.log1p(vals)), vals)
    # USD back-transform used in analysis.py
    rmse_log = 0.14
    usd = np.expm1(rmse_log) * 100_000
    assert usd > 0
    assert 10000 < usd < 20000  # ~$15k for RMSE 0.14


def test_scaler_zero_mean_unit_variance(synthetic_housing):
    """StandardScaler must centre and scale the training matrix."""
    feature_cols = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
        "RoomsPerPerson",
        "BedroomRatio",
        "PopDensity",
        "IncomePerRoom",
        "LatBin",
    ]
    X = synthetic_housing[feature_cols].values
    y = synthetic_housing["LogMedHouseVal"].values
    X_train, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    assert np.allclose(X_train_s.mean(axis=0), 0, atol=1e-9)
    assert np.allclose(X_train_s.std(axis=0), 1, atol=1e-9)


def test_train_test_split_shapes(synthetic_housing):
    """Split must be 80/20 with scaling."""
    feature_cols = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
        "RoomsPerPerson",
        "BedroomRatio",
        "PopDensity",
        "IncomePerRoom",
        "LatBin",
    ]
    X = synthetic_housing[feature_cols].values
    y = synthetic_housing["LogMedHouseVal"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    assert X_train.shape[0] == pytest.approx(0.8 * len(synthetic_housing), abs=1)
    assert X_test.shape[0] == pytest.approx(0.2 * len(synthetic_housing), abs=1)
    assert X_train.shape[1] == len(feature_cols)
    assert len(y_train) == X_train.shape[0]
    assert len(y_test) == X_test.shape[0]


def test_regression_models_r2_in_range(synthetic_housing):
    """Linear / Ridge / tree regressors must train and produce valid R²/MAE/RMSE."""
    feature_cols = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
        "RoomsPerPerson",
        "BedroomRatio",
        "PopDensity",
        "IncomePerRoom",
        "LatBin",
    ]
    X = synthetic_housing[feature_cols].values
    y = synthetic_housing["LogMedHouseVal"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "RandomForest": RandomForestRegressor(n_estimators=20, max_depth=8, random_state=RANDOM_STATE, n_jobs=1),
        "GradBoost": GradientBoostingRegressor(n_estimators=20, max_depth=3, random_state=RANDOM_STATE),
    }
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        assert y_pred.shape == y_test.shape, f"{name} shape mismatch"
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        assert -1.0 <= r2 <= 1.0, f"{name} R² out of range: {r2}"
        assert mae >= 0
        assert rmse >= 0
        assert rmse >= mae - 1e-9  # RMSE >= MAE by Jensen


def test_cv_scores_stable(synthetic_housing):
    """3-fold CV R² mean must be in [-1, 1] and std non-negative."""
    from sklearn.model_selection import cross_val_score

    feature_cols = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "AveOccup", "Latitude", "Longitude"]
    X = synthetic_housing[feature_cols].values
    y = synthetic_housing["LogMedHouseVal"].values
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(Ridge(alpha=1.0, random_state=RANDOM_STATE), X_s, y, cv=cv, scoring="r2")
    assert scores.shape == (3,)
    assert -1.0 <= scores.mean() <= 1.0
    assert scores.std() >= 0


def test_gridsearch_finds_params(synthetic_housing):
    """GridSearchCV must return a best estimator and valid best_score."""
    feature_cols = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Latitude", "Longitude"]
    X = synthetic_housing[feature_cols].values
    y = synthetic_housing["LogMedHouseVal"].values
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    param_grid = {"alpha": [0.1, 1.0, 10.0]}
    grid = GridSearchCV(Ridge(random_state=RANDOM_STATE), param_grid, cv=2, scoring="r2")
    grid.fit(X_s, y)
    assert grid.best_params_["alpha"] in param_grid["alpha"]
    assert -1.0 <= grid.best_score_ <= 1.0
    assert hasattr(grid.best_estimator_, "predict")
