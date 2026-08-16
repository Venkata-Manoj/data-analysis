"""Smoke tests for the MovieLens Recommender System pipeline.

Uses a small, deterministic synthetic rating matrix so the tests run fast and
need no download. Verifies the split, baseline, KNN, biased SVD, metrics, and
the Top-N recommender end to end.
"""

import numpy as np
import pandas as pd
import pytest
import recommender_analysis as rec


@pytest.fixture
def synthetic_ratings():
    """Deterministic 15-user x 10-movie rating matrix with a clear pattern."""
    rng = np.random.default_rng(7)
    n_users, n_movies = 15, 10
    # Latent pattern: users split into two taste groups; movies into two types.
    user_group = rng.integers(0, 2, n_users)
    movie_group = rng.integers(0, 2, n_movies)
    base = np.where(user_group[:, None] == movie_group[None, :], 4.5, 2.5)
    noise = rng.normal(0, 0.4, size=base.shape)
    R = np.clip(np.round(base + noise), 1, 5).astype(float)
    # Mask ~40% as unrated to mimic sparsity.
    mask = rng.random(R.shape) < 0.4
    R[mask] = np.nan
    df = pd.DataFrame(
        {
            "user_id": np.repeat(np.arange(n_users), n_movies),
            "movie_id": np.tile(np.arange(n_movies), n_users),
            "rating": R.flatten(),
            "timestamp": 0,
        }
    ).dropna(subset=["rating"])
    df["rating"] = df["rating"].astype(int)
    df["user_id"] = df["user_id"].astype(int)
    df["movie_id"] = df["movie_id"].astype(int)
    return df


def test_train_test_split_sizes_and_disjoint(synthetic_ratings):
    """Holdout split must be ~80/20 and contain no overlapping rows."""
    train, test = rec.train_test_split_ratings(synthetic_ratings, random_state=1)
    total = len(synthetic_ratings)
    assert len(train) == pytest.approx(0.8 * total, abs=1)
    assert len(test) == pytest.approx(0.2 * total, abs=1)
    merged = pd.merge(train, test, on=["user_id", "movie_id", "rating"], how="inner")
    assert len(merged) == 0  # no row appears in both


def test_metrics_in_range():
    """RMSE and MAE must be non-negative and match trivial cases."""
    y_true = np.array([1.0, 3.0, 5.0])
    y_pred = np.array([1.0, 3.0, 5.0])
    assert rec.rmse(y_true, y_pred) == pytest.approx(0.0)
    assert rec.mae(y_true, y_pred) == pytest.approx(0.0)
    # One unit off everywhere -> RMSE = MAE = 1.0
    assert rec.rmse(y_true, y_true + 1) == pytest.approx(1.0)
    assert rec.mae(y_true, y_true + 1) == pytest.approx(1.0)


def test_baseline_predict_in_range(synthetic_ratings):
    """Baseline predictions must stay within the valid rating bounds."""
    train, test = rec.train_test_split_ratings(synthetic_ratings)
    gm, ub, ib = rec.fit_baseline(train)
    preds = rec.baseline_predict(gm, ub, ib, test["user_id"].to_numpy(), test["movie_id"].to_numpy())
    assert preds.min() >= 1.0
    assert preds.max() <= 5.0


def test_knn_runs_and_predicts(synthetic_ratings):
    """Both KNN variants must return one prediction per test row, in range."""
    train, test = rec.train_test_split_ratings(synthetic_ratings)
    R, user_means, item_means = rec.build_user_item(train)
    R = R.reindex(
        index=rec.build_user_item(synthetic_ratings)[0].index,
        columns=rec.build_user_item(synthetic_ratings)[0].columns,
    )
    tu = test["user_id"].to_numpy()
    ti = test["movie_id"].to_numpy()
    for fn, means in ((rec.user_knn_predict, user_means), (rec.item_knn_predict, item_means)):
        preds = fn(R, means, tu, ti, k=5)
        assert len(preds) == len(test)
        assert np.all(preds >= 1.0)
        assert np.all(preds <= 5.0)


def test_biased_svd_beats_baseline_on_pattern(synthetic_ratings):
    """On a matrix with a strong latent pattern, biased SVD should beat the
    naive baseline in RMSE (this is the project's core claim)."""
    train, test = rec.train_test_split_ratings(synthetic_ratings, random_state=3)
    R_full, _, _ = rec.build_user_item(synthetic_ratings)
    R, _, _ = rec.build_user_item(train)
    R = R.reindex(index=R_full.index, columns=R_full.columns)

    gm, ub, ib = rec.fit_baseline(train)
    tu = test["user_id"].to_numpy()
    ti = test["movie_id"].to_numpy()
    y_true = test["rating"].to_numpy()

    base_rmse = rec.rmse(y_true, rec.baseline_predict(gm, ub, ib, tu, ti))
    svd_rmse = rec.rmse(y_true, rec.svd_predict(R, tu, ti, train=train, n_components=4, n_epochs=40))
    assert svd_rmse < base_rmse


def test_recommend_top_k_excludes_rated(synthetic_ratings):
    """Top-N recommendations must omit movies the user already rated."""
    R_full, _, _ = rec.build_user_item(synthetic_ratings)
    # Fit a quick biased-SVD matrix on the full set for demo purposes.
    matrix = rec.svd_predict_matrix(R_full, synthetic_ratings, n_components=4, n_epochs=30)
    user_id = int(R_full.notna().sum(axis=1).idxmax())
    rated = set(R_full.loc[user_id].dropna().index.tolist())
    recs = rec.recommend_top_k(matrix, R_full, user_id, _movies(synthetic_ratings), k=2)
    # The most-active user here has only 2 unrated movies, so k is capped to that
    # and NO rated movie should ever be surfaced.
    assert len(recs) == 2
    for movie_id, _title, score in recs:
        assert movie_id not in rated
        assert 1.0 <= score <= 5.0


def _movies(ratings):
    """Minimal movie lookup matching recommend_top_k's title_map contract."""
    ids = sorted(ratings["movie_id"].unique())
    return pd.DataFrame({"movie_id": ids, "title": [f"Movie {i}" for i in ids]})
