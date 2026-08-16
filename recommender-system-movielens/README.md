# Project 8: Recommender System - MovieLens 100k

A collaborative-filtering recommender built on the classic MovieLens 100k dataset
(100,000 ratings from 943 users across 1,683 movies). This is the first project in
the portfolio to tackle **ranking and personalization** - a different family of
problems from the regression, classification, NLP, computer-vision, clustering, and
topic-modeling work in Projects 1-7.

## What it does

- Loads and explores the MovieLens 100k ratings + movie metadata.
- Compares four recommendation approaches on a held-out 20% test split:

| Model | Idea |
|-------|------|
| **Baseline (bias)** | Global mean + per-user and per-item bias offsets |
| **UserKNN** | Centered user-user cosine similarity, k=40 nearest neighbors |
| **ItemKNN** | Centered item-item cosine similarity, k=40 nearest neighbors |
| **SVD** | TruncatedSVD (50 latent components) on the centered rating matrix |

- Evaluates every model with **RMSE** and **MAE** (lower is better).
- Generates 8 charts (rating distribution, movie popularity, avg-vs-count, genre
  spread, model comparison, SVD explained variance, predicted-vs-actual, and a
  Top-10 personalized recommendation demo).
- Ships a personalized **Top-N** recommendation helper (`recommend_top_k`) that
  produces real movie picks for any user, excluding movies they already rated.

## Dataset

- **MovieLens 100k** (GroupLens). Auto-downloads to `data/` on first run
  (`data/` is gitignored, so the repo stays light).
- Sparsity: ~93.7% of user-movie pairs are unrated - the core challenge of CF.

## Techniques demonstrated

- Collaborative filtering (memory-based user/item KNN)
- Matrix factorization (SVD / latent factor models)
- Bias baseline modeling
- RMSE / MAE evaluation
- Cold-start intuition (handled gracefully via global/mean fallbacks)
- Reproducible seeded train/test split

## How to run

```bash
# from the repo root, with the project venv activated
cd recommender-system-movielens
python -m pip install -r requirements.txt
python analysis.py
```

Outputs land in `charts/` (8 PNGs) and `outputs/results_summary.md`.

## Results

See `outputs/results_summary.md` (regenerated each run). Headline: on this dense,
forgiving benchmark all four models land within a narrow error band, with
matrix factorization (SVD) and item-based CF edging out the naive bias baseline -
the real gap appears on sparser, colder data.

## Files

- `analysis.py` - importable functions + `main()` guard (follows the portfolio
  convention so it is testable).
- `tests/test_recommender.py` - unit tests on synthetic data (no download needed).
- `charts/`, `outputs/`, `data/` - generated / cached (data is gitignored).

## Future improvements

- Add an explicit cold-start track (popularity / content-based fallback).
- Try ALS or a neural matrix factorization.
- Tune k and latent dimension, or use cross-validated RMSE for model selection.
- Build a small Streamlit demo reusing `recommend_top_k`.
