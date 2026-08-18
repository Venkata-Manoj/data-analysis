# Project 9: Credit Card Fraud Detection - Anomaly Detection

The portfolio's first **anomaly detection / extreme-imbalance** project. It
compares **unsupervised** anomaly detectors - which need no fraud labels at
training time - against a **supervised** baseline that uses labels, on the
canonical Credit Card Fraud Detection dataset (284,807 transactions, only 492
fraud = **0.172%**). This is a different problem family from the regression,
classification, NLP, computer-vision, clustering, topic-modeling, and
recommender work in Projects 1-8.

## What it does

- Loads and explores the fraud dataset (severe class imbalance).
- Standardizes features and runs a stratified 80/20 train/test split.
- Trains three unsupervised detectors on the (mostly normal) training data:

| Model | Idea |
|-------|------|
| **Isolation Forest** | Anomalies isolated quickly in random trees |
| **LOF** | Local density deviation from neighbors (novelty mode) |
| **One-Class SVM** | Hypersphere boundary around the normal bulk |

- Trains a supervised **class-weighted Logistic Regression** using fraud labels
  as the "if you had labels" ceiling.
- Ranks every method with **ROC-AUC**, **Average Precision**, and
  **Precision@k** (k = the fraud alert budget).
- **Cost-sensitive alert-budget optimization (new):** converts the detector
  scores into an actionable policy by sweeping fixed manual-review volumes
  (0.1%-10% of transactions) and reporting precision/recall at each operating
  point. This answers "if a review team can check X% of volume, how much fraud
  do we catch at what precision?" - the question a lean fraud team actually
  operates on. Writes `outputs/alert_budget.md`.
- Generates 8 charts (imbalance, amount distribution, feature boxplots, PCA
  variance, ROC curves, PR curves, Precision@k, model comparison).
- Writes `outputs/results_summary.md` with the headline metrics.

## Dataset

- **Credit Card Fraud Detection** (OpenML `data_id=1597`, ULB/MLG).
  Auto-downloads to `data/` on first run (`data/` is gitignored, so the repo
  stays light). 284,807 rows, 30 columns (28 PCA components `V1..V28`, `Time`,
  `Amount`), binary label.

## Techniques demonstrated

- Unsupervised anomaly detection (Isolation Forest, LOF, One-Class SVM)
- Supervised imbalanced classification (class-weighted Logistic Regression)
- Imbalance-appropriate evaluation (ROC-AUC, PR-AUC, Precision@k)
- PCA for dimensionality / variance analysis
- Stratified train/test split under extreme imbalance

## How to run

```bash
# from the repo root, with the project venv activated
cd anomaly-detection-fraud
python -m pip install -r requirements.txt
python analysis.py
```

Outputs land in `charts/` (8 PNGs), `outputs/results_summary.md`, and
`outputs/alert_budget.md` (the cost-sensitive operating-point sweep).

## Results

See `outputs/results_summary.md` (regenerated each run). Headline: unsupervised
detectors surface fraud using only the *shape* of normal activity - no labels
required - while the supervised model shows the ceiling when labels exist. Both
are valuable: the former for cold-start monitoring, the latter for tuned
detection.

## Files

- `analysis.py` - importable functions + `main()` guard (testable convention).
- `tests/test_anomaly.py` - unit tests on synthetic data (no download needed;
  CI runs these only).
- `charts/`, `outputs/`, `data/` - generated / cached (data is gitignored).

## Future improvements

- ~~Add cost-sensitive thresholding / alert-budget optimization (maximize
  Precision@k for a fixed review team size).~~ **Implemented** - see
  `cost_sensitive_threshold`, `alert_budget_sweep`, and `outputs/alert_budget.md`.
- Try autoencoders / deep SVDD for learned anomaly representations.
- Use the `Time` feature for temporal drift / concept-shift analysis.
- Compare against gradient-boosted supervised models (XGBoost, LightGBM).
