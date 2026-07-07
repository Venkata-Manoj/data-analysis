# 🏠 California House Price Prediction

A comprehensive regression analysis that predicts median house values across California census block groups using demographic, geographic, and housing features. Five models compared, tuned, and evaluated with full residual analysis and feature importance interpretation.

## Project Highlights

| Detail | Value |
|--------|-------|
| **Dataset** | California Housing (sklearn) — 20,640 census block groups |
| **Target** | Median house value (log-transformed for normality) |
| **Best Model** | Gradient Boosting (tuned) |
| **Test R²** | **0.8363** |
| **Prediction Error** | ~$10,707 MAE (back-transformed to USD) |

## What I Built

An end-to-end regression pipeline that:
1. **Explores** the data — distributions, correlations, geospatial patterns
2. **Engineers** features — interaction terms (income/room, bedroom ratio), location bins
3. **Trains & compares** 5 models — Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting
4. **Tunes** the winner with grid search (3-fold CV on 50% sample for speed)
5. **Analyzes** residuals — normality check, heteroscedasticity, actual-vs-predicted
6. **Interprets** features — the engineered `IncomePerRoom` feature dominates (43.8% importance)
7. **Plots learning curves** — confirms the model benefits from more data

## Skills Demonstrated

- **Regression modeling** — multiple algorithms, hyperparameter tuning
- **Feature engineering** — interaction terms, binning, log transforms
- **Cross-validation** — K-fold CV for robust model selection
- **Residual analysis** — diagnosing model assumptions and fit quality
- **Feature importance** — tree-based importance for interpretability
- **Geospatial EDA** — mapping prices across California's geography
- **Model interpretation** — translating log-space metrics back to dollars

## Results

```
Model                          R²      RMSE (log)  MAE (log)
──────────────────────────────────────────────────────────────
Linear Regression             0.6721    0.2033      0.1525
Ridge (alpha=1.0)             0.6721    0.2033      0.1525
Lasso (alpha=0.001)           0.6707    0.2037      0.1532
Random Forest (50 trees)      0.8149    0.1527      0.1073
Gradient Boosting (untuned)   0.8117    0.1541      0.1104
⭐ Gradient Boosting (tuned)  0.8363    0.1436      0.1017
```

**Back-transformed to dollars:** The tuned model predicts house values with a typical error of **~$10,707** (MAE) and RMSE of **~$15,448**.

## Key Insight

The single most predictive feature is `IncomePerRoom` — an interaction I engineered by dividing median income by average rooms. This captures neighborhood affluence density more effectively than income alone (which came in 2nd at 14% importance). Geographic features (latitude/longitude) rank 3rd and 4th, confirming location is a fundamental driver of California housing prices.

## Files

| File | Purpose |
|------|---------|
| `analysis.py` | Full analysis pipeline (runs end-to-end) |
| `requirements.txt` | Python dependencies |
| `outputs/results.json` | All metrics and comparisons in JSON |
| `outputs/model_comparison.csv` | Model scores table |
| `outputs/feature_importance.csv` | Feature importance ranking |
| `charts/*.png` | 9 publication-quality visualizations |

## Quick Start

```bash
pip install -r requirements.txt
python analysis.py
```

## Future Improvements

- Try XGBoost/LightGBM for potentially higher accuracy
- Add polynomial features for non-linear relationships
- Implement spatial clustering of regions before modeling
- Build a Streamlit dashboard for interactive exploration
- Add automated hyperparameter optimization with Optuna
