# 🌬️ PM2.5 Air Quality Forecasting

**Project 5** — A multivariate time series forecasting pipeline that predicts hourly PM2.5 concentration in Beijing using lag features, rolling statistics, and temporal features. Four models compared on a strict time-based train/test split.

| Detail | Value |
|--------|-------|
| **Dataset** | [UCI Beijing PM2.5](https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data) (2010–2014, 43,824 hourly readings) |
| **Target** | PM2.5 concentration (μg/m³) |
| **Best Model** | Linear Regression (tied with Ridge) |
| **Test R²** | **0.9465** |
| **Prediction Error** | ~11.8 MAE (μg/m³) |

## What I Built

An end-to-end time series forecasting pipeline that:

1. **Downloads & cleans** the UCI Beijing PM2.5 dataset — handles 2,067 NaN values with temporal interpolation
2. **Engineers** 32 features:
   - 7 lag features (1h, 3h, 6h, 12h, 24h, 48h, 72h)
   - 8 rolling window statistics (mean + std at 6/12/24/48h)
   - 6 time-based features (hour sine/cosine, month sine/cosine, day of week, is_weekend)
   - Wind direction one-hot encoding
3. **Splits** strictly by time — train on 2010–2013, test on 2014 (simulates real-world forecasting)
4. **Trains & compares** 4 models — Linear Regression, Ridge (α=10), Random Forest (200 trees), XGBoost (300 trees)
5. **Evaluates** with MAE, RMSE, R², and MAPE
6. **Visualizes** 8 charts — time series trends, predicted vs actual, residuals distribution, feature importance, hourly & weekly seasonality patterns

## Results

| Model | MAE | RMSE | R² | MAPE |
|-------|-----|------|----|------|
| **Linear Regression** | **11.78** | **21.56** | **0.9465** | **23.56%** |
| Ridge (α=10) | 11.78 | 21.56 | 0.9465 | 23.57% |
| Random Forest | 12.05 | 23.42 | 0.9369 | 21.81% |
| XGBoost | 12.59 | 24.06 | 0.9334 | 22.50% |

**Key insight:** Linear models matched tree-based models because PM2.5 forecasting relies heavily on recent lag values — a linear relationship that's well captured by simple regression when given engineered features. Feature engineering quality matters more than model complexity here.

## Charts Gallery

| Chart | File |
|-------|------|
| Training Time Series (2010–2013) | `charts/01-training-timeseries.png` |
| Predictions vs Actual (2014) | `charts/02-predictions-vs-actual.png` |
| Scatter: Predicted vs Actual | `charts/03-scatter-predicted-vs-actual.png` |
| Model Comparison Bar Chart | `charts/04-model-comparison.png` |
| Feature Importance (XGBoost) | `charts/05-feature-importance.png` |
| Residuals Distribution | `charts/06-residuals-distribution.png` |
| Average Hourly Pattern | `charts/07-hourly-pattern.png` |
| Average Weekly Pattern | `charts/08-weekly-pattern.png` |

## Skills Demonstrated

- **Time series forecasting** — temporal train/test split, lag feature engineering, rolling statistics
- **Feature engineering** — domain-specific feature creation for temporal data
- **Model comparison** — 4 models across linear and non-linear families
- **Temporal validation** — time-based (not random) train/test split to simulate real forecasting
- **Seasonality analysis** — hourly and weekly patterns in air quality
- **Residual analysis** — distribution of prediction errors across models
- **Feature importance** — interpreting which temporal features drive predictions

## Usage

```bash
cd pm25-air-quality-forecasting
pip install -r requirements.txt
python analysis.py
```

The data auto-downloads on first run. All charts land in `charts/`, metrics in `outputs/`.
