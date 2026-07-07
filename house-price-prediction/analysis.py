# House Price Prediction — California Housing
#
# A complete regression pipeline: EDA → feature engineering → model comparison
# → hyperparameter tuning → evaluation → interpretation.
#
# Dataset: California Housing (sklearn) — 20,640 census block groups from 1990
# Target: log(median house value) in USD
#
# Skills demonstrated: regression, feature engineering, cross-validation,
# ensemble methods, geospatial analysis, model interpretation.

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os, json, warnings
from datetime import datetime

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, KFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

OUTPUTS = os.path.join(os.path.dirname(__file__) or ".", "outputs")
CHARTS = os.path.join(os.path.dirname(__file__) or ".", "charts")
os.makedirs(OUTPUTS, exist_ok=True)
os.makedirs(CHARTS, exist_ok=True)

print("=" * 65)
print("  CALIFORNIA HOUSING — REGRESSION ANALYSIS")
print("=" * 65)

# ── 1. Load Data ──────────────────────────────────────────────────────────────

print("\n[1] Loading California Housing dataset...")
housing = fetch_california_housing(as_frame=True)
df = housing.data.copy()
df["MedHouseVal"] = housing.target  # target in $100k units
df["LogMedHouseVal"] = np.log1p(df["MedHouseVal"])  # log-transform target

print(f"  Samples: {df.shape[0]:,}")
print(f"  Features: {df.shape[1] - 2}")
print(f"  Missing values: {df.isnull().sum().sum()}")
print(f"\n  Feature descriptions:")
for name, desc in zip(housing.feature_names, [
    "Median income in block group (scaled)",
    "Median house age in block group",
    "Average rooms per household",
    "Average bedrooms per household",
    "Block group population",
    "Average household occupancy",
    "Block group latitude",
    "Block group longitude",
]):
    print(f"    {name:15s} — {desc}")

df.info()
df.describe().to_csv(os.path.join(OUTPUTS, "summary_stats.csv"))

# ── 2. Exploratory Data Analysis ──────────────────────────────────────────────

print("\n[2] Exploratory Data Analysis...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("California Housing — Distributions & Correlations", fontsize=16)

features_plot = ["MedInc", "HouseAge", "AveRooms", "AveOccup", "Latitude", "Longitude"]
targets_plot = ["MedHouseVal", "LogMedHouseVal"]

for ax, feat in zip(axes.flat, features_plot):
    ax.hist(df[feat], bins=50, edgecolor="white", alpha=0.7)
    ax.set_title(feats := feat, fontsize=11)
    ax.set_xlabel("")
plt.tight_layout()
plt.savefig(os.path.join(CHARTS, "feature_distributions.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Feature distributions chart saved")

# Correlation matrix
corr = df[list(housing.feature_names) + ["MedHouseVal"]].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            square=True, linewidths=0.5)
plt.title("Feature Correlation Matrix", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS, "correlation_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Correlation heatmap saved")

# Target distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.hist(df["MedHouseVal"], bins=50, edgecolor="white", alpha=0.7, color="steelblue")
ax1.set_title("Median House Value ($100k)", fontsize=13)
ax1.set_xlabel("Median House Value")
ax1.set_ylabel("Frequency")
ax2.hist(df["LogMedHouseVal"], bins=50, edgecolor="white", alpha=0.7, color="coral")
ax2.set_title("Log-Transformed Median House Value", fontsize=13)
ax2.set_xlabel("log(MedHouseVal + 1)")
ax2.set_ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(CHARTS, "target_distribution.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Target distribution chart saved")

# Geospatial scatter
plt.figure(figsize=(10, 8))
sc = plt.scatter(
    df["Longitude"], df["Latitude"], c=df["MedHouseVal"],
    cmap="viridis", alpha=0.4, s=5, edgecolors="none"
)
plt.colorbar(sc, label="Median House Value ($100k)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("California Housing Prices — Geospatial View", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS, "geospatial_prices.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Geospatial price map saved")

# Income vs House Value scatter
plt.figure(figsize=(8, 6))
plt.scatter(df["MedInc"], df["MedHouseVal"], alpha=0.3, s=3, c="steelblue")
plt.xlabel("Median Income (scaled)")
plt.ylabel("Median House Value ($100k)")
plt.title("Income vs House Price", fontsize=13)
# Add trend line
z = np.polyfit(df["MedInc"], df["MedHouseVal"], 1)
p = np.poly1d(z)
x_range = np.linspace(df["MedInc"].min(), df["MedInc"].max(), 100)
plt.plot(x_range, p(x_range), "r--", alpha=0.8, linewidth=2)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS, "income_vs_price.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Income vs price scatter saved")

# ── 3. Feature Engineering ────────────────────────────────────────────────────

print("\n[3] Feature Engineering...")

# Add interaction features
df["RoomsPerPerson"] = df["AveRooms"] / df["AveOccup"]
df["BedroomRatio"] = df["AveBedrms"] / df["AveRooms"]
df["PopDensity"] = df["Population"] / (df["Latitude"].nunique() * df["Longitude"].nunique())
df["IncomePerRoom"] = df["MedInc"] / df["AveRooms"]

# Location-based features (coarse region encoding)
df["LatBin"] = pd.qcut(df["Latitude"], q=10, labels=False, duplicates="drop")
df["LonBin"] = pd.qcut(df["Longitude"], q=10, labels=False, duplicates="drop")

print("  ✓ Added 5 engineered features")
print(f"  Feature matrix: {df.shape[1]} total features")

# ── 4. Train/Test Split & Scaling ─────────────────────────────────────────────

print("\n[4] Preparing train/test split...")

feature_cols = (
    list(housing.feature_names)
    + ["RoomsPerPerson", "BedroomRatio", "PopDensity", "IncomePerRoom",
       "LatBin", "LonBin"]
)
# Handle any NaN from qcut
df[feature_cols] = df[feature_cols].fillna(0)

X = df[feature_cols].values
y = df["LogMedHouseVal"].values  # log-transformed target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"  Train: {X_train.shape[0]:,} samples")
print(f"  Test:  {X_test.shape[0]:,} samples")
print(f"  Features: {X_train.shape[1]}")

# ── 5. Model Training & Comparison ────────────────────────────────────────────

print("\n[5] Training & comparing regression models...")

models = {
    "Linear Regression": LinearRegression(),
    "Ridge (alpha=1.0)": Ridge(alpha=1.0, random_state=RANDOM_STATE),
    "Lasso (alpha=0.001)": Lasso(alpha=0.001, random_state=RANDOM_STATE),
    "Random Forest (50)": RandomForestRegressor(
        n_estimators=50, max_depth=12, n_jobs=-1, random_state=RANDOM_STATE
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=RANDOM_STATE
    ),
}

results = []
cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
best_model = None
best_score = -np.inf

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train,
                                cv=cv, scoring="r2", n_jobs=-1)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)

    results.append({
        "Model": name,
        "R² Score": round(r2, 4),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "CV R² Mean": round(cv_scores.mean(), 4),
        "CV R² Std": round(cv_scores.std(), 4),
    })

    print(f"  {name:30s}  R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}  "
          f"CV R²={cv_scores.mean():.4f}±{cv_scores.std():.4f}")

    if cv_scores.mean() > best_score:
        best_score = cv_scores.mean()
        best_model = (name, model)

results_df = pd.DataFrame(results).sort_values("R² Score", ascending=False)
results_df.to_csv(os.path.join(OUTPUTS, "model_comparison.csv"), index=False)
print(f"\n  ⭐ Best model: {best_model[0]} (CV R² = {best_score:.4f})")

# Model comparison chart
plt.figure(figsize=(10, 6))
bars = plt.barh(results_df["Model"], results_df["R² Score"],
                color=sns.color_palette("viridis", len(results_df)))
plt.xlabel("R² Score (test set)")
plt.title("Regression Model Comparison — R² Score", fontsize=14)
plt.xlim(0, 1)
for bar, score in zip(bars, results_df["R² Score"]):
    plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
             f"{score:.4f}", va="center", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS, "model_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Model comparison chart saved")

# ── 6. Best Model: Hyperparameter Tuning ──────────────────────────────────────

print(f"\n[6] Tuning best model ({best_model[0]})...")

if "Random Forest" in best_model[0]:
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 15],
        "min_samples_split": [5],
    }
    base_model = RandomForestRegressor(n_jobs=-1, random_state=RANDOM_STATE)
elif "Gradient Boosting" in best_model[0]:
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
    }
    base_model = GradientBoostingRegressor(random_state=RANDOM_STATE)
else:
    param_grid = {"alpha": [0.01, 0.1, 1.0, 10.0]}
    base_model = Ridge(random_state=RANDOM_STATE)

# Speed up tuning by using a subset for grid search
X_gs, _, y_gs, _ = train_test_split(
    X_train_scaled, y_train, train_size=0.5, random_state=RANDOM_STATE
)
grid = GridSearchCV(
    base_model, param_grid, cv=3, scoring="r2", n_jobs=-1, verbose=0
)
grid.fit(X_gs, y_gs)

print(f"  Best params: {grid.best_params_}")
print(f"  Best CV R²: {grid.best_score_:.4f}")

tuned_model = grid.best_estimator_
y_pred_tuned = tuned_model.predict(X_test_scaled)
r2_tuned = r2_score(y_test, y_pred_tuned)
rmse_tuned = np.sqrt(mean_squared_error(y_test, y_pred_tuned))
mae_tuned = mean_absolute_error(y_test, y_pred_tuned)

print(f"  Tuned R²:  {r2_tuned:.4f}")
print(f"  Tuned RMSE: {rmse_tuned:.4f}")
print(f"  Tuned MAE:  {mae_tuned:.4f}")

# ── 7. Residual Analysis ──────────────────────────────────────────────────────

print("\n[7] Residual analysis...")

residuals = y_test - y_pred_tuned

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Residual Analysis — Tuned Model", fontsize=15)

# Residuals vs Predicted
axes[0, 0].scatter(y_pred_tuned, residuals, alpha=0.3, s=5, c="steelblue")
axes[0, 0].axhline(y=0, color="red", linestyle="--", alpha=0.5)
axes[0, 0].set_xlabel("Predicted (log scale)")
axes[0, 0].set_ylabel("Residuals")
axes[0, 0].set_title("Residuals vs Predicted Values")

# Histogram of residuals
axes[0, 1].hist(residuals, bins=50, edgecolor="white", alpha=0.7, color="coral")
axes[0, 1].axvline(x=0, color="red", linestyle="--", alpha=0.5)
axes[0, 1].set_xlabel("Residual")
axes[0, 1].set_ylabel("Frequency")
axes[0, 1].set_title("Residual Distribution")

# Q-Q plot (using numpy percentile)
from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title("Q-Q Plot (Normality Check)")

# Actual vs Predicted
axes[1, 1].scatter(y_test, y_pred_tuned, alpha=0.3, s=5, c="green")
min_val = min(y_test.min(), y_pred_tuned.min())
max_val = max(y_test.max(), y_pred_tuned.max())
axes[1, 1].plot([min_val, max_val], [min_val, max_val],
                "r--", alpha=0.5, linewidth=2)
axes[1, 1].set_xlabel("Actual (log scale)")
axes[1, 1].set_ylabel("Predicted (log scale)")
axes[1, 1].set_title("Actual vs Predicted")

plt.tight_layout()
plt.savefig(os.path.join(CHARTS, "residual_analysis.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Residual analysis charts saved")

# ── 8. Feature Importance ─────────────────────────────────────────────────────

print("\n[8] Feature importance analysis...")

if hasattr(tuned_model, "feature_importances_"):
    importances = tuned_model.feature_importances_
    imp_method = "Built-in (Gini importance)"
elif hasattr(tuned_model, "coef_"):
    importances = np.abs(tuned_model.coef_)
    imp_method = "|Coefficient|"
else:
    # Permutation importance fallback
    perm = permutation_importance(
        tuned_model, X_test_scaled, y_test,
        n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
    )
    importances = perm.importances_mean
    imp_method = "Permutation importance"

feat_importance = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": importances
}).sort_values("Importance", ascending=False)

feat_importance.to_csv(os.path.join(OUTPUTS, "feature_importance.csv"), index=False)

print(f"  Method: {imp_method}")
print("  Top 10 features:")
for _, row in feat_importance.head(10).iterrows():
    print(f"    {row['Feature']:20s}  {row['Importance']:.4f}")

# Feature importance chart
plt.figure(figsize=(10, 8))
top_n = min(14, len(feat_importance))
top_feats = feat_importance.head(top_n)
colors = plt.cm.viridis(np.linspace(0.2, 0.8, top_n))
plt.barh(range(top_n), top_feats["Importance"].values, color=colors)
plt.yticks(range(top_n), top_feats["Feature"].values)
plt.xlabel("Importance")
plt.title(f"Top {top_n} Feature Importances ({imp_method})", fontsize=14)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(CHARTS, "feature_importance.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Feature importance chart saved")

# ── 9. Learning Curves ────────────────────────────────────────────────────────

print("\n[9] Learning curve analysis...")

from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    tuned_model, X_train_scaled, y_train,
    train_sizes=np.linspace(0.1, 1.0, 6),
    cv=3, scoring="r2", n_jobs=-1, random_state=RANDOM_STATE
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                 alpha=0.2, color="steelblue")
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                 alpha=0.2, color="coral")
plt.plot(train_sizes, train_mean, "o-", label="Training R²", color="steelblue")
plt.plot(train_sizes, val_mean, "s-", label="Validation R²", color="coral")
plt.xlabel("Training Set Size")
plt.ylabel("R² Score")
plt.title("Learning Curves", fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS, "learning_curves.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Learning curves saved")
print(f"  Final training R²: {train_mean[-1]:.4f} ± {train_std[-1]:.4f}")
print(f"  Final validation R²: {val_mean[-1]:.4f} ± {val_std[-1]:.4f}")

# ── 10. Summary Results Table ─────────────────────────────────────────────────

print("\n" + "=" * 65)
print("  RESULTS SUMMARY")
print("=" * 65)
print(f"\n  Best Model: {best_model[0]}"
      f"\n  Tuned Params: {grid.best_params_}"
      f"\n  Test R² Score:  {r2_tuned:.4f}"
      f"\n  Test RMSE:      {rmse_tuned:.4f}  (in log-space)"
      f"\n  Test MAE:       {mae_tuned:.4f}  (in log-space)")
print(f"\n  Back-transform: exp(RMSE) ≈ ${np.expm1(rmse_tuned) * 100_000:,.0f}")
print(f"  Back-transform: exp(MAE) ≈ ${np.expm1(mae_tuned) * 100_000:,.0f}")

# Save results as JSON
final_results = {
    "dataset": "California Housing",
    "samples": df.shape[0],
    "features_original": len(housing.feature_names),
    "features_engineered": len(feature_cols),
    "best_model": best_model[0],
    "best_params": {k: str(v) for k, v in grid.best_params_.items()},
    "test_r2": round(r2_tuned, 4),
    "test_rmse_log": round(rmse_tuned, 4),
    "test_mae_log": round(mae_tuned, 4),
    "rmse_backtransformed_usd": round(np.expm1(rmse_tuned) * 100_000, 0),
    "mae_backtransformed_usd": round(np.expm1(mae_tuned) * 100_000, 0),
    "cv_r2_mean": round(grid.best_score_, 4),
    "top_features": feat_importance.head(10).to_dict("records"),
    "all_models": results,
    "generated": datetime.now().isoformat(),
}
with open(os.path.join(OUTPUTS, "results.json"), "w") as f:
    json.dump(final_results, f, indent=2, default=str)

print(f"\n  Results saved to {OUTPUTS}/")
print(f"  Charts saved to {CHARTS}/")
print("\n" + "=" * 65)
print("  ANALYSIS COMPLETE")
print("=" * 65)
