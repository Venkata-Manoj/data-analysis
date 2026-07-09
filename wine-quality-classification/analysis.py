#!/usr/bin/env python3
"""Wine Quality Classification — Quick version with timing."""

import os
import time
import urllib.request
import warnings
from collections import OrderedDict
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")

PDIR = Path(__file__).resolve().parent
CHARTS = PDIR / "charts"
OUTPUTS = PDIR / "outputs"
os.makedirs(CHARTS, exist_ok=True)
os.makedirs(OUTPUTS, exist_ok=True)

t_start = time.time()


def step(s):
    print(f"[{time.time() - t_start:.1f}s] {s}")


step("Loading data...")
DATA_DIR = PDIR / "data"
DATA_PATH = DATA_DIR / "winequality-red.csv"
if not DATA_PATH.exists():
    DATA_DIR.mkdir(exist_ok=True)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    step(f"Downloading from {url}...")
    urllib.request.urlretrieve(url, DATA_PATH)
    step("Download complete.")
df = pd.read_csv(DATA_PATH, sep=";")
feats = [c for c in df.columns if c != "quality"]
df["quality_binary"] = (df["quality"] >= 7).astype(int)

step(f"Dataset: {df.shape[0]} samples, {len(feats)} features")

# Chart 1: Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.countplot(x="quality", data=df, palette="viridis", ax=axes[0])
axes[0].set_title("Wine Quality Distribution")
bc = df["quality_binary"].value_counts()
axes[1].bar(["Poor (<7)", "Good (>=7)"], bc.values, color=["#e74c3c", "#2ecc71"], width=0.5)
axes[1].set_title("Binary Split")
for i, v in enumerate(bc.values):
    axes[1].text(i, v + 10, str(v), ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig(CHARTS / "01-quality-distribution.png", dpi=200, bbox_inches="tight")
plt.close()
step("Chart 1 done")

# Chart 2: Correlation
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(df[feats].corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True, ax=ax)
ax.set_title("Feature Correlation")
plt.tight_layout()
plt.savefig(CHARTS / "02-correlation-heatmap.png", dpi=200, bbox_inches="tight")
plt.close()
step("Chart 2 done")

# Chart 3: Boxplots
f_scores, _ = f_classif(df[feats], df["quality"])
top6 = [feats[i] for i in np.argsort(f_scores)[-6:]]
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for i, feat in enumerate(top6):
    sns.boxplot(x="quality", y=feat, data=df, palette="viridis", ax=axes.flat[i])
    axes.flat[i].set_title(f"{feat}")
plt.suptitle("Top 6 Features by Quality", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(CHARTS / "03-feature-boxplots.png", dpi=200, bbox_inches="tight")
plt.close()
step("Chart 3 done")

# Train/Test split
X = df[feats].values
y = df["quality_binary"].values
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)
step(f"Split: {X_tr.shape[0]} train, {X_te.shape[0]} test")

# Models
models = OrderedDict(
    [
        ("Logistic Regression", LogisticRegression(max_iter=2000, random_state=42)),
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ("Gradient Boosting", GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)),
        ("SVM (RBF)", SVC(kernel="rbf", probability=True, random_state=42)),
    ]
)

results = {}
for name, model in models.items():
    t0 = time.time()
    model.fit(X_tr_s, y_tr)
    yp = model.predict(X_te_s)
    yprob = model.predict_proba(X_te_s)[:, 1]
    results[name] = {
        "acc": accuracy_score(y_te, yp),
        "prec": precision_score(y_te, yp, zero_division=0),
        "rec": recall_score(y_te, yp, zero_division=0),
        "f1": f1_score(y_te, yp, zero_division=0),
        "auc": roc_auc_score(y_te, yprob),
    }
    r = results[name]
    step(f"{name:25s} fit in {time.time() - t0:.2f}s | Acc={r['acc']:.4f} F1={r['f1']:.4f} AUC={r['auc']:.4f}")

best = max(results, key=lambda k: results[k]["f1"])
step(f"Best: {best}")

# Chart 4: ROC
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
for name, r in results.items():
    yprob = models[name].predict_proba(X_te_s)[:, 1]
    fpr, tpr, _ = roc_curve(y_te, yprob)
    ax.plot(fpr, tpr, lw=2, label=f"{name} ({r['auc']:.3f})")
ax.legend(loc="lower right")
ax.set_title("ROC Curves", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(CHARTS / "04-roc-curves.png", dpi=200, bbox_inches="tight")
plt.close()
step("Chart 4 done")

# Chart 5: Comparison bar
pdf = pd.DataFrame(results).T[["acc", "prec", "rec", "f1", "auc"]]
fig, ax = plt.subplots(figsize=(12, 6))
pdf.plot(kind="bar", ax=ax, width=0.8, colormap="viridis", edgecolor="white")
ax.set_title("Model Comparison", fontsize=14, fontweight="bold")
ax.set_ylim([0, 1.05])
ax.legend(loc="lower right")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
for c in ax.containers:
    ax.bar_label(c, fmt="%.3f", fontsize=7, rotation=90)
plt.tight_layout()
plt.savefig(CHARTS / "05-model-comparison.png", dpi=200, bbox_inches="tight")
plt.close()
step("Chart 5 done")

# Chart 6: Confusion matrix
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_te, models[best].predict(X_te_s), ax=ax, display_labels=["Poor", "Good"], cmap="Blues", colorbar=False
)
ax.set_title(f"Confusion Matrix — {best}", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(CHARTS / "06-confusion-matrix.png", dpi=200, bbox_inches="tight")
plt.close()
step("Chart 6 done")

# Chart 7: Feature importance
if hasattr(models[best], "feature_importances_"):
    imps = models[best].feature_importances_
elif hasattr(models[best], "coef_"):
    imps = np.abs(models[best].coef_[0])
else:
    imps = None
if imps is not None:
    fidf = pd.DataFrame({"f": feats, "i": imps}).sort_values("i", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="i", y="f", data=fidf, palette="viridis", ax=ax)
    ax.set_title(f"Feature Importance — {best}")
    for i, v in enumerate(fidf["i"]):
        ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(CHARTS / "07-feature-importance.png", dpi=200, bbox_inches="tight")
    plt.close()
    step("Chart 7 done")

# Multi-class
step("Multi-class...")
mask = (df["quality"] >= 3) & (df["quality"] <= 8)
Xmc = df.loc[mask, feats].values
ymc = df.loc[mask, "quality"].values
Xmctr, Xmcte, ymctr, ymcte = train_test_split(Xmc, ymc, test_size=0.2, random_state=42, stratify=ymc)
smc = StandardScaler()
Xmctrs = smc.fit_transform(Xmctr)
Xmctes = smc.transform(Xmcte)
mc_models = OrderedDict(
    [
        ("LR", LogisticRegression(max_iter=2000, random_state=42)),
        ("RF", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ("GB", GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)),
    ]
)
for name, model in mc_models.items():
    t0 = time.time()
    model.fit(Xmctrs, ymctr)
    yp = model.predict(Xmctes)
    step(
        f"  MC {name:5s} in {time.time() - t0:.2f}s | Acc={accuracy_score(ymcte, yp):.4f} F1w={f1_score(ymcte, yp, average='weighted'):.4f}"
    )

# MC confusion matrix
best_mc = "LR"
best_mc_acc = 0
for n, m in mc_models.items():
    yp = m.predict(Xmctes)
    acc = accuracy_score(ymcte, yp)
    if acc > best_mc_acc:
        best_mc = n
        best_mc_acc = acc
yp_mc = mc_models[best_mc].predict(Xmctes)
fig, ax = plt.subplots(figsize=(10, 8))
ConfusionMatrixDisplay.from_predictions(ymcte, yp_mc, ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Multi-Class Matrix — {best_mc}")
plt.tight_layout()
plt.savefig(CHARTS / "08-multiclass-matrix.png", dpi=200, bbox_inches="tight")
plt.close()
step("Chart 8 done")

# Summary
summary = f"""# Wine Quality Classification — Results

## Dataset
- {df.shape[0]} samples, {len(feats)} features
- Quality range 3-8, binary threshold >=7

## Binary Classification
| Model | Acc | Prec | Rec | F1 | AUC |
|-------|-----|------|-----|----|-----|
"""
for n, r in results.items():
    summary += f"| {n} | {r['acc']:.4f} | {r['prec']:.4f} | {r['rec']:.4f} | {r['f1']:.4f} | {r['auc']:.4f} |\n"
summary += f"\nBest: **{best}** (F1={results[best]['f1']:.4f}, AUC={results[best]['auc']:.4f})"
if imps is not None:
    summary += "\n\n## Top 5 Features\n"
    for _, row in fidf.head(5).iterrows():
        summary += f"- {row['f']}: {row['i']:.4f}\n"
with open(OUTPUTS / "results_summary.md", "w") as f:
    f.write(summary)

step(f"Done! Total: {time.time() - t_start:.1f}s")
