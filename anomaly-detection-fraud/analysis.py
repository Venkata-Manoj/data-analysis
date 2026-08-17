#!/usr/bin/env python3
"""Credit Card Fraud Detection - Anomaly Detection.

Unsupervised anomaly detectors (Isolation Forest, Local Outlier Factor,
One-Class SVM) are compared against a supervised baseline (class-weighted
Logistic Regression) on the canonical Credit Card Fraud Detection dataset:
284,807 transactions, only 492 fraud (0.172%) - an extreme class imbalance
that is exactly the setting where anomaly detection earns its keep.

The script:
  1. Loads the dataset (auto-downloads to data/ on first run, gitignored).
  2. Standardizes features and performs a stratified train/test split.
  3. Fits three unsupervised detectors on *normal-looking* training data
     (no fraud labels used at training time) and scores the test set.
  4. Trains a supervised Logistic Regression using fraud labels, as the
     "if you had labels" ceiling.
  5. Ranks every method with ROC-AUC, Average Precision, and Precision@k.
  6. Produces 8 portfolio charts and a results summary.

Run locally:  python analysis.py

All functions are importable so the unit tests (tests/test_anomaly.py) can
exercise the detection and ranking logic on tiny synthetic data without any
download - the same pattern used by the project's other analysis modules.
"""

import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / CI-safe backend
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

warnings.filterwarnings("ignore")

PDIR = Path(__file__).resolve().parent
CHARTS = PDIR / "charts"
OUTPUTS = PDIR / "outputs"
DATA_DIR = PDIR / "data"
for d in (CHARTS, OUTPUTS, DATA_DIR):
    d.mkdir(exist_ok=True)

DATA_ID = 1597  # OpenML "CreditCardFraudDetection"
RANDOM_STATE = 42
TEST_SIZE = 0.2
OCSVM_TRAIN_CAP = 20000  # One-Class SVM is O(n^2); cap training sample for speed


# --------------------------------------------------------------------------- #
# Data loading & prep
# --------------------------------------------------------------------------- #
def load_data(data_dir=DATA_DIR):
    """Load the Credit Card Fraud Detection dataset.

    Returns (X, y) where X is a 29-column float DataFrame (V1..V28, Time, Amount)
    and y is a 0/1 int Series (1 = fraud). Cached by sklearn's fetcher.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)
    X, y = fetch_openml(data_id=DATA_ID, as_frame=True, parser="auto", return_X_y=True)
    y = y.astype(int)
    return X, y


def split_data(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Stratified train/test split preserving the rare fraud class."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def make_scaler():
    """StandardScaler fits per-run; V1..V28 are already near-standard but Time
    and Amount need scaling, so scale everything uniformly."""
    return StandardScaler()


# --------------------------------------------------------------------------- #
# Unsupervised detectors
# --------------------------------------------------------------------------- #
def fit_isolation_forest(X, random_state=RANDOM_STATE, n_estimators=150, contamination="auto"):
    """Isolation Forest: anomalies are isolated quickly in random trees."""
    clf = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=random_state)
    clf.fit(np.asarray(X, dtype=float))
    return clf


def fit_lof(X, n_neighbors=20):
    """Local Outlier Factor (novelty mode) - density deviation from neighbors."""
    clf = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
    clf.fit(np.asarray(X, dtype=float))
    return clf


def fit_ocsvm(X, nu=0.05, gamma="scale", train_cap=OCSVM_TRAIN_CAP, random_state=RANDOM_STATE):
    """One-Class SVM - hypersphere boundary around the (mostly normal) bulk.

    OCSVM is O(n^2); when the training set exceeds ``train_cap`` rows we fit on a
    stratified subsample for speed and still score the full test set.
    """
    X = np.asarray(X, dtype=float)
    if X.shape[0] > train_cap:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(X.shape[0], size=train_cap, replace=False)
        X = X[idx]
    clf = OneClassSVM(nu=nu, kernel="rbf", gamma=gamma)
    clf.fit(X)
    return clf


def anomaly_scores(clf, X):
    """Higher score => more anomalous (inverts sklearn's "higher = normal")."""
    return -clf.score_samples(np.asarray(X, dtype=float))


# --------------------------------------------------------------------------- #
# Ranking metrics
# --------------------------------------------------------------------------- #
def rank_metrics(y_true, scores):
    """Summarize an anomaly-ranker with imbalance-appropriate metrics.

    y_true: 0/1 array. scores: higher = more anomalous.
    Returns ROC-AUC, Average Precision (PR-AUC), Precision@k and Recall@k where
    k equals the number of true frauds in the set (the "alert budget").
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n_fraud = int(y_true.sum())
    k = max(1, n_fraud)

    order = np.argsort(-scores)  # most anomalous first
    top_k = y_true[order[:k]]
    tp_at_k = int(top_k.sum())

    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "precision_at_k": tp_at_k / k,
        "recall_at_k": tp_at_k / n_fraud if n_fraud else 0.0,
        "n_fraud": n_fraud,
        "k": k,
    }


# --------------------------------------------------------------------------- #
# Supervised baseline (uses fraud labels)
# --------------------------------------------------------------------------- #
def fit_supervised(X_train, y_train):
    """Standardized Logistic Regression with balanced class weights."""
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )
    pipe.fit(np.asarray(X_train, dtype=float), np.asarray(y_train, dtype=int))
    return pipe


def supervised_metrics(pipe, X_test, y_test):
    """Evaluate the supervised model, returning metrics + fraud-positive scores."""
    proba = pipe.predict_proba(np.asarray(X_test, dtype=float))[:, 1]
    pred = (proba >= 0.5).astype(int)
    y_test = np.asarray(y_test, dtype=int)
    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, pred)
    return {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "average_precision": float(average_precision_score(y_test, proba)),
        "precision_fraud": float(report.get("1", {}).get("precision", 0.0)),
        "recall_fraud": float(report.get("1", {}).get("recall", 0.0)),
        "f1_fraud": float(report.get("1", {}).get("f1-score", 0.0)),
        "confusion_matrix": cm.tolist(),
        "scores": proba,
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_all(X, y, random_state=RANDOM_STATE):
    """End-to-end pipeline. Returns a results dict with metrics + curves."""
    t0 = time.time()
    X_train, X_test, y_train, y_test = split_data(X, y, random_state=random_state)
    X_train_a = np.asarray(X_train, dtype=float)
    X_test_a = np.asarray(X_test, dtype=float)
    scaler = make_scaler()
    X_train_s = scaler.fit_transform(X_train_a)
    X_test_s = scaler.transform(X_test_a)

    detectors = {
        "IsolationForest": fit_isolation_forest(X_train_s, random_state=random_state),
        "LOF": fit_lof(X_train_s),
        "OneClassSVM": fit_ocsvm(X_train_s, random_state=random_state),
    }
    det_results = {}
    for name, clf in detectors.items():
        scores = anomaly_scores(clf, X_test_s)
        det_results[name] = {"metrics": rank_metrics(y_test, scores), "scores": scores}

    sup = fit_supervised(X_train_s, y_train)
    sup_res = supervised_metrics(sup, X_test_s, y_test)

    pca = PCA().fit(X_train_s)
    result = {
        "detectors": det_results,
        "supervised": sup_res,
        "pca_explained": np.cumsum(pca.explained_variance_ratio_),
        "y_test": np.asarray(y_test, dtype=int),
        "X_train": X_train_s,
        "elapsed": time.time() - t0,
    }
    return result


# --------------------------------------------------------------------------- #
# Charts & summary
# --------------------------------------------------------------------------- #
def make_charts(result, X, y, out_dir=CHARTS):
    """Render the 8 portfolio charts from a run_all() result."""
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    y_test = result["y_test"]
    n_fraud = int(y_test.sum())

    # 1. Class imbalance
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = y.value_counts().sort_index()
    ax.bar(["Legit (0)", "Fraud (1)"], [counts.get(0, 0), counts.get(1, 0)], color=["#4C72B0", "#C44E52"])
    ax.set_yscale("log")
    ax.set_title("Class imbalance (log scale)")
    ax.set_ylabel("transactions")
    for i, v in enumerate([counts.get(0, 0), counts.get(1, 0)]):
        ax.text(i, v, f" {v:,}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_dir / "01-class-imbalance.png", dpi=120)
    plt.close(fig)

    # 2. Amount distribution by class
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, color in [(0, "#4C72B0"), (1, "#C44E52")]:
        ax.hist(X.loc[y == label, "Amount"], bins=60, alpha=0.6, density=True, label=f"class {label}", color=color)
    ax.set_yscale("log")
    ax.set_title("Transaction Amount by class")
    ax.set_xlabel("Amount")
    ax.set_ylabel("density (log)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "02-amount-distribution.png", dpi=120)
    plt.close(fig)

    # 3. Feature boxplots for a few PCA components
    fig, axes = plt.subplots(1, 4, figsize=(12, 4))
    for ax, col in zip(axes, ["V1", "V3", "V4", "V14"]):
        data0 = X.loc[y == 0, col]
        data1 = X.loc[y == 1, col]
        ax.boxplot([data0, data1], tick_labels=["0", "1"])
        ax.set_title(col)
    fig.suptitle("PCA feature distributions by class")
    fig.tight_layout()
    fig.savefig(out_dir / "03-feature-boxplots.png", dpi=120)
    plt.close(fig)

    # 4. PCA explained variance
    fig, ax = plt.subplots(figsize=(6, 4))
    cum = result["pca_explained"]
    ax.plot(range(1, len(cum) + 1), cum, marker="o")
    ax.axhline(0.95, ls="--", color="gray", label="95%")
    ax.set_title("PCA cumulative explained variance")
    ax.set_xlabel("components")
    ax.set_ylabel("cumulative variance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "04-pca-explained-variance.png", dpi=120)
    plt.close(fig)

    # 5. ROC curves
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, res in result["detectors"].items():
        fpr, tpr, _ = roc_curve(y_test, res["scores"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={res['metrics']['roc_auc']:.3f})")
    fpr, tpr, _ = roc_curve(y_test, result["supervised"]["scores"])
    ax.plot(fpr, tpr, ls="--", label=f"LogReg (AUC={result['supervised']['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k:", alpha=0.4)
    ax.set_title("ROC curves")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "05-detector-roc-curves.png")
    plt.close(fig)

    # 6. PR curves
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, res in result["detectors"].items():
        prec, rec, _ = precision_recall_curve(y_test, res["scores"])
        ax.plot(rec, prec, label=f"{name} (AP={res['metrics']['average_precision']:.3f})")
    prec, rec, _ = precision_recall_curve(y_test, result["supervised"]["scores"])
    ax.plot(rec, prec, ls="--", label=f"LogReg (AP={result['supervised']['average_precision']:.3f})")
    ax.set_title("Precision-Recall curves")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "06-detector-pr-curves.png", dpi=120)
    plt.close(fig)

    # 7. Precision@k (alert budget = number of true frauds)
    fig, ax = plt.subplots(figsize=(6, 4))
    names = list(result["detectors"].keys()) + ["LogReg"]
    pk = [result["detectors"][n]["metrics"]["precision_at_k"] for n in result["detectors"]] + [
        result["supervised"]["precision_fraud"]
    ]
    ax.bar(names, pk, color=["#55A868", "#4C72B0", "#8172B3", "#C44E52"])
    ax.axhline(n_fraud / len(y_test), ls="--", color="gray", label="random baseline")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Precision@k (k={n_fraud} fraud alerts)")
    ax.set_ylabel("precision")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "07-precision-at-k.png", dpi=120)
    plt.close(fig)

    # 8. Model comparison (ROC-AUC + AP)
    fig, ax = plt.subplots(figsize=(6, 4))
    all_names = list(result["detectors"].keys()) + ["LogReg"]
    aucs = [result["detectors"][n]["metrics"]["roc_auc"] for n in result["detectors"]] + [
        result["supervised"]["roc_auc"]
    ]
    aps = [result["detectors"][n]["metrics"]["average_precision"] for n in result["detectors"]] + [
        result["supervised"]["average_precision"]
    ]
    x = np.arange(len(all_names))
    w = 0.35
    ax.bar(x - w / 2, aucs, w, label="ROC-AUC", color="#4C72B0")
    ax.bar(x + w / 2, aps, w, label="Avg Precision", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(all_names, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title("Detector comparison")
    ax.set_ylabel("score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "08-model-comparison.png", dpi=120)
    plt.close(fig)


def build_summary(result, out_dir=OUTPUTS):
    """Write outputs/results_summary.md with the headline metrics."""
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    lines = ["# Credit Card Fraud Detection - Results Summary", ""]
    n_test = len(result["y_test"])
    n_fraud = result["detectors"]["IsolationForest"]["metrics"]["n_fraud"]
    lines.append(f"- Test split: {n_test:,} transactions ({TEST_SIZE:.0%})")
    lines.append(f"- Test fraud: {n_fraud} ({n_fraud / n_test * 100:.3f}%)")
    lines.append(f"- Runtime: {result['elapsed']:.1f}s")
    lines.append("")
    lines.append("## Unsupervised detectors (no fraud labels at training time)")
    lines.append("")
    lines.append("| Detector | ROC-AUC | Avg Precision | Precision@k | Recall@k |")
    lines.append("|----------|---------|---------------|-------------|-----------|")
    for name, res in result["detectors"].items():
        m = res["metrics"]
        lines.append(
            f"| {name} | {m['roc_auc']:.4f} | {m['average_precision']:.4f} | {m['precision_at_k']:.4f} | {m['recall_at_k']:.4f} |"
        )
    lines.append("")
    lines.append("## Supervised baseline (Logistic Regression, labels used)")
    lines.append("")
    s = result["supervised"]
    lines.append(f"- ROC-AUC: {s['roc_auc']:.4f}")
    lines.append(f"- Avg Precision: {s['average_precision']:.4f}")
    lines.append(
        f"- Fraud precision / recall / F1: {s['precision_fraud']:.4f} / {s['recall_fraud']:.4f} / {s['f1_fraud']:.4f}"
    )
    lines.append(f"- Confusion matrix (rows: actual 0/1; cols: pred 0/1): {s['confusion_matrix']}")
    lines.append("")
    lines.append("## Takeaway")
    lines.append("")
    lines.append("Unsupervised detectors surface fraud using only the *shape* of normal activity -")
    lines.append("no labels required - while the supervised model shows the ceiling when labels exist.")
    lines.append("Both are valuable: the former for cold-start monitoring, the latter for tuned detection.")
    text = "\n".join(lines) + "\n"
    (out_dir / "results_summary.md").write_text(text)
    return text


def main():
    print("Loading Credit Card Fraud Detection dataset (OpenML id 1597)...")
    X, y = load_data()
    print(f"  loaded: {len(X):,} transactions, fraud rate = {y.mean() * 100:.3f}%")
    result = run_all(X, y)
    make_charts(result, X, y)
    summary = build_summary(result)
    print(summary)
    print(f"Done in {result['elapsed']:.1f}s. Charts in charts/, summary in outputs/.")


if __name__ == "__main__":
    main()
