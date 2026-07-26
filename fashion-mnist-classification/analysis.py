#!/usr/bin/env python3
"""
Fashion-MNIST Image Classification
==================================
A complete image classification pipeline comparing traditional ML (Random Forest,
Logistic Regression) with a Neural Network (MLPClassifier) on the Fashion-MNIST
dataset — 70,000 grayscale 28×28 images across 10 clothing categories.

This project introduces computer vision and neural network techniques to the
Data Analysis Portfolio, demonstrating:
  - High-dimensional image data handling
  - Dimensionality reduction (PCA) for visualization
  - Multi-class classification with traditional ML vs. neural networks
  - Per-class performance analysis
  - Learning dynamics of neural networks
"""

import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for chart generation
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── Configuration ──────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2
SUBSAMPLE_SIZE = 2000  # Use 2k of 70k images for fast training
N_COMPONENTS_PCA = 30  # Components for PCA transformation
np.random.seed(RANDOM_STATE)

CHARTS_DIR = Path(__file__).parent / "charts"
DATA_DIR = Path(__file__).parent / "data"
OUTPUTS_DIR = Path(__file__).parent / "outputs"
for d in [CHARTS_DIR, DATA_DIR, OUTPUTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Plot style
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
    }
)

CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

print("=" * 65)
print("  Fashion-MNIST Image Classification")
print("  Comparing Neural Networks vs Traditional ML")
print("=" * 65)


# ═══════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════
def load_data():
    """Load Fashion-MNIST from OpenML and subsample."""
    print("\n[1/8] Loading Fashion-MNIST dataset...")
    t0 = time.time()

    x, y = fetch_openml("Fashion-MNIST", version=1, return_X_y=True, as_frame=False, data_home=str(DATA_DIR))
    print(f"  Full dataset: {x.shape[0]} samples, {x.shape[1]} features")
    print(f"  Classes: {np.unique(y).tolist()}")

    # Subsample for faster execution
    idx = np.random.choice(x.shape[0], SUBSAMPLE_SIZE, replace=False)
    x, y = x[idx], y[idx]
    print(f"  Subsampled to {SUBSAMPLE_SIZE} samples")

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Train/test split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
    )
    print(f"  Train: {x_train.shape[0]}, Test: {x_test.shape[0]}")
    print(f"  Time: {time.time() - t0:.1f}s")

    return x_train, x_test, y_train, y_test, le


# ═══════════════════════════════════════════════════════════════════════════
# 2. EXPLORATORY DATA ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
def exploratory_analysis(x_train, y_train):
    """Generate EDA charts."""
    print("\n[2/8] Exploratory Data Analysis...")
    t0 = time.time()

    # ── Chart 1: Sample images grid (one per class) ──
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.flatten()
    for i in range(10):
        mask = y_train == i
        idx = np.where(mask)[0][0]
        img = x_train[idx].reshape(28, 28)
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(CLASS_NAMES[i], fontsize=11, fontweight="bold")
        axes[i].axis("off")
    fig.suptitle("Fashion-MNIST — Sample Images (One Per Class)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(CHARTS_DIR / "01-sample-images.png")
    plt.close(fig)
    print("  ✅ Chart 1: Sample images grid")

    # ── Chart 2: Class distribution ──
    fig, ax = plt.subplots(figsize=(10, 5))
    counts = np.bincount(y_train)
    bars = ax.bar(CLASS_NAMES, counts, color=sns.color_palette("muted", 10), edgecolor="white", linewidth=0.8)
    ax.set_title("Class Distribution (Training Set)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_xlabel("Class")
    ax.tick_params(axis="x", rotation=45)
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 20, str(count), ha="center", va="bottom", fontsize=9
        )
    plt.tight_layout()
    fig.savefig(CHARTS_DIR / "02-class-distribution.png")
    plt.close(fig)
    print("  ✅ Chart 2: Class distribution")

    # ── Chart 3: Pixel intensity distribution ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, label, data in zip(
        axes,
        ["All Pixels", "Mean Image (averaged over all samples)"],
        [x_train.ravel(), x_train.mean(axis=0).reshape(28, 28)],
    ):
        if label == "All Pixels":
            ax.hist(data, bins=64, color="steelblue", edgecolor="white", alpha=0.8, linewidth=0.5)
            ax.set_title("Pixel Intensity Distribution", fontsize=12, fontweight="bold")
            ax.set_xlabel("Pixel Intensity")
            ax.set_ylabel("Frequency")
        else:
            im = ax.imshow(data, cmap="viridis")
            ax.set_title("Mean Image (All Classes)", fontsize=12, fontweight="bold")
            plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    fig.savefig(CHARTS_DIR / "03-pixel-intensity.png")
    plt.close(fig)
    print("  ✅ Chart 3: Pixel intensity analysis")

    print(f"  Time: {time.time() - t0:.1f}s")


# ═══════════════════════════════════════════════════════════════════════════
# 3. PCA VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════
def pca_analysis(x_train, y_train, x_test):
    """PCA for dimensionality reduction, visualization, and transformed data."""
    print("\n[3/8] PCA dimensionality reduction...")
    t0 = time.time()

    # Scale first
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # PCA: use n_components for both variance analysis and training transform
    n_components = min(N_COMPONENTS_PCA, x_scaled.shape[1], x_scaled.shape[0])
    pca_obj = PCA(n_components=n_components, random_state=RANDOM_STATE)
    x_train_pca = pca_obj.fit_transform(x_scaled)
    x_test_pca = pca_obj.transform(x_test_scaled)
    cum_var = np.cumsum(pca_obj.explained_variance_ratio_)
    print(f"  PCA: {n_components} components, {cum_var[-1]:.1%} total variance explained")

    # ── Chart 4: Cumulative explained variance ──
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(cum_var) + 1), cum_var, linewidth=2, color="darkorange", marker=".", markersize=3)
    ax.axhline(0.8, color="gray", linestyle="--", alpha=0.7, label="80% threshold")
    n_80 = np.argmax(cum_var >= 0.8) + 1 if cum_var[-1] >= 0.8 else n_components
    ax.axvline(n_80, color="green", linestyle=":", alpha=0.6, label=f"~{n_80} comps → 80%")
    ax.set_title("Cumulative Explained Variance by PCA Components", fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_xlim(1, n_components)
    ax.legend(fontsize=9)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.tight_layout()
    fig.savefig(CHARTS_DIR / "04-pca-explained-variance.png")
    plt.close(fig)
    print(f"  ✅ Chart 4: PCA explained variance ({n_80} comps → 80%)")

    # PCA to 2D for visualization
    pca_2d = PCA(n_components=2, random_state=RANDOM_STATE)
    x_pca_2d = pca_2d.fit_transform(x_scaled)

    # ── Chart 5: 2D PCA scatter ──
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(x_pca_2d[:, 0], x_pca_2d[:, 1], c=y_train, cmap="tab10", alpha=0.4, s=4, edgecolors="none")
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes", loc="upper right", fontsize=8)
    ax.add_artist(legend1)
    for text, name in zip(legend1.get_texts(), CLASS_NAMES):
        text.set_text(name)
    ax.set_title(
        f"Fashion-MNIST — 2D PCA Projection\n"
        f"({pca_2d.explained_variance_ratio_[0]:.1%} + "
        f"{pca_2d.explained_variance_ratio_[1]:.1%} variance explained)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})")
    plt.tight_layout()
    fig.savefig(CHARTS_DIR / "05-pca-2d-visualization.png")
    plt.close(fig)
    print("  ✅ Chart 5: 2D PCA visualization")

    print(f"  Time: {time.time() - t0:.1f}s")
    return x_train_pca, x_test_pca


# ═══════════════════════════════════════════════════════════════════════════
# 4-6. MODEL TRAINING AND COMPARISON
# ═══════════════════════════════════════════════════════════════════════════
def train_models(x_train, y_train, x_test, y_test):
    """Train and evaluate multiple classifiers on PCA-reduced data."""
    results = {}

    # ── Model 1: Logistic Regression ──
    print("\n[4/8] Training Logistic Regression...")
    t0 = time.time()
    lr = LogisticRegression(solver="lbfgs", max_iter=500, C=1.0, random_state=RANDOM_STATE, n_jobs=-1)
    lr.fit(x_train, y_train)
    y_pred_lr = lr.predict(x_test)
    results["Logistic Regression"] = {
        "model": lr,
        "y_pred": y_pred_lr,
        "accuracy": accuracy_score(y_test, y_pred_lr),
        "time": time.time() - t0,
    }
    print(f"  Accuracy: {results['Logistic Regression']['accuracy']:.4f} ({time.time() - t0:.1f}s)")

    # ── Model 2: Random Forest ──
    print("\n[5/8] Training Random Forest...")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=15, min_samples_split=5, random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(x_train, y_train)
    y_pred_rf = rf.predict(x_test)
    results["Random Forest"] = {
        "model": rf,
        "y_pred": y_pred_rf,
        "accuracy": accuracy_score(y_test, y_pred_rf),
        "time": time.time() - t0,
    }
    print(f"  Accuracy: {results['Random Forest']['accuracy']:.4f} ({time.time() - t0:.1f}s)")

    # ── Model 3: Neural Network (MLPClassifier) ──
    print("\n[6/8] Training Neural Network (MLPClassifier)...")
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        mlp = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=0.001,
            batch_size=128,
            learning_rate="adaptive",
            learning_rate_init=0.001,
            max_iter=80,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=RANDOM_STATE,
            verbose=False,
        )
        mlp.fit(x_train, y_train)
    y_pred_mlp = mlp.predict(x_test)
    results["Neural Network (MLP)"] = {
        "model": mlp,
        "y_pred": y_pred_mlp,
        "accuracy": accuracy_score(y_test, y_pred_mlp),
        "time": time.time() - t0,
    }
    print(f"  Accuracy: {results['Neural Network (MLP)']['accuracy']:.4f} ({time.time() - t0:.1f}s)")
    print(f"  Iterations: {mlp.n_iter_}/{mlp.max_iter}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# 7. EVALUATION CHARTS
# ═══════════════════════════════════════════════════════════════════════════
def evaluation_charts(results, y_test):
    """Generate comparison and per-model charts."""
    print("\n[7/8] Generating evaluation charts...")
    t0 = time.time()

    # ── Chart 6: Model Comparison Bar ──
    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(results.keys())
    accs = [results[n]["accuracy"] for n in names]
    times = [results[n]["time"] for n in names]
    colors = ["#3498db", "#2ecc71", "#e74c3c"]

    x_pos = np.arange(len(names))
    bars = ax.bar(x_pos, accs, color=colors, edgecolor="white", linewidth=1.2, width=0.5)

    for i, (bar, acc, t) in enumerate(zip(bars, accs, times)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{acc:.4f}\n({t:.1f}s)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, fontsize=11, fontweight="bold")
    ax.set_ylim(0.7, 0.92)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Model Comparison — Test Accuracy", fontsize=14, fontweight="bold")
    ax.axhline(y=max(accs), color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(CHARTS_DIR / "06-model-comparison.png")
    plt.close(fig)
    print("  ✅ Chart 6: Model comparison")

    # ── Chart 7: Confusion Matrix (best model) ──
    best_name = max(results, key=lambda n: results[n]["accuracy"])
    best_pred = results[best_name]["y_pred"]

    fig, ax = plt.subplots(figsize=(10, 9))
    cm = confusion_matrix(y_test, best_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=True)
    ax.set_title(
        f"Confusion Matrix — {best_name}\n(Accuracy: {results[best_name]['accuracy']:.4f})",
        fontsize=13,
        fontweight="bold",
    )
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    fig.savefig(CHARTS_DIR / "07-confusion-matrix-best-model.png")
    plt.close(fig)
    print(f"  ✅ Chart 7: Confusion matrix ({best_name})")

    # ── Chart 8: Per-class Performance ──
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(CLASS_NAMES))
    width = 0.25
    multiplier = 0

    metrics_data = {}
    for name in results:
        y_pred = results[name]["y_pred"]
        metrics_data[name] = {
            "precision": precision_score(y_test, y_pred, average=None),
            "recall": recall_score(y_test, y_pred, average=None),
            "f1": f1_score(y_test, y_pred, average=None),
        }

    for name, color in [
        ("Logistic Regression", "#3498db"),
        ("Random Forest", "#2ecc71"),
        ("Neural Network (MLP)", "#e74c3c"),
    ]:
        offset = width * multiplier
        ax.bar(x + offset, metrics_data[name]["f1"], width, label=name, color=color, alpha=0.85, edgecolor="white")
        multiplier += 1

    ax.set_xlabel("Class", fontsize=11)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_title("Per-Class F1 Score Comparison", fontsize=13, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(CLASS_NAMES, fontsize=9, rotation=45, ha="right")
    ax.legend(fontsize=10)
    ax.set_ylim(0.5, 1.0)
    ax.axhline(y=0.8, color="gray", linestyle="--", alpha=0.3)
    plt.tight_layout()
    fig.savefig(CHARTS_DIR / "08-per-class-performance.png")
    plt.close(fig)
    print("  ✅ Chart 8: Per-class F1 comparison")

    # ── Chart 9: MLP Learning Curves ──
    mlp = results["Neural Network (MLP)"]["model"]
    if hasattr(mlp, "loss_curve_") and len(mlp.loss_curve_) > 0:
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(mlp.loss_curve_, linewidth=2, color="#e74c3c", label="Training loss")
        if hasattr(mlp, "validation_scores_") and mlp.validation_scores_:
            ax2 = ax1.twinx()
            ax2.plot(mlp.validation_scores_, linewidth=2, color="#2ecc71", linestyle="--", label="Validation accuracy")
            ax2.set_ylabel("Validation Accuracy", color="#2ecc71", fontsize=11)

        ax1.set_xlabel("Iteration", fontsize=11)
        ax1.set_ylabel("Loss", color="#e74c3c", fontsize=11)
        ax1.set_title("Neural Network Training — Loss Curve", fontsize=13, fontweight="bold")
        lines1, labels1 = ax1.get_legend_handles_labels()
        if hasattr(mlp, "validation_scores_") and mlp.validation_scores_:
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        else:
            ax1.legend(loc="upper right")
        plt.tight_layout()
        fig.savefig(CHARTS_DIR / "09-mlp-learning-curves.png")
        plt.close(fig)
        print("  ✅ Chart 9: MLP learning curves")

    print(f"  Time: {time.time() - t0:.1f}s")
    return best_name


# ═══════════════════════════════════════════════════════════════════════════
# 8. REPORT
# ═══════════════════════════════════════════════════════════════════════════
def generate_report(results, best_name, y_test):
    """Print and save a detailed classification report."""
    print("\n[8/8] Generating final report...")

    report_lines = []
    report_lines.append("=" * 65)
    report_lines.append("  FASHION-MNIST CLASSIFICATION — FINAL REPORT")
    report_lines.append("=" * 65)
    report_lines.append("")

    # Summary table
    report_lines.append(f"{'Model':<25} {'Accuracy':>10} {'Time (s)':>10}")
    report_lines.append("-" * 47)
    for name in results:
        acc = results[name]["accuracy"]
        t = results[name]["time"]
        report_lines.append(f"{name:<25} {acc:>10.4f} {t:>10.1f}")
    report_lines.append("")
    report_lines.append(f"🏆 Best Model: {best_name}")
    report_lines.append(f"   Test Accuracy: {results[best_name]['accuracy']:.4f}")
    report_lines.append("")

    # Per-class breakdown for best model
    report_lines.append(f"--- Per-Class Performance ({best_name}) ---")
    report_lines.append("")
    cr = classification_report(y_test, results[best_name]["y_pred"], target_names=CLASS_NAMES, digits=4)
    report_lines.append(cr)

    report_lines.append("")
    report_lines.append("=" * 65)
    report_lines.append("  All charts saved to: charts/")
    report_lines.append("=" * 65)

    report = "\n".join(report_lines)
    print(report)

    # Save report
    with open(OUTPUTS_DIR / "classification_report.txt", "w") as f:
        f.write(report)
    print("  ✅ Report saved to outputs/classification_report.txt")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════
def main():
    pipeline_start = time.time()

    # Step 1: Load
    x_train, x_test, y_train, y_test, label_encoder = load_data()

    # Step 2: EDA
    exploratory_analysis(x_train, y_train)

    # Step 3: PCA (returns scaled+transformed data for training)
    x_train_pca, x_test_pca = pca_analysis(x_train, y_train, x_test)

    # Steps 4-6: Train models on PCA-reduced data
    results = train_models(x_train_pca, y_train, x_test_pca, y_test)

    # Step 7: Evaluation charts
    best_name = evaluation_charts(results, y_test)

    # Step 8: Report
    generate_report(results, best_name, y_test)

    total_time = time.time() - pipeline_start
    print(f"\n{'=' * 65}")
    print(f"  Total pipeline time: {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"  Charts generated: {len(list(CHARTS_DIR.glob('*.png')))}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
