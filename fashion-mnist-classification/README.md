# Project 7: Fashion-MNIST Image Classification

A complete image classification pipeline comparing traditional machine learning (Random Forest, Logistic Regression) with a Neural Network (MLPClassifier) on the **Fashion-MNIST** dataset — 70,000 grayscale 28×28 images across 10 clothing categories.

This project introduces computer vision and neural network techniques to the portfolio, demonstrating high-dimensional image data handling, dimensionality reduction, and multi-class classification.

## Objective

Compare the performance of traditional ML models vs. a neural network on image classification, using dimensionality reduction (PCA) for visualization and model efficiency.

## Dataset

**Fashion-MNIST** ([OpenML](https://www.openml.org/d/40996)) — a modern drop-in replacement for the original MNIST digit dataset. 
- **70,000** grayscale 28×28 images (60k train / 10k test)
- **10 classes**: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
- **784 features** (28×28 pixels flattened)
- Subsampled to **2,000** for pipeline efficiency (1600 train / 400 test)

## Methodology

| Step | Description |
|------|-------------|
| **1. Data Loading** | Load Fashion-MNIST via `fetch_openml`, subsample 2,000, stratified train/test split |
| **2. EDA** | Sample images grid, class distribution, pixel intensity analysis |
| **3. PCA** | 30-component PCA for dimensionality reduction (75.6% variance retained) |
| **4-6. Model Training** | 3 models trained on PCA-reduced data |
| **7. Evaluation** | Model comparison, confusion matrix, per-class F1, learning curves |
| **8. Report** | Text report with all metrics saved to `outputs/` |

## Results

| Model | Accuracy | Time (s) |
|-------|----------|----------|
| **Neural Network (MLP)** | **81.75%** | **0.4** |
| Random Forest | 81.25% | 0.5 |
| Logistic Regression | 80.00% | 0.4 |

**🏆 Best Model:** Neural Network (MLP) — 81.75% accuracy in 0.4 seconds

### Per-Class Performance (MLP)

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Trouser | 1.000 | 0.941 | 0.970 |
| Bag | 0.927 | 0.950 | 0.938 |
| Ankle boot | 0.891 | 0.932 | 0.911 |
| Sandal | 0.897 | 0.854 | 0.875 |
| Sneaker | 0.846 | 0.868 | 0.857 |
| Dress | 0.889 | 0.762 | 0.821 |
| Pullover | 0.756 | 0.816 | 0.785 |
| Coat | 0.711 | 0.821 | 0.762 |
| T-shirt/top | 0.717 | 0.786 | 0.750 |
| Shirt | 0.571 | 0.476 | 0.520 |

> **Key insight:** The MLP classifier significantly outperforms both traditional ML models, but struggles with visually similar classes (Shirt vs. T-shirt/top vs. Pullover). Items with distinct silhouettes (Trouser, Bag, Ankle boot) achieve near-perfect classification.

## Charts

| Chart | Description |
|-------|-------------|
| `01-sample-images.png` | One sample image per class |
| `02-class-distribution.png` | Training set class distribution |
| `03-pixel-intensity.png` | Pixel intensity histogram + mean image |
| `04-pca-explained-variance.png` | Cumulative variance by PCA components |
| `05-pca-2d-visualization.png` | 2D PCA projection colored by class |
| `06-model-comparison.png` | Bar chart comparing model accuracies |
| `07-confusion-matrix-best-model.png` | MLP confusion matrix |
| `08-per-class-performance.png` | Per-class F1 score comparison across models |
| `09-mlp-learning-curves.png` | MLP training loss + validation accuracy curves |

## Tools & Libraries

- **Python** 3.12+
- **scikit-learn** — PCA, all 3 classifiers, metrics
- **Matplotlib** + **Seaborn** — all visualizations
- **NumPy** + **Pandas** — data processing
- **OpenML** — dataset loading

## Run It

```bash
cd fashion-mnist-classification
pip install -r requirements.txt
python analysis.py
```

All charts output to `charts/`, numerical report to `outputs/`.
