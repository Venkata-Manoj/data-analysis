# Data Analysis Portfolio

A collection of hands-on data analysis and machine learning projects I built to explore real-world datasets and solve practical problems. Each project is a complete story, from raw data to insights.

---

## Projects

### Project 1: Customer Segmentation (RFM Clustering)

**Directory:** [`customer-rfm-segmentation/`](customer-rfm-segmentation/)

An interactive Streamlit dashboard that groups customers into segments based on how recently they bought, how often they buy, and how much they spend. Helps businesses identify their best customers, re-engage at-risk ones, and understand what's happening with their base at a glance.

| Detail | Value |
|--------|-------|
| Technique | KMeans clustering, RFM scoring |
| Dataset | UCI Online Retail |
| Tools | Python, Pandas, Scikit-learn, Streamlit, Plotly |
| Status | Complete |

### Project 2: NLP Sentiment Analysis - IMDB Reviews

**Directory:** [`nlp-sentiment-analysis/`](nlp-sentiment-analysis/)

A complete NLP pipeline that reads movie reviews and tells you whether they're positive or negative. I trained and compared three classifiers, then picked the best one. You can test it yourself through the interactive Streamlit dashboard (run `streamlit run app.py` to launch locally).

| Detail | Value |
|--------|-------|
| Technique | TF-IDF vectorization, Logistic Regression, Naive Bayes, Random Forest |
| Dataset | Stanford IMDB Large Movie Review Dataset (50k reviews) |
| Tools | Hugging Face Datasets, Scikit-learn, NLTK, WordCloud, Plotly |
| Status | Complete |

**Results:**

| Model | Accuracy | F1 Score | ROC-AUC |
|-------|----------|----------|---------|
| **Logistic Regression** | **88.1%** | **0.882** | **0.953** |
| Multinomial Naive Bayes | ~85% | ~0.85 | ~0.93 |
| Random Forest | ~84% | ~0.84 | ~0.92 |

### Project 3: House Price Prediction — California Housing

**Directory:** [`house-price-prediction/`](house-price-prediction/)

A complete regression pipeline that predicts median house values across California census block groups. Five models compared (Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting), with the tuned Gradient Boosting achieving **R² = 0.836** and a typical prediction error of **~$10,707**. Includes feature engineering, geospatial EDA, residual analysis, and learning curves.

| Detail | Value |
|--------|-------|
| Technique | Gradient Boosting, Random Forest, Ridge/Lasso, feature engineering |
| Dataset | California Housing (sklearn) — 20,640 block groups |
| Tools | Scikit-learn, Pandas, Matplotlib, Seaborn |
| Status | Complete |

**Results:**

| Model | R² Score | MAE (log) | RMSE (log) |
|-------|----------|-----------|------------|
| **Gradient Boosting (tuned)** | **0.8363** | **0.1017** | **0.1436** |
| Random Forest (50) | 0.8149 | 0.1073 | 0.1527 |
| Ridge (alpha=1.0) | 0.6721 | 0.1525 | 0.2033 |
| Linear Regression | 0.6721 | 0.1525 | 0.2033 |

| Key finding | The engineered `IncomePerRoom` feature dominates (43.8% importance) — neighborhood affluence density predicts price better than income alone. |

### Project 4: Wine Quality Classification

**Directory:** [`wine-quality-classification/`](wine-quality-classification/)

A complete classification pipeline predicting red wine quality (0–10) from 11 physicochemical properties. Two approaches: binary (good wine >= 7 vs. poor) with four models compared, and multi-class exact score prediction. Random Forest achieves the best recall (58%) and ROC-AUC (0.955).

| Detail | Value |
|--------|-------|
| Technique | Random Forest, Gradient Boosting, Logistic Regression, SVM, feature importance analysis |
| Dataset | UCI Wine Quality — 1,599 red wine samples |
| Tools | Scikit-learn, Pandas, Matplotlib, Seaborn |
| Status | Complete |

**Results:**

| Model | Accuracy | F1 Score | ROC-AUC |
|-------|----------|----------|---------|
| **Random Forest** | **93.8%** | **0.714** | **0.955** |
| Gradient Boosting | 93.1% | 0.703 | 0.916 |
| SVM (RBF) | 90.0% | 0.500 | 0.889 |
| Logistic Regression | 89.4% | 0.485 | 0.880 |

| Key finding | Alcohol content (17.4%), sulphates (11.1%), and volatile acidity (10.2%) are the strongest predictors — confirming domain knowledge in oenology. |

### Project 5: PM2.5 Air Quality Forecasting

**Directory:** [`pm25-air-quality-forecasting/`](pm25-air-quality-forecasting/)

A multivariate time series forecasting pipeline that predicts hourly PM2.5 concentration in Beijing using lag features, rolling statistics, and temporal features. Four models compared on a strict time-based split (train 2010–2013, test 2014). Linear Regression achieves **R² = 0.9465** and **MAE = 11.8 μg/m³**.

| Detail | Value |
|--------|-------|
| Technique | Time series feature engineering, lag/rolling features, 4-model comparison |
| Dataset | UCI Beijing PM2.5 — 43,824 hourly readings (2010–2014) |
| Tools | Scikit-learn, XGBoost, Pandas, Matplotlib, Seaborn |
| Status | Complete |

**Results:**

| Model | MAE | RMSE | R² | MAPE |
|-------|-----|------|----|------|
| **Linear Regression** | **11.78** | **21.56** | **0.9465** | **23.56%** |
| Ridge (α=10) | 11.78 | 21.56 | 0.9465 | 23.57% |
| Random Forest | 12.05 | 23.42 | 0.9369 | 21.81% |
| XGBoost | 12.59 | 24.06 | 0.9334 | 22.50% |

|| Key finding | Feature engineering (lag + rolling) matters more than model complexity for PM2.5 — even Linear Regression matches tree-based models given the right temporal features. |

### Project 6: Topic Modeling — 20 Newsgroups

**Directory:** [`topic-modeling-newsgroups/`](topic-modeling-newsgroups/)

An unsupervised text mining pipeline that discovers latent themes across 8 categories of the 20 Newsgroups dataset. Compares three algorithms: NMF, Latent Dirichlet Allocation (LDA), and TruncatedSVD (LSA). LDA achieves the best coherence (0.2017), while NMF produces the most interpretable keyword sets. Includes 6 visualizations: word clouds, topic-term heatmaps, model comparison, topic-category alignment, confidence distributions, and 3D topic space.

| Detail | Value |
|--------|-------|
| Technique | NMF, LDA, TruncatedSVD (LSA), topic coherence evaluation |
| Dataset | [20 Newsgroups](https://scikit-learn.org/stable/datasets/real_world.html#newsgroups-dataset) (sklearn) — 1,600 docs, 8 categories |
| Tools | Scikit-learn, Pandas, Matplotlib, Seaborn, WordCloud |
| Status | Complete |

**Results:**

| Model | Mean Coherence | Strength |
|-------|---------------|----------|
| **LDA (Online)** | **0.2017** | Best coherence; clear topic separation |
| NMF | 0.0374 | Cleanest keyword sets per topic |
| TruncatedSVD (LSA) | 0.0451 | Best document similarity space |

|| Key finding | LDA produces the most coherent topics (probabilistic separation), while NMF excels at interpretability (human-readable keyword sets). Topic modeling recovers ground-truth categories without ever seeing labels. |

### Project 7: Fashion-MNIST Image Classification

**Directory:** [`fashion-mnist-classification/`](fashion-mnist-classification/)

A complete image classification pipeline comparing traditional ML (Random Forest, Logistic Regression) with a Neural Network (MLPClassifier) on **Fashion-MNIST** — 70,000 grayscale 28×28 images across 10 clothing categories. Uses PCA dimensionality reduction (30 components, 75.6% variance retained) for efficient training and visualization.

| Detail | Value |
|--------|-------|
| Technique | PCA, Logistic Regression, Random Forest, MLPClassifier (128→64), confusion matrix analysis |
| Dataset | [Fashion-MNIST](https://www.openml.org/d/40996) (OpenML) — 70k grayscale images |
| Tools | Scikit-learn, Matplotlib, Seaborn, NumPy |
| Status | Complete |

**Results:**

| Model | Accuracy | Time (s) |
|-------|----------|----------|
| **Neural Network (MLP)** | **81.75%** | **0.4** |
| Random Forest | 81.25% | 0.5 |
| Logistic Regression | 80.00% | 0.4 |

| Key finding | The MLP outperforms traditional methods, but struggles with visually similar classes (Shirt vs. T-shirt/top). Items with distinct silhouettes (Trouser, Bag, Boot) achieve near-perfect F1 (>0.90). |

### Project 8: Recommender System - MovieLens 100k

**Directory:** [`recommender-system-movielens/`](recommender-system-movielens/)

The portfolio's first **ranking and personalization** project, built on the classic **MovieLens 100k** dataset (100,000 ratings from 943 users across 1,682 movies). It compares four recommendation approaches on a held-out 20% test split and ships a working Top-N recommender. This is a different problem family from the regression, classification, NLP, computer-vision, clustering, and topic-modeling work in Projects 1-7.

| Detail | Value |
|--------|-------|
| Technique | Biased matrix factorization (Funk SVD), User/Item KNN collaborative filtering, bias baseline |
| Dataset | [MovieLens 100k](https://grouplens.org/datasets/movielens/100k/) (GroupLens) |
| Evaluation | RMSE, MAE on a seeded 20% holdout |
| Tools | Pandas, NumPy, Scikit-learn, Matplotlib |
| Status | Complete |

**Results (20% holdout):**

| Model | RMSE | MAE |
|-------|------|-----|
| **Biased SVD (50 factors)** | **0.9262** | **0.7248** |
| Baseline (bias) | 0.9607 | 0.7530 |
| ItemKNN (k=40) | 0.9869 | 0.7856 |
| UserKNN (k=40) | 0.9931 | 0.7894 |

| Key finding | The biased matrix factorization (Funk SVD with global + user + item bias and 50 latent factors) clearly beats the standalone bias baseline and both KNN variants. Raw bias offsets capture a lot, but learning latent factors closes the remaining gap. KNN helps only marginally here because MovieLens 100k is dense enough that global structure dominates - matrix factorization scales and generalizes far better on colder catalogs. |

### Project 9: Credit Card Fraud Detection - Anomaly Detection

The portfolio's first **anomaly detection / extreme-imbalance** project, built on the canonical Credit Card Fraud Detection dataset (284,807 transactions, 492 fraud, **0.172%**). It compares **unsupervised** anomaly detectors - which need no fraud labels at training time - against a **supervised** baseline that uses labels. This is a different problem family from the regression, classification, NLP, computer-vision, clustering, topic-modeling, and recommender work in Projects 1-8.

| Detail | Value |
|--------|-------|
| Technique | Isolation Forest, Local Outlier Factor, One-Class SVM (unsupervised); class-weighted Logistic Regression (supervised) |
| Dataset | [Credit Card Fraud Detection](https://www.openml.org/d/1597) (OpenML, ULB/MLG) - 284,807 rows, 30 features, 0.172% fraud |
| Evaluation | ROC-AUC, Average Precision, Precision@k (imbalance-appropriate) |
| Tools | Pandas, NumPy, Scikit-learn, Matplotlib |
| Status | Complete |

**Representative results (20% holdout, seeded):**

| Model | ROC-AUC | Avg Precision | Precision@k |
|-------|---------|---------------|-------------|
| **Isolation Forest** | **0.953** | **0.180** | **0.306** |
| One-Class SVM | 0.954 | 0.120 | 0.153 |
| Local Outlier Factor | 0.485 | 0.002 | 0.000 |
| Logistic Regression (supervised) | 0.971 | 0.718 | 0.059* |

\* Supervised precision is low because the model casts a wide net at the 0.5 threshold; with cost-sensitive thresholding the alert budget can be tuned.

| Key finding | Unsupervised detectors surface fraud using only the *shape* of normal activity - Isolation Forest and One-Class SVM reach ~0.95 ROC-AUC with zero labels - while the supervised model shows the ceiling when labels exist. LOF in novelty mode underperforms here, a useful honest signal about detector-vs-scale fit. |

### Visual Gallery

| Confusion Matrices | Review Length Distribution |
|:---:|:---:|
| ![Confusion matrices](nlp-sentiment-analysis/outputs/confusion_matrices.png) | ![Review length distribution](nlp-sentiment-analysis/outputs/review_length_distribution.png) |

| Positive Reviews Word Cloud | Negative Reviews Word Cloud |
|:---:|:---:|
| ![Positive reviews word cloud](nlp-sentiment-analysis/outputs/wordcloud_positive.png) | ![Negative reviews word cloud](nlp-sentiment-analysis/outputs/wordcloud_negative.png) |

**Top Predictive Features:**

![Top predictive features](nlp-sentiment-analysis/outputs/top_features.png)

### Wine Quality — Charts

| Quality Distribution | ROC Curves |
|:---:|:---:|
| ![Distribution](wine-quality-classification/charts/01-quality-distribution.png) | ![ROC](wine-quality-classification/charts/04-roc-curves.png) |

| Feature Importance | Model Comparison |
|:---:|:---:|
| ![Importance](wine-quality-classification/charts/07-feature-importance.png) | ![Comparison](wine-quality-classification/charts/05-model-comparison.png) |

| Confusion Matrix (Best Model) | Multi-Class Matrix |
|:---:|:---:|
| ![Confusion](wine-quality-classification/charts/06-confusion-matrix.png) | ![Multi-class](wine-quality-classification/charts/08-multiclass-matrix.png) |

### PM2.5 Air Quality — Charts

| Predictions vs Actual | Feature Importance |
|:---:|:---:|
| ![Predictions](pm25-air-quality-forecasting/charts/02-predictions-vs-actual.png) | ![Feature importance](pm25-air-quality-forecasting/charts/05-feature-importance.png) |

| Model Comparison | Residuals Distribution |
|:---:|:---:|
| ![Model comparison](pm25-air-quality-forecasting/charts/04-model-comparison.png) | ![Residuals](pm25-air-quality-forecasting/charts/06-residuals-distribution.png) |

| Hourly Pattern | Weekly Pattern |
|:---:|:---:|
| ![Hourly](pm25-air-quality-forecasting/charts/07-hourly-pattern.png) | ![Weekly](pm25-air-quality-forecasting/charts/08-weekly-pattern.png) |

### Fashion-MNIST — Charts

| Sample Images | PCA 2D Projection |
|:---:|:---:|
| ![Samples](fashion-mnist-classification/charts/01-sample-images.png) | ![PCA](fashion-mnist-classification/charts/05-pca-2d-visualization.png) |

| Model Comparison | Confusion Matrix (MLP) |
|:---:|:---:|
| ![Comparison](fashion-mnist-classification/charts/06-model-comparison.png) | ![Confusion](fashion-mnist-classification/charts/07-confusion-matrix-best-model.png) |

| Per-Class F1 | MLP Learning Curves |
|:---:|:---:|
| ![F1](fashion-mnist-classification/charts/08-per-class-performance.png) | ![Learning curves](fashion-mnist-classification/charts/09-mlp-learning-curves.png) |

### Topic Modeling — Charts

| Word Clouds (NMF) | Topic-Term Heatmap |
|:---:|:---:|
| ![Word clouds](topic-modeling-newsgroups/charts/01-topic-wordclouds.png) | ![Topic-term heatmap](topic-modeling-newsgroups/charts/02-topic-term-heatmap.png) |

| Model Comparison | Topic-Category Alignment |
|:---:|:---:|
| ![Model comparison](topic-modeling-newsgroups/charts/03-model-comparison.png) | ![Topic-category heatmap](topic-modeling-newsgroups/charts/04-topic-category-heatmap.png) |

| Assignment Confidence | 3D Topic Space |
|:---:|:---:|
| ![Confidence histogram](topic-modeling-newsgroups/charts/05-topic-confidence-histogram.png) | ![3D topic space](topic-modeling-newsgroups/charts/06-3d-topic-space.png) |

### Recommender System - Charts

| Rating Distribution | Model Comparison |
|:---:|:---:|
| ![Distribution](recommender-system-movielens/charts/01-rating-distribution.png) | ![Comparison](recommender-system-movielens/charts/05-model-comparison.png) |

| SVD Training Curve | Top-10 Recommendations |
|:---:|:---:|
| ![Training](recommender-system-movielens/charts/06-svd-training-curve.png) | ![Top picks](recommender-system-movielens/charts/08-top-recommendations.png) |

### Fraud Detection - Charts

| Class Imbalance | Detector ROC Curves |
|:---:|:---:|
| ![Imbalance](anomaly-detection-fraud/charts/01-class-imbalance.png) | ![ROC](anomaly-detection-fraud/charts/05-detector-roc-curves.png) |

| Precision@k | Model Comparison |
|:---:|:---:|
| ![Precision@k](anomaly-detection-fraud/charts/07-precision-at-k.png) | ![Comparison](anomaly-detection-fraud/charts/08-model-comparison.png) |

---

## Quick Start

```bash
# Clone the repo
git clone https://github.com/Venkata-Manoj/data-analysis.git
cd data-analysis

# Project 1: Customer Segmentation
cd customer-rfm-segmentation
pip install -r requirements.txt
streamlit run app.py

# Project 2: NLP Sentiment Analysis
cd nlp-sentiment-analysis
pip install -r requirements.txt
jupyter notebook sentiment_analysis_executed.ipynb
# Or launch the interactive dashboard:
streamlit run app.py

# Project 4: Wine Quality Classification
cd wine-quality-classification
pip install -r requirements.txt
python analysis.py

# Project 5: PM2.5 Air Quality Forecasting
cd pm25-air-quality-forecasting
pip install -r requirements.txt
python analysis.py

# Project 6: Topic Modeling — 20 Newsgroups
cd topic-modeling-newsgroups
pip install -r requirements.txt
python topic_modeling.py

# Project 7: Fashion-MNIST Image Classification
cd fashion-mnist-classification
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python analysis.py

# Project 8: Recommender System - MovieLens 100k
cd recommender-system-movielens
pip install -r requirements.txt
python analysis.py

# Project 9: Credit Card Fraud Detection - Anomaly Detection
cd anomaly-detection-fraud
pip install -r requirements.txt
python analysis.py
```

## Tech Stack

- **Languages:** Python 3.11+
- **Data:** Pandas, NumPy
- **ML:** Scikit-learn, NLTK
- **Visualisation:** Matplotlib, Seaborn, Plotly, WordCloud
- **NLP:** NLTK, WordCloud
- **Notebooks:** Jupyter
- **Datasets:** Hugging Face Datasets, UCI Repository, sklearn datasets, OpenML

## License

MIT - feel free to use, modify, and share.
