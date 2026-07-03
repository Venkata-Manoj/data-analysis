# 📊 Data Analysis Portfolio

Collection of data analysis and machine learning projects. Each project lives in its own top-level directory with a complete pipeline, visualizations, and documented results.

---

## 📂 Projects

### Project 1: 📊 Customer Segmentation (RFM Clustering)

**Directory:** [`customer-rfm-segmentation/`](customer-rfm-segmentation/)

→ Interactive Streamlit dashboard exploring customer segments using **RFM (Recency, Frequency, Monetary)** clustering with KMeans.

| Detail | Value |
|--------|-------|
| Technique | KMeans clustering, RFM scoring |
| Dataset | UCI Online Retail |
| Tools | Python, Pandas, Scikit-learn, Streamlit, Plotly |
| Status | ✅ Complete |

### Project 2: 🎬 NLP Sentiment Analysis — IMDB Reviews

**Directory:** [`nlp-sentiment-analysis/`](nlp-sentiment-analysis/)

→ End-to-end NLP pipeline classifying IMDB movie reviews as positive or negative using TF-IDF and multiple classifiers.

| Detail | Value |
|--------|-------|
| Technique | TF-IDF vectorization, Logistic Regression, Naive Bayes, Random Forest |
| Dataset | Stanford IMDB Large Movie Review Dataset (50k reviews) |
| Tools | Hugging Face Datasets, Scikit-learn, NLTK, WordCloud, Plotly |
| Status | ✅ Complete |

**Results:**

| Model | Accuracy | F1 Score | ROC-AUC |
|-------|----------|----------|---------|
| **Logistic Regression** | **88.1%** | **0.882** | **0.953** |
| Multinomial Naive Bayes | ~85% | ~0.85 | ~0.93 |
| Random Forest | ~84% | ~0.84 | ~0.92 |

---

## 🚀 Quick Start

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
jupyter notebook sentiment_analysis.ipynb
```

## 🛠️ Tech Stack

- **Languages:** Python 3.11+
- **Data:** Pandas, NumPy
- **ML:** Scikit-learn, NLTK
- **Visualisation:** Matplotlib, Seaborn, Plotly, WordCloud
- **Notebooks:** Jupyter
- **Datasets:** Hugging Face Datasets, UCI Repository

## 📝 License

MIT — feel free to use, modify, and share.
