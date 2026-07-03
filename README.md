# 📊 Data Analysis Portfolio

Collection of data analysis and machine learning projects. Each project lives in its own subdirectory under `projects/` with a complete pipeline, visualizations, and documented results.

---

## 📂 Projects

### Project 1: Customer Segmentation (RFM Clustering)

**Directory:** [`projects/customer-rfm-segmentation`](projects/customer-rfm-segmentation/) *(coming soon)*

→ Interactive Streamlit dashboard for exploring customer segments using **RFM (Recency, Frequency, Monetary)** clustering and KMeans.

| Detail | Value |
|--------|-------|
| Technique | KMeans clustering, RFM scoring |
| Dataset | UCI Online Retail |
| Tools | Python, Pandas, Scikit-learn, Streamlit, Plotly |
| Status | ✅ See `app.py`, `segmentation.ipynb`, `rfm_segments.csv` (root level) |

### Project 2: NLP Sentiment Analysis — IMDB Reviews 🆕

**Directory:** [`projects/nlp-sentiment-analysis`](projects/nlp-sentiment-analysis/)

→ End-to-end NLP pipeline that classifies IMDB movie reviews as positive or negative using TF-IDF and multiple classifiers.

| Detail | Value |
|--------|-------|
| Technique | TF-IDF vectorization, Logistic Regression, Naive Bayes, Random Forest |
| Dataset | Stanford IMDB Large Movie Review Dataset (50k reviews) |
| Tools | Hugging Face Datasets, Scikit-learn, NLTK, Matplotlib, WordCloud, Plotly |
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
pip install -r requirements.txt
streamlit run app.py

# Project 2: NLP Sentiment Analysis
cd projects/nlp-sentiment-analysis
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
