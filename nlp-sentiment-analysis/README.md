# NLP Sentiment Analysis - IMDB Movie Reviews

Project 2 of the Data Analysis Portfolio. This is a complete NLP pipeline that reads a movie review and tells you whether it's positive or negative. I trained three classifiers and picked the best one.

## The Dataset

**Stanford IMDB Large Movie Review Dataset** ([HuggingFace](https://huggingface.co/datasets/stanfordnlp/imdb))

- 50,000 highly polarised reviews (25k train / 25k test)
- Balanced 50/50 between positive and negative classes
- Real-world text with HTML tags, varied vocabulary, and natural language patterns

## How It Works

```
Raw Text -> Clean & Preprocess -> TF-IDF Vectorization -> Train Classifiers -> Evaluate & Compare
```

### Step 1: Exploratory Data Analysis
- Review length distribution (median around 800 chars)
- Word clouds for positive vs negative reviews
- No significant class imbalance

### Step 2: Text Preprocessing
- HTML tag removal (`<br />`, etc.)
- Lowercase conversion
- Punctuation and digit removal
- Stop word filtering (NLTK English stop words)
- Porter stemming for word normalisation

### Step 3: Feature Extraction
- **TF-IDF** with 5,000 most important features
- Unigrams + bigrams (captures phrases like "not good")
- Minimum document frequency: 5 | Maximum DF: 80%

### Step 4: Model Training
Three classifiers trained and compared:

| Model | Accuracy | F1 Score | ROC-AUC |
|-------|----------|----------|---------|
| **Logistic Regression** | **88.1%** | **0.882** | **0.953** |
| Multinomial Naive Bayes | ~85% | ~0.85 | ~0.93 |
| Random Forest | ~84% | ~0.84 | ~0.92 |

### Step 5: Results

**Logistic Regression** is the winner. Here's why:
- **Highest accuracy** (~89%)
- **Fast training** (seconds on 25k samples)
- **Easy to interpret** (coefficients show which words matter most)
- **Strong ROC-AUC** (~0.95) means it separates positive from negative reviews really well

Top predictive words found:
- Positive: *excel, amaz, wonder, perfect, great, love, best, fantastic*
- Negative: *worst, aw, boring, terrible, bad, disappoint, horrible, waste*

## Project Structure

```
nlp-sentiment-analysis/
├── README.md                           <- This file
├── sentiment_analysis.ipynb            <- Full analysis notebook (clean)
├── sentiment_analysis_executed.ipynb   <- Executed notebook with outputs
├── app.py                              <- Streamlit interactive dashboard
├── requirements.txt                    <- Python dependencies
├── data/                               <- Dataset (downloaded by notebook)
├── outputs/                            <- Generated files
│   ├── best_model.pkl                  <- Trained Logistic Regression model
│   ├── tfidf_vectorizer.pkl            <- TF-IDF vectorizer
│   ├── model_comparison.html           <- Interactive Plotly comparison chart
│   ├── metrics.json                    <- Model performance metrics
│   ├── predictions.csv                 <- Test set predictions with probabilities
│   ├── review_length_distribution.png  <- EDA chart
│   ├── wordcloud_positive.png          <- Positive review word cloud
│   ├── wordcloud_negative.png          <- Negative review word cloud
│   ├── confusion_matrices.png          <- Confusion matrices for all models
│   └── top_features.png                <- Most important words per class
```

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook sentiment_analysis.ipynb

# Or launch the interactive Streamlit dashboard
streamlit run app.py
```

All notebook cells are self-contained. The dataset downloads automatically from HuggingFace on first run.

## Key Takeaways

1. **TF-IDF + Logistic Regression** is a strong baseline for text classification. Simple, fast, and interpretable.
2. **N-grams** (unigrams + bigrams) capture both individual words and common two-word phrases.
3. **Stop word removal + stemming** reduces vocabulary noise without losing signal.
4. **Review length** has little correlation with sentiment. Both short and long reviews span the full polarity spectrum.

## Future Improvements

- Try deep learning (LSTM, Transformer) for higher accuracy
- Use pre-trained embeddings (GloVe, BERT) instead of TF-IDF
- Deploy as a Streamlit web app for real-time inference
- Add cross-validation for more robust evaluation
- Extend to multi-class sentiment (very positive, neutral, very negative)

## License

MIT - feel free to use, modify, and share.
