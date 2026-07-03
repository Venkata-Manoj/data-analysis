"""
NLP Sentiment Analysis — Streamlit Dashboard
==============================================
Interactive dashboard for IMDB review sentiment analysis.

Tabs:
  1. 📊 Dashboard — metrics, charts, predictions table
  2. 🎯 Live Test — type a review, get instant sentiment
  3. ℹ️ About — project info
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import re
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
OUTPUTS = BASE / "outputs"

MODEL_PATH = OUTPUTS / "best_model.pkl"
VEC_PATH = OUTPUTS / "tfidf_vectorizer.pkl"
METRICS_PATH = OUTPUTS / "metrics.json"
PREDS_PATH = OUTPUTS / "predictions.csv"

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NLP Sentiment Analysis — IMDB Reviews",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Cached resources ───────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_vectorizer():
    return joblib.load(VEC_PATH)

@st.cache_data
def load_metrics():
    with open(METRICS_PATH) as f:
        return json.load(f)

@st.cache_data
def load_predictions():
    df = pd.read_csv(PREDS_PATH)
    df.columns = df.columns.str.strip()
    return df


# ── Text preprocessing (identical to notebook pipeline) ────────────────────
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk

nltk.download("stopwords", quiet=True)

_stemmer = PorterStemmer()
_stop_words = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """Clean & stem a single review — mirrors the notebook's `clean()`."""
    text = re.sub(r"<[^>]+>", " ", text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = [
        _stemmer.stem(t) for t in text.split()
        if t not in _stop_words and len(t) > 2
    ]
    return " ".join(tokens)


# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.title("🎬 Sentiment Analysis")
st.sidebar.markdown("IMDB Movie Reviews · Logistic Regression")
st.sidebar.divider()
st.sidebar.markdown("**Accuracy:** 88.1% &nbsp;·&nbsp; **F1:** 0.882")
st.sidebar.markdown("**ROC-AUC:** 0.953")

# ── Tabs ───────────────────────────────────────────────────────────────────
tab_dash, tab_live, tab_about = st.tabs([
    "📊 Dashboard",
    "🎯 Live Test",
    "ℹ️ About",
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — Dashboard
# ═══════════════════════════════════════════════════════════════════════════
with tab_dash:
    st.header("📊 Model Performance Dashboard")

    metrics = load_metrics()
    preds_df = load_predictions()

    # ── Metric cards ──────────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    col2.metric("Precision", f"{metrics['precision']:.1%}")
    col3.metric("Recall", f"{metrics['recall']:.1%}")
    col4.metric("F1 Score", f"{metrics['f1']:.3f}")
    col5.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

    st.divider()

    # ── Confusion Matrix + Review Length ──────────────────────────────────
    col_cm, col_rl = st.columns(2)
    with col_cm:
        st.subheader("Confusion Matrices")
        cm_path = OUTPUTS / "confusion_matrices.png"
        if cm_path.exists():
            st.image(str(cm_path), use_container_width=True)
    with col_rl:
        st.subheader("Review Length Distribution")
        rl_path = OUTPUTS / "review_length_distribution.png"
        if rl_path.exists():
            st.image(str(rl_path), use_container_width=True)

    # ── Word clouds ───────────────────────────────────────────────────────
    st.subheader("Word Clouds")
    wc1, wc2 = st.columns(2)
    with wc1:
        wc_pos = OUTPUTS / "wordcloud_positive.png"
        if wc_pos.exists():
            st.image(str(wc_pos), caption="Positive Reviews", use_container_width=True)
    with wc2:
        wc_neg = OUTPUTS / "wordcloud_negative.png"
        if wc_neg.exists():
            st.image(str(wc_neg), caption="Negative Reviews", use_container_width=True)

    # ── Top features ──────────────────────────────────────────────────────
    st.subheader("Top Predictive Features")
    feat_path = OUTPUTS / "top_features.png"
    if feat_path.exists():
        st.image(str(feat_path), use_container_width=True)

    st.divider()

    # ── Predictions table ─────────────────────────────────────────────────
    st.subheader("📋 Test Set Predictions")

    filter_col, search_col = st.columns([1, 2])
    with filter_col:
        sent_filter = st.radio(
            "Filter by sentiment",
            ["All", "Correct ✅", "Wrong ❌"],
            horizontal=True,
        )
    with search_col:
        search_term = st.text_input(
            "🔍 Search reviews (text contains…)", placeholder="Type to filter…"
        )

    # Apply filters
    display = preds_df.copy()
    if sent_filter == "Correct ✅":
        display = display[display["label"] == display["prediction"]]
    elif sent_filter == "Wrong ❌":
        display = display[display["label"] != display["prediction"]]

    if search_term:
        display = display[
            display["text"].str.contains(search_term, case=False, na=False)
        ]

    # Show sample size
    st.caption(f"Showing {len(display):,} of {len(preds_df):,} reviews")

    # Truncate text for display
    display_disp = display.head(100).copy()
    display_disp["text_short"] = display_disp["text"].str[:200] + "…"
    display_disp["actual"] = display_disp["label"].map({1: "🟢 Positive", 0: "🔴 Negative"})
    display_disp["predicted_text"] = display_disp["prediction"].map(
        {1: "🟢 Positive", 0: "🔴 Negative"}
    )
    display_disp["confidence"] = display_disp.apply(
        lambda r: f"{r['probability']:.1%}" if r["prediction"] == 1
        else f"{1 - r['probability']:.1%}",
        axis=1,
    )

    st.dataframe(
        display_disp[["text_short", "actual", "predicted_text", "confidence"]],
        column_config={
            "text_short": st.column_config.TextColumn("Review (truncated)", width="large"),
            "actual": "True Sentiment",
            "predicted_text": "Predicted",
            "confidence": "Confidence",
        },
        height=400,
        use_container_width=True,
        hide_index=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — Live Test
# ═══════════════════════════════════════════════════════════════════════════
with tab_live:
    st.header("🎯 Test the Model Yourself")
    st.markdown(
        "Type or paste a movie review below and see what the model predicts."
    )

    review_input = st.text_area(
        "Your review",
        height=180,
        placeholder=(
            "e.g. This movie was absolutely fantastic! The acting was superb "
            "and the story kept me on the edge of my seat."
        ),
    )

    if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
        if not review_input.strip():
            st.warning("Please enter a review to analyze.")
        else:
            with st.spinner("Analyzing…"):
                vec = load_vectorizer()
                model = load_model()

                cleaned = clean_text(review_input)
                features = vec.transform([cleaned])
                proba = model.predict_proba(features)[0]
                pred = model.predict(features)[0]

            # ── Result display ────────────────────────────────────────────
            result_col, prob_col = st.columns([1, 2])

            with result_col:
                if pred == 1:
                    st.success("### 🟢 Positive Sentiment")
                else:
                    st.error("### 🔴 Negative Sentiment")

            with prob_col:
                pos_prob = proba[1]
                neg_prob = proba[0]
                st.markdown("**Confidence**")
                st.progress(
                    float(max(pos_prob, neg_prob)),
                    text=f"{(max(pos_prob, neg_prob) * 100):.1f}% confident",
                )

            # ── Probability breakdown ─────────────────────────────────────
            st.divider()
            st.subheader("Probability Breakdown")
            prob_df = pd.DataFrame(
                {
                    "Sentiment": ["Positive 🟢", "Negative 🔴"],
                    "Probability": [f"{pos_prob:.1%}", f"{neg_prob:.1%}"],
                }
            )
            st.dataframe(prob_df, hide_index=True, use_container_width=True)

            # ── Show the cleaned text ─────────────────────────────────────
            with st.expander("🔧 See preprocessed text (after cleaning & stemming)"):
                st.code(cleaned if cleaned else "(empty after preprocessing)", language="text")

            # ── Similar reviews from dataset ──────────────────────────────
            st.divider()
            st.subheader("📌 Similar Reviews from Test Set")
            preds_df = load_predictions()
            same_pred = preds_df[preds_df["prediction"] == pred].head(3)
            for _, row in same_pred.iterrows():
                label = "🟢 Positive" if row["label"] == 1 else "🔴 Negative"
                correct = "✅" if row["label"] == row["prediction"] else "❌"
                st.markdown(
                    f"> {row['text'][:250]}…  \n"
                    f"> *Actual: {label} {correct}*"
                )
                st.markdown("---")

    else:
        st.info("👆 Type a review above and click **Analyze Sentiment** to get started!")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — About
# ═══════════════════════════════════════════════════════════════════════════
with tab_about:
    st.header("ℹ️ About This Project")

    st.markdown("""
    ### 🎬 NLP Sentiment Analysis — IMDB Movie Reviews

    **Project 2** of the Data Analysis Portfolio — a complete end-to-end NLP
    pipeline that classifies movie reviews as **positive** or **negative**.

    #### 📊 Dataset
    [Stanford IMDB Large Movie Review Dataset](https://huggingface.co/datasets/stanfordnlp/imdb)
    — 50,000 highly polarised reviews (25k train / 25k test), balanced 50/50.

    #### 🔬 Pipeline

    ```
    Raw Text → Clean & Preprocess → TF-IDF → Train Classifiers → Evaluate
    ```

    1. **EDA** — word clouds, review length distribution, class balance
    2. **Preprocessing** — HTML removal, lowercase, punctuation/digit removal,
       stop-word filtering, Porter stemming
    3. **Feature extraction** — TF-IDF (5,000 features, unigrams + bigrams)
    4. **Training** — 3 classifiers compared:
       - Logistic Regression 🏆 **88.1% accuracy**
       - Multinomial Naive Bayes (~85%)
       - Random Forest (~84%)
    5. **Evaluation** — accuracy, precision, recall, F1, confusion matrices,
       ROC-AUC

    #### 🛠️ Tech Stack
    | Tool | Purpose |
    |------|---------|
    | HuggingFace Datasets | Data loading |
    | Scikit-learn | TF-IDF, classifiers, metrics |
    | NLTK | Stop words, Porter stemmer |
    | WordCloud, Plotly | Visualisations |
    | Streamlit | Interactive dashboard |

    #### 🔮 Future Improvements
    - Deep learning (LSTM / Transformer)
    - Pre-trained embeddings (GloVe, BERT)
    - Multi-class sentiment (very positive, neutral, very negative)
    - Cross-validation for robust evaluation

    #### 📝 License
    MIT — feel free to use, modify, and share.
    """)

    st.divider()
    st.caption(f"Built with ❤️ using Streamlit · {BASE.name} v1.0")
