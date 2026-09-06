"""Offline smoke tests for the NLP Sentiment Analysis pipeline.

Uses a tiny synthetic corpus so tests run fast in CI without downloading
the IMDB dataset. Verifies preprocessing, TF-IDF vectorization, and
classifier training that mirror the notebook / app.py pipeline.
"""

import re

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer

    nltk.download("stopwords", quiet=True)
    _stemmer = PorterStemmer()
    _stop_words = set(stopwords.words("english"))
    HAS_NLTK = True
except Exception:
    HAS_NLTK = False
    _stemmer = None
    _stop_words = set()


def clean_text(text: str) -> str:
    """Mirrors app.py clean_text — HTML strip, lower, alpha-only, stop+stem."""
    text = re.sub(r"<[^>]+>", " ", text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    if HAS_NLTK and _stemmer is not None:
        tokens = [_stemmer.stem(t) for t in text.split() if t not in _stop_words and len(t) > 2]
    else:
        tokens = [t for t in text.split() if len(t) > 2]
    return " ".join(tokens)


@pytest.fixture
def small_corpus():
    """12 synthetic IMDB-like reviews — 6 positive, 6 negative, with HTML noise."""
    positives = [
        "I absolutely loved this movie! Brilliant acting and wonderful story. <br />",
        "Fantastic film, great direction, amazing cinematography!!!",
        "Best movie ever, I enjoyed every minute of it.",
        "Wonderful performance by the cast, highly recommended.",
        "An excellent masterpiece, beautiful and touching.",
        "Superb! Loved the plot and the characters were great.",
    ]
    negatives = [
        "Terrible movie, worst film I have ever seen. Boring and dull.",
        "Awful acting, horrible script, I hated it.",
        "Disappointing and tedious, waste of time.",
        "Bad direction, poor story, completely boring.",
        "Horrible movie, not worth watching at all.",
        "Dull, slow, and utterly disappointing film.",
    ]
    texts = positives + negatives
    labels = [1] * 6 + [0] * 6
    # Deterministic shuffle
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(texts))
    return [texts[i] for i in idx], [labels[i] for i in idx]


def test_clean_text_removes_html_and_punctuation():
    raw = "Great movie! <br /> Loved it... 10/10!!!"
    cleaned = clean_text(raw)
    assert "<" not in cleaned
    assert ">" not in cleaned
    assert "!" not in cleaned
    assert "/" not in cleaned
    assert cleaned == cleaned.lower()


def test_clean_text_stemming_and_stopwords():
    # 'loved' should stem, stopwords like 'the' and 'and' removed, short words removed
    raw = "The movie was loved and wonderful"
    cleaned = clean_text(raw)
    # 'the'/'and'/'was' are stopwords -> removed; 'loved' -> 'love' if nltk available
    assert "the" not in cleaned.split()
    assert "and" not in cleaned.split()
    assert len(cleaned) > 0
    # At least 2 tokens remain (movie + love/wonderful)
    assert len(cleaned.split()) >= 2


def test_tfidf_shape_and_vocab(small_corpus):
    texts, _ = small_corpus
    cleaned = [clean_text(t) for t in texts]
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
    X = vec.fit_transform(cleaned)
    assert X.shape[0] == 12
    assert 5 <= X.shape[1] <= 5000
    assert len(vec.get_feature_names_out()) == X.shape[1]
    # No empty rows
    assert X.nnz > 0


def test_tfidf_bigram_includes_phrases(small_corpus):
    texts, _ = small_corpus
    cleaned = [clean_text(t) for t in texts]
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
    vec.fit_transform(cleaned)
    vocab = set(vec.get_feature_names_out())
    # At least one bigram (contains space) should exist given 12 docs
    bigrams = [w for w in vocab if " " in w]
    assert len(bigrams) > 0


def test_classifiers_train_and_predict(small_corpus):
    texts, labels = small_corpus
    cleaned = [clean_text(t) for t in texts]
    vec = TfidfVectorizer(max_features=100, ngram_range=(1, 2), stop_words="english")
    X = vec.fit_transform(cleaned)
    y = np.array(labels)
    # Logistic Regression
    lr = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(X, y)
    pred_lr = lr.predict(X)
    assert pred_lr.shape == y.shape
    assert set(pred_lr.tolist()).issubset({0, 1})
    # MultinomialNB
    nb = MultinomialNB()
    nb.fit(X, y)
    pred_nb = nb.predict(X)
    assert pred_nb.shape == y.shape
    # RandomForest
    rf = RandomForestClassifier(n_estimators=20, random_state=42, max_depth=5)
    rf.fit(X, y)
    pred_rf = rf.predict(X)
    assert pred_rf.shape == y.shape


def test_accuracy_and_confusion_matrix(small_corpus):
    texts, labels = small_corpus
    cleaned = [clean_text(t) for t in texts]
    vec = TfidfVectorizer(max_features=100, ngram_range=(1, 1), stop_words="english")
    X = vec.fit_transform(cleaned)
    y = np.array(labels)
    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(X, y)
    pred = clf.predict(X)
    acc = accuracy_score(y, pred)
    assert 0.0 <= acc <= 1.0
    # Training accuracy on tiny diverse set should be reasonably high (memorization)
    assert acc >= 0.6
    cm = confusion_matrix(y, pred)
    assert cm.shape == (2, 2)
    assert cm.sum() == len(y)


def test_vectorizer_deterministic(small_corpus):
    texts, _ = small_corpus
    cleaned = [clean_text(t) for t in texts]
    vec1 = TfidfVectorizer(max_features=50, ngram_range=(1, 1), stop_words="english")
    vec2 = TfidfVectorizer(max_features=50, ngram_range=(1, 1), stop_words="english")
    X1 = vec1.fit_transform(cleaned)
    X2 = vec2.fit_transform(cleaned)
    assert X1.shape == X2.shape
    assert (X1.toarray() == X2.toarray()).all()
