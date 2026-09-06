"""Offline smoke tests for the Topic Modeling pipeline (20 Newsgroups).

Uses 60 synthetic documents with planted vocab buckets so tests run fast in
CI without downloading fetch_20newsgroups. Verifies vectorization, NMF / LDA /
TruncatedSVD training, top-word extraction, and coherence scoring.
"""

import numpy as np
import pytest
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

RANDOM_STATE = 42
N_TOPICS = 3
N_TOP_WORDS = 8


@pytest.fixture
def synthetic_docs():
    """60 docs from 3 planted topics: space / hockey / politics."""
    rng = np.random.default_rng(RANDOM_STATE)
    vocab = {
        "space": ["space", "nasa", "orbit", "satellite", "launch", "planet", "galaxy", "telescope", "rocket"],
        "hockey": ["hockey", "game", "team", "player", "goal", "season", "league", "coach", "score"],
        "politics": ["government", "policy", "election", "senate", "vote", "law", "president", "congress", "bill"],
    }
    filler = ["the", "is", "and", "with", "for", "this", "that", "very", "more", "about"]
    topics = ["space", "hockey", "politics"]
    docs = []
    labels = []
    for i in range(60):
        topic = topics[i % 3]
        words = list(rng.choice(vocab[topic], 12)) + list(rng.choice(filler, 8))
        rng.shuffle(words)
        docs.append(" ".join(words))
        labels.append(topics.index(topic))
    return docs, labels


def test_vectorizer_shapes(synthetic_docs):
    docs, _ = synthetic_docs
    tfidf = TfidfVectorizer(max_features=100, max_df=0.95, min_df=2, stop_words="english", ngram_range=(1, 2))
    X_tfidf = tfidf.fit_transform(docs)
    assert X_tfidf.shape[0] == 60
    assert 10 <= X_tfidf.shape[1] <= 100
    assert len(tfidf.get_feature_names_out()) == X_tfidf.shape[1]

    count = CountVectorizer(max_features=100, max_df=0.95, min_df=2, stop_words="english")
    X_count = count.fit_transform(docs)
    assert X_count.shape[0] == 60
    assert X_count.shape[1] >= 5


def test_nmf_topics_shape(synthetic_docs):
    docs, _ = synthetic_docs
    tfidf = TfidfVectorizer(max_features=50, stop_words="english")
    X = tfidf.fit_transform(docs)
    model = NMF(n_components=N_TOPICS, random_state=RANDOM_STATE, max_iter=200, init="nndsvdar")
    W = model.fit_transform(X)
    H = model.components_
    assert W.shape == (60, N_TOPICS)
    assert H.shape == (N_TOPICS, X.shape[1])
    # All non-negative
    assert (W >= 0).all()
    assert (H >= 0).all()


def test_lda_topics_shape(synthetic_docs):
    docs, _ = synthetic_docs
    count = CountVectorizer(max_features=50, stop_words="english")
    X = count.fit_transform(docs)
    lda = LatentDirichletAllocation(
        n_components=N_TOPICS, random_state=RANDOM_STATE, max_iter=20, learning_method="batch"
    )
    W = lda.fit_transform(X)
    assert W.shape == (60, N_TOPICS)
    assert lda.components_.shape == (N_TOPICS, X.shape[1])
    # Each doc topic distribution sums to ~1
    row_sums = W.sum(axis=1)
    assert np.allclose(row_sums, 1, atol=1e-5)


def test_truncated_svd_shape(synthetic_docs):
    docs, _ = synthetic_docs
    tfidf = TfidfVectorizer(max_features=50, stop_words="english")
    X = tfidf.fit_transform(docs)
    svd = TruncatedSVD(n_components=N_TOPICS, random_state=RANDOM_STATE)
    W = svd.fit_transform(X)
    assert W.shape == (60, N_TOPICS)
    assert svd.components_.shape == (N_TOPICS, X.shape[1])
    assert svd.singular_values_.shape == (N_TOPICS,)


def _coherence_score(components, dtm, n_top=5):
    """Tiny coherence proxy: avg pairwise co-document rate for top words."""
    dtm_binary = (dtm > 0).astype(float)
    # Convert to dense for small synthetic dtm
    if hasattr(dtm_binary, "toarray"):
        dtm_binary = dtm_binary.toarray()
    scores = []
    for t in range(components.shape[0]):
        top = components[t].argsort()[-n_top:][::-1]
        pairs = []
        for i in range(len(top)):
            col_i = dtm_binary[:, top[i]]
            for j in range(i + 1, len(top)):
                col_j = dtm_binary[:, top[j]]
                both = np.logical_and(col_i, col_j).sum()
                wi = col_i.sum()
                if wi > 0:
                    pairs.append(both / wi)
        scores.append(float(np.mean(pairs)) if pairs else 0.0)
    return scores


def test_coherence_score_range(synthetic_docs):
    docs, _ = synthetic_docs
    count = CountVectorizer(max_features=50, stop_words="english")
    X = count.fit_transform(docs)
    tfidf = TfidfVectorizer(max_features=50, stop_words="english")
    X_tfidf = tfidf.fit_transform(docs)
    nmf = NMF(n_components=N_TOPICS, random_state=RANDOM_STATE, max_iter=200, init="nndsvdar")
    nmf.fit(X_tfidf)
    scores = _coherence_score(nmf.components_, X)
    assert len(scores) == N_TOPICS
    for s in scores:
        assert 0.0 <= s <= 1.0


def test_top_words_extraction(synthetic_docs):
    docs, _ = synthetic_docs
    tfidf = TfidfVectorizer(max_features=50, stop_words="english")
    X = tfidf.fit_transform(docs)
    feature_names = tfidf.get_feature_names_out()
    nmf = NMF(n_components=N_TOPICS, random_state=RANDOM_STATE, max_iter=200, init="nndsvdar")
    nmf.fit(X)
    # Extract top words per topic
    for t in range(N_TOPICS):
        top_idx = nmf.components_[t].argsort()[-N_TOP_WORDS:][::-1]
        assert len(top_idx) == N_TOP_WORDS
        for idx in top_idx:
            assert 0 <= idx < len(feature_names)
            assert isinstance(feature_names[idx], str)


def test_document_topic_distribution_properties(synthetic_docs):
    docs, _ = synthetic_docs
    tfidf = TfidfVectorizer(max_features=50, stop_words="english")
    X = tfidf.fit_transform(docs)
    nmf = NMF(n_components=N_TOPICS, random_state=RANDOM_STATE, max_iter=200, init="nndsvdar")
    W = nmf.fit_transform(X)
    # Dominant topic per doc is in [0, N_TOPICS)
    dominant = np.argmax(W, axis=1)
    assert set(dominant.tolist()).issubset({0, 1, 2})
    # Confidence (max prob / max weight) should be > 0
    max_vals = W.max(axis=1)
    assert (max_vals > 0).all()
    # LDA sums to 1, NMF mean > 0 — basic sanity
    assert W.mean() > 0
