"""Tests for the hybrid RAG document-QA pipeline.

Runs on tiny, deterministic synthetic data with no downloads. Verifies chunking,
both retrievers, Reciprocal Rank Fusion, the extractive answer baseline, and the
offline evaluation metrics.
"""

import numpy as np
import pandas as pd
import pytest
import rag_analysis as rag


@pytest.fixture
def tiny_corpus():
    """A 3-document corpus with clear, separable vocabulary."""
    docs = [
        rag.Doc("a", "Apples", "apple fruit orchard harvest red sweet"),
        rag.Doc("b", "Rockets", "rocket engine thrust orbit space launch fuel"),
        rag.Doc("c", "Rivers", "river water stream flow bank fish salmon"),
    ]
    rows = []
    for d in docs:
        for i, chunk in enumerate(rag.chunk_document(d.text)):
            rows.append(
                {
                    "doc_id": d.doc_id,
                    "title": d.title,
                    "text": d.text,
                    "chunk_id": f"{d.doc_id}#{i}",
                    "chunk": chunk,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def fitted(tiny_corpus):
    """Fitted BM25 + dense retrievers + a shared embedder over the tiny corpus."""
    emb = rag.make_embeddings(tiny_corpus["chunk"].tolist())
    bm25 = rag.BM25Retriever().fit(tiny_corpus)
    dense = rag.FaissDenseRetriever()
    try:
        dense.fit(emb, tiny_corpus["chunk_id"].tolist())
    except Exception:
        dense.fallback_fit(emb, tiny_corpus["chunk_id"].tolist())
    embedder = rag.Embedder().fit(tiny_corpus["chunk"].tolist())
    return tiny_corpus, bm25, dense, embedder


def test_chunk_document_respects_size_and_overlap():
    text = " ".join(f"w{i}" for i in range(100))
    chunks = rag.chunk_document(text, chunk_size=20, overlap=5)
    assert len(chunks) >= 4
    # Token count per chunk (except possibly the last) == chunk_size.
    assert all(len(c.split()) == 20 for c in chunks[:-1])
    # Overlap preserved content: end of one chunk appears at start of next.
    assert chunks[0].split()[-5:] == chunks[1].split()[:5]


def test_chunk_document_short_text_single_chunk():
    assert rag.chunk_document("a b c", chunk_size=20, overlap=5) == ["a b c"]


def test_chunk_document_bad_params():
    with pytest.raises(ValueError):
        rag.chunk_document("a b c d", chunk_size=5, overlap=5)


def test_embeddings_shape_and_normalised(fitted):
    _, _, _, embedder = fitted
    emb = embedder.transform(["apple fruit", "rocket engine"])
    assert emb.shape[0] == 2
    # TF-IDF->SVD path yields unit-norm rows (Normalizer).
    norms = np.linalg.norm(emb, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_bm25_ranks_relevant_first(fitted):
    _, bm25, _, _ = fitted
    res = bm25.search("rocket orbit launch", k=3)
    assert res[0][1].startswith("b#")  # rockets doc wins


def test_dense_ranks_relevant_first(fitted):
    _, _, dense, embedder = fitted
    q = embedder.transform(["river water fish"])
    res = dense.search(q, k=3)
    assert res[0][1].startswith("c#")  # rivers doc wins


def test_rrf_combines_and_ranks():
    fused = rag.reciprocal_rank_fusion([["x", "y", "z"], ["y", "x", "w"]])
    # 'x' and 'y' appear in both lists (rank 0 in one, 1 in the other) so they
    # tie at the top; both must outrank single-list items 'z' and 'w'.
    assert fused["x"] >= fused["y"]
    assert fused["x"] > fused["z"]
    assert fused["y"] > fused["w"]


def test_hybrid_retrieve_returns_k_with_text(fitted):
    corpus, bm25, dense, embedder = fitted
    out = rag.hybrid_retrieve("rocket launch", bm25, dense, corpus, embedder, k=2)
    assert len(out) == 2
    for score, cid, text in out:
        assert isinstance(score, float)
        assert text  # text must be populated from the corpus
    # Top result should be the rockets doc for this query.
    assert out[0][1].startswith("b#")


def test_extract_answer_picks_overlapping_sentence(fitted):
    _, bm25, dense, _ = fitted
    # Build a toy passage set and ensure extractor returns a non-empty string.
    passages = [
        "A rocket is a vehicle that uses engine thrust to travel into space.",
        "Apples are a sweet red fruit grown in orchards.",
    ]
    ans = rag.extract_answer("rocket engine thrust space", passages)
    assert "rocket" in ans.lower()


def test_extract_answer_fallback_on_no_overlap():
    # No query-term overlap -> returns leading slice of first passage.
    ans = rag.extract_answer("zzz qqq", ["First passage with no match here."])
    assert ans.startswith("First passage")


def test_evaluate_retrieval_gold_recall(fitted):
    corpus, bm25, dense, embedder = fitted
    queries = [
        {"question": "rocket orbit launch fuel", "gold": "b"},
        {"question": "river water fish salmon", "gold": "c"},
        {"question": "apple fruit orchard harvest", "gold": "a"},
    ]
    metrics = rag.evaluate_retrieval(queries, bm25, dense, corpus, embedder, ks=(1, 3))
    # Hybrid must hit gold at@1 in this separable synthetic corpus.
    assert metrics["hybrid"]["recall@1"] == 100.0
    assert metrics["hybrid"]["mrr"] >= metrics["sparse"]["mrr"]
    assert 0.0 < metrics["hybrid"]["mrr"] <= 1.0


def test_run_pipeline_real_corpus_smoke():
    """Run the full offline pipeline on the built-in corpus (no network)."""
    out = rag.run_pipeline()
    assert out["n_documents"] == len(rag.CORPUS)
    assert out["n_passages"] > out["n_documents"]
    assert "hybrid" in out["metrics"]
    assert out["sample_answer"]
    assert out["sample_sources"]
