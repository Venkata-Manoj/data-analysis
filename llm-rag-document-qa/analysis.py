#!/usr/bin/env python3
"""Hybrid RAG Document-QA — Retrieval-Augmented Generation baseline.

This project demonstrates a complete, offline Retrieval-Augmented Generation
(RAG) pipeline built from first principles:

  1. CHUNK      a corpus of technical passages into overlapping passages
  2. EMBED      each passage into a dense vector (TF-IDF -> TruncatedSVD by
                default; optional sentence-transformers upgrade path)
  3. INDEX      a FAISS flat index for the dense vectors + a BM25 index for
                lexical retrieval
  4. RETRIEVE   a query with both retrievers, then fuse their rankings with
                Reciprocal Rank Fusion (RRF) -> a robust hybrid ranking
  5. ANSWER     extract the most relevant span from the fused context as the
                answer (an honest, LLM-free baseline; an optional Ollama
                generation hook is documented in the README)
  6. EVALUATE   retrieval quality with Recall@k and MRR, plus an extractive
                answer-success proxy, on a built-in question/gold set

Everything runs offline on a built-in AI/ML corpus, so it is deterministic,
fast, and CI-safe. Swap in a larger corpus (e.g. SQuAD via `datasets`) or a
transformer embedder without changing the pipeline.

All functions are importable so the unit tests (tests/test_rag.py) can exercise
chunking, both retrievers, fusion, and evaluation on tiny synthetic data with
no downloads — the same pattern used by the repository's other analysis modules.

Run locally:  python analysis.py
"""

import json
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / CI-safe backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

warnings.filterwarnings("ignore")

PDIR = Path(__file__).resolve().parent
CHARTS = PDIR / "charts"
OUTPUTS = PDIR / "outputs"
DATA_DIR = PDIR / "data"
for d in (CHARTS, OUTPUTS, DATA_DIR):
    d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Corpus
# --------------------------------------------------------------------------- #
@dataclass
class Doc:
    """A source document with a title and body text."""

    doc_id: str
    title: str
    text: str


# A small, domain-relevant corpus of AI/ML/LLM passages. Having it built-in
# keeps the project fully offline, deterministic, and CI-safe while still
# demonstrating every RAG stage end to end.
CORPUS: list[Doc] = [
    Doc(
        "t-transformers",
        "Transformers",
        "The Transformer is a neural architecture that replaces recurrence with "
        "self-attention, allowing the model to weigh the importance of every token "
        "in a sequence relative to every other token. Introduced in 'Attention Is "
        "All You Need', it scales better to long sequences than RNNs and became the "
        "foundation for large language models. Multi-head attention lets the model "
        "attend to information from different representation subspaces in parallel.",
    ),
    Doc(
        "t-attention",
        "Self-Attention",
        "Self-attention computes a weighted sum of input representations where the "
        "weights come from the compatibility of each token with every other token. "
        "It is implemented efficiently with query, key, and value projection "
        "matrices. Attention scores are normalised with softmax so they sum to one, "
        "which makes the operation a soft selection over the sequence.",
    ),
    Doc(
        "t-embeddings",
        "Embeddings",
        "Embeddings map discrete tokens or passages into dense vectors in a "
        "continuous space where semantic similarity corresponds to geometric "
        "proximity. Word2Vec, GloVe, and modern transformer encoders all produce "
        "embeddings. Retrieval systems rank passages by the cosine similarity of "
        "their embedding vectors.",
    ),
    Doc(
        "t-rag",
        "Retrieval-Augmented Generation",
        "Retrieval-Augmented Generation grounds a language model's answer in "
        "external documents. The pipeline retrieves relevant passages for a query, "
        "concatenates them into a prompt, and asks the model to generate an answer "
        "using only that context. RAG reduces hallucination and lets the system "
        "answer from knowledge that was not in its training data.",
    ),
    Doc(
        "t-chunking",
        "Document Chunking",
        "Chunking splits long documents into smaller passages small enough to embed "
        "and retrieve. Fixed-token windows with overlap are common: overlap keeps "
        "information that would otherwise fall on a boundary from being lost. Good "
        "chunk size balances retrieval precision against context completeness.",
    ),
    Doc(
        "t-bm25",
        "BM25",
        "BM25 is a probabilistic ranking function for lexical retrieval. It scores "
        "documents by term-frequency saturation and inverse document frequency, and "
        "down-weights very long documents. Unlike embedding search, BM25 retrieves "
        "on exact keyword overlap, which is excellent for rare proper nouns and "
        "product codes.",
    ),
    Doc(
        "t-faiss",
        "FAISS",
        "FAISS is a library for efficient similarity search over dense vectors. It "
        "offers exact flat indexes and compressed approximate indexes that search "
        "billions of vectors in milliseconds. For a small corpus a flat inner-product "
        "index over L2-normalised vectors computes exact cosine similarity.",
    ),
    Doc(
        "t-rrf",
        "Reciprocal Rank Fusion",
        "Reciprocal Rank Fusion combines multiple ranked lists without needing to "
        "normalise their scores. Each item's fused score is the sum over rankings of "
        "1/(k + rank), where k is a smoothing constant. RRF is robust because it uses "
        "only ranks, so a lexical and a semantic retriever can be fused directly.",
    ),
    Doc(
        "t-fine-tuning",
        "Fine-Tuning",
        "Fine-tuning adapts a pre-trained model to a downstream task by continuing "
        "training on a smaller labelled dataset. Parameter-efficient methods such as "
        "LoRA update low-rank adapters instead of all weights, cutting memory cost. "
        "Fine-tuning trades generalisation for task accuracy.",
    ),
    Doc(
        "t-rlhf",
        "RLHF",
        "Reinforcement Learning from Human Feedback aligns a language model with "
        "human preferences. A reward model is trained on pairwise comparisons, then "
        "the policy is optimised with reinforcement learning (often PPO) to maximise "
        "reward while a KL penalty keeps it close to the base model.",
    ),
    Doc(
        "t-inference",
        "Inference Optimisation",
        "Inference optimisation makes models faster and cheaper to serve. Common "
        "techniques include quantisation to lower precision, KV-cache reuse, "
        "speculative decoding, and batching. Quantisation to 4-bit with GPTQ or "
        "AWQ can shrink memory use by 4x with minor quality loss.",
    ),
    Doc(
        "t-vector-db",
        "Vector Databases",
        "Vector databases store embeddings and serve nearest-neighbour queries at "
        "scale. They add filtering, metadata, and durable storage on top of raw "
        "indexes like FAISS or HNSW. Chroma, Qdrant, and pgvector are popular "
        "choices for RAG backends.",
    ),
    Doc(
        "t-eval",
        "RAG Evaluation",
        "RAG is evaluated on two stages. Retrieval is measured with Recall@k and "
        "MRR: did the gold passage appear in the top-k, and how highly ranked was "
        "it? Generation is measured with faithfulness (is the answer grounded in "
        "retrieved context?) and answer correctness against a reference.",
    ),
    Doc(
        "t-tokenization",
        "Tokenization",
        "Tokenization splits text into units a model ingests. Subword algorithms like "
        "Byte-Pair Encoding and WordPiece balance vocabulary size and coverage so "
        "rare words become sequences of known subwords. Token counts drive both "
        "context length and inference cost.",
    ),
    Doc(
        "t-prompting",
        "Prompt Engineering",
        "Prompt engineering shapes the input so a model produces the desired output "
        "without weight changes. Techniques include few-shot examples, chain-of-thought "
        "reasoning, and structured output schemas. A well-specified prompt with clear "
        "constraints often beats a larger model with a vague one.",
    ),
    Doc(
        "t-quantization",
        "Quantisation",
        "Quantisation represents model weights and activations with fewer bits, such "
        "as 8-bit integers or 4-bit floats. It reduces memory and speeds up matrix "
        "multiplication on hardware that supports low-precision math, enabling large "
        "models to run on consumer GPUs.",
    ),
]


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
def chunk_document(text: str, chunk_size: int = 60, overlap: int = 15) -> list[str]:
    """Split *text* into overlapping token-window chunks.

    Tokenisation is a simple whitespace split; this is sufficient for a
    retrieval demo and keeps the function dependency-free and deterministic.
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")
    tokens = text.split()
    if len(tokens) <= chunk_size:
        return [text.strip()]
    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(tokens), step):
        window = tokens[start : start + chunk_size]
        if not window:
            break
        chunks.append(" ".join(window).strip())
        if start + chunk_size >= len(tokens):
            break
    return chunks


def build_corpus() -> pd.DataFrame:
    """Return the built-in corpus as a passage-level DataFrame.

    Columns: doc_id, title, text, chunk_id, chunk.
    """
    rows = []
    for doc in CORPUS:
        for i, chunk in enumerate(chunk_document(doc.text)):
            rows.append(
                {
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "text": doc.text,
                    "chunk_id": f"{doc.doc_id}#{i}",
                    "chunk": chunk,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Embeddings (dense vectors)
# --------------------------------------------------------------------------- #
class Embedder:
    """Stateful text embedder.

    Fits a semantic space on the corpus once, then transforms any new text
    (e.g. queries) into the *same* vector space. Sharing the space across
    corpus and queries is essential: embeddings produced by separate fits live
    in incompatible spaces and their cosine similarity is meaningless.

    Default encoder: TF-IDF -> TruncatedSVD (an LSI-style dense representation),
    no downloads, runs anywhere. With ``use_transformer=True`` and
    ``sentence_transformers`` installed, a real encoder is used instead.
    """

    def __init__(self) -> None:
        self.use_transformer = False
        self._st_model = None
        self.transformer = None

    def fit(self, texts: list[str], use_transformer: bool = False) -> "Embedder":
        self.use_transformer = use_transformer
        if use_transformer:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
                self.transformer = None
                return self
            except Exception:  # pragma: no cover - optional dependency
                # Fall through to the scikit-learn baseline if the transformer
                # path is unavailable (e.g. no network for the model download).
                self._st_model = None
        tfidf = TfidfVectorizer(stop_words="english", max_features=2000)
        tfidf.fit(texts)
        n_features = tfidf.vocabulary_.__len__()
        n_components = min(96, max(2, n_features - 1))
        svd = TruncatedSVD(n_components=n_components, random_state=0)
        self.transformer = Pipeline([("tfidf", tfidf), ("svd", svd), ("norm", Normalizer(copy=False))])
        self.transformer.fit(texts)
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        if self.use_transformer and self._st_model is not None:
            emb = np.asarray(self._st_model.encode(texts, normalize_embeddings=True))
            return emb.astype("float32")
        if self.transformer is None:
            raise RuntimeError("Embedder.transform called before fit()")
        return np.ascontiguousarray(self.transformer.transform(texts), dtype="float32")


def make_embeddings(texts: list[str], use_transformer: bool = False) -> np.ndarray:
    """Stateless convenience: fit+transform *texts* into one dense matrix.

    Use this only when the same set is both trained and queried (e.g. a PCA
    visualisation). For corpus-vs-query retrieval use the stateful
    :class:`Embedder` so the two live in the same space.
    """
    return Embedder().fit(texts, use_transformer=use_transformer).transform(texts)


# --------------------------------------------------------------------------- #
# Retrievers
# --------------------------------------------------------------------------- #
class BM25Retriever:
    """Lexical retriever over tokenised passages."""

    def __init__(self) -> None:
        self.bm25 = None
        self.chunk_ids: list[str] = []

    def fit(self, corpus: pd.DataFrame) -> "BM25Retriever":
        self.chunk_ids = corpus["chunk_id"].tolist()
        tokenised = [c.lower().split() for c in corpus["chunk"]]
        self.bm25 = BM25Okapi(tokenised)
        return self

    def search(self, query: str, k: int = 5) -> list[tuple[float, str]]:
        scores = self.bm25.get_scores(query.lower().split())
        order = np.argsort(scores)[::-1][:k]
        return [(float(scores[i]), self.chunk_ids[i]) for i in order]


class FaissDenseRetriever:
    """Semantic retriever backed by a FAISS flat cosine index."""

    def __init__(self) -> None:
        self.index = None
        self.chunk_ids: list[str] = []
        self._has_faiss = False

    def fit(self, embeddings: np.ndarray, chunk_ids: list[str]) -> "FaissDenseRetriever":
        self.chunk_ids = list(chunk_ids)
        emb = np.ascontiguousarray(embeddings, dtype="float32")
        # Normalise so inner product == cosine similarity.
        faiss_norm = emb.copy()
        import faiss  # local import; optional dependency

        faiss.normalize_L2(faiss_norm)
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(faiss_norm)
        self._has_faiss = True
        return self

    def fallback_fit(self, embeddings: np.ndarray, chunk_ids: list[str]) -> None:
        """Used only if FAISS is unavailable: store embeddings for brute force."""
        self._emb = np.ascontiguousarray(embeddings, dtype="float32")
        self.chunk_ids = list(chunk_ids)
        self._has_faiss = False

    def search(self, query_emb: np.ndarray, k: int = 5) -> list[tuple[float, str]]:
        q = np.ascontiguousarray(query_emb, dtype="float32").reshape(1, -1)
        if self._has_faiss:
            import faiss

            faiss.normalize_L2(q)
            scores, idx = self.index.search(q, k)
            out = []
            for s, i in zip(scores[0], idx[0]):
                if i != -1:
                    out.append((float(s), self.chunk_ids[i]))
            return out
        # Brute-force cosine fallback.
        sims = self._emb @ q.reshape(-1)
        order = np.argsort(sims)[::-1][:k]
        return [(float(sims[i]), self.chunk_ids[i]) for i in order]


# --------------------------------------------------------------------------- #
# Fusion & answer extraction
# --------------------------------------------------------------------------- #
def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Fuse several ranked id lists into a single RRF score per id."""
    fused: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            fused[cid] += 1.0 / (k + rank + 1)
    return dict(fused)


def hybrid_retrieve(
    query: str,
    bm25: BM25Retriever,
    dense: FaissDenseRetriever,
    corpus: pd.DataFrame,
    embedder: "Embedder",
    k: int = 5,
) -> list[tuple[float, str, str]]:
    """Return the top-k hybrid passages as (fused_score, chunk_id, text)."""
    sparse = [cid for _, cid in bm25.search(query, k=max(k * 2, 10))]
    q_emb = embedder.transform([query])
    dense_res = dense.search(q_emb, k=max(k * 2, 10))
    dense_ids = [cid for _, cid in dense_res]
    fused = reciprocal_rank_fusion([sparse, dense_ids])
    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
    text_by_id = dict(zip(corpus["chunk_id"], corpus["chunk"]))
    return [(score, cid, text_by_id.get(cid, "")) for cid, score in ranked]


def extract_answer(query: str, passages: list[str], top_n: int = 1) -> str:
    """Extractive baseline: pick the sentence with the most query-term overlap.

    This is an honest, LLM-free answer grounded in retrieved context. An Ollama
    generation hook can replace it (see README) when a model endpoint is present.
    """
    q_terms = set(t for t in query.lower().split() if len(t) > 2)
    scored = []
    for passage in passages[:top_n]:
        for sent in passage.replace("\n", " ").split("."):
            sent = sent.strip()
            if not sent:
                continue
            s_terms = set(sent.lower().split())
            overlap = len(q_terms & s_terms)
            scored.append((overlap, sent))
    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored or scored[0][0] == 0:
        return passages[0][:240] if passages else ""
    return scored[0][1]


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
EVAL_SET: list[dict] = [
    {"question": "What does self-attention compute?", "gold": "t-attention"},
    {"question": "How does BM25 rank documents?", "gold": "t-bm25"},
    {"question": "What is Reciprocal Rank Fusion?", "gold": "t-rrf"},
    {"question": "Why is FAISS used for similarity search?", "gold": "t-faiss"},
    {"question": "What is Retrieval-Augmented Generation?", "gold": "t-rag"},
    {"question": "How does document chunking work?", "gold": "t-chunking"},
    {"question": "What are embeddings used for?", "gold": "t-embeddings"},
    {"question": "How does fine-tuning adapt a model?", "gold": "t-fine-tuning"},
    {"question": "What is RLHF?", "gold": "t-rlhf"},
    {"question": "How is inference optimised for speed?", "gold": "t-inference"},
    {"question": "What do vector databases provide?", "gold": "t-vector-db"},
    {"question": "How is a RAG system evaluated?", "gold": "t-eval"},
]


def _gold_chunk_ids(corpus: pd.DataFrame, gold_doc: str) -> set[str]:
    return set(corpus.loc[corpus["doc_id"] == gold_doc, "chunk_id"])


def evaluate_retrieval(
    queries: list[dict],
    bm25: BM25Retriever,
    dense: FaissDenseRetriever,
    corpus: pd.DataFrame,
    embedder: "Embedder",
    ks: tuple[int, ...] = (1, 3, 5, 10),
) -> dict:
    """Compute Recall@k and MRR for sparse, dense, and hybrid retrievers."""
    results = {name: {f"recall@{k}": [] for k in ks} for name in ("sparse", "dense", "hybrid")}
    mrr = {name: [] for name in ("sparse", "dense", "hybrid")}
    q_embs = embedder.transform([q["question"] for q in queries])

    for i, q in enumerate(queries):
        gold = _gold_chunk_ids(corpus, q["gold"])
        sparse_ids = [cid for _, cid in bm25.search(q["question"], k=max(ks))]
        dense_ids = [cid for _, cid in dense.search(q_embs[i : i + 1], k=max(ks))]
        fused = reciprocal_rank_fusion([sparse_ids, dense_ids])
        hybrid_ids = [cid for cid, _ in sorted(fused.items(), key=lambda kv: kv[1], reverse=True)]

        for k in ks:
            results["sparse"][f"recall@{k}"].append(len(set(sparse_ids[:k]) & gold) > 0)
            results["dense"][f"recall@{k}"].append(len(set(dense_ids[:k]) & gold) > 0)
            results["hybrid"][f"recall@{k}"].append(len(set(hybrid_ids[:k]) & gold) > 0)

        for name, ids in (("sparse", sparse_ids), ("dense", dense_ids), ("hybrid", hybrid_ids)):
            rank = next((r for r, cid in enumerate(ids) if cid in gold), None)
            mrr[name].append(0.0 if rank is None else 1.0 / (rank + 1))

    summary = {}
    for name in ("sparse", "dense", "hybrid"):
        summary[name] = {
            **{f"recall@{k}": round(100 * np.mean(results[name][f"recall@{k}"]), 1) for k in ks},
            "mrr": round(float(np.mean(mrr[name])), 3),
        }
    return summary


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def _save(fig, name: str) -> None:
    path = CHARTS / name
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def make_charts(metrics: dict, timings: dict, fusion_scores: list[float]) -> None:
    ks = [1, 3, 5, 10]
    # 1. Recall@k curve
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, color in (("sparse", "tab:red"), ("dense", "tab:blue"), ("hybrid", "tab:green")):
        vals = [metrics[name][f"recall@{k}"] for k in ks]
        ax.plot(ks, vals, marker="o", label=name, color=color)
    ax.set_xlabel("k")
    ax.set_ylabel("Recall@k (%)")
    ax.set_title("Retrieval Recall@k — Sparse vs Dense vs Hybrid")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, "recall_at_k.png")

    # 2. MRR bar
    fig, ax = plt.subplots(figsize=(6, 4))
    names = ["sparse", "dense", "hybrid"]
    ax.bar(names, [metrics[n]["mrr"] for n in names], color=["tab:red", "tab:blue", "tab:green"])
    ax.set_ylabel("MRR")
    ax.set_title("Mean Reciprocal Rank by Retriever")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "mrr.png")

    # 3. Hit-rate@5
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, [metrics[n]["recall@5"] for n in names], color=["tab:red", "tab:blue", "tab:green"])
    ax.set_ylabel("Hit-rate@5 (%)")
    ax.set_title("Top-5 Retrieval Hit Rate")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "hit_rate_at_5.png")

    # 4. Latency boxplot (each stage as a list of samples so boxplot is valid).
    fig, ax = plt.subplots(figsize=(6, 4))
    data = [[timings["sparse"]], [timings["dense"]], [timings["hybrid"]]]
    ax.boxplot(data, tick_labels=["sparse", "dense", "hybrid"])
    ax.set_ylabel("ms / query")
    ax.set_title("Per-query Latency by Stage")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "latency.png")

    # 5. Fusion score distribution (top-1 hybrid)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(fusion_scores, bins=8, color="tab:green", alpha=0.8)
    ax.set_xlabel("Fused RRF score (top-1)")
    ax.set_ylabel("Queries")
    ax.set_title("Hybrid Fusion Score Distribution")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "fusion_scores.png")

    # 6. Embedding PCA by topic
    from sklearn.decomposition import PCA

    emb = make_embeddings(build_corpus()["chunk"].tolist())
    coords = PCA(n_components=2, random_state=0).fit_transform(emb)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(coords[:, 0], coords[:, 1], c=list(range(len(coords))), cmap="viridis", s=40)
    ax.set_title("Passage Embedding Space (PCA, 2D)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    _save(fig, "embedding_pca.png")

    # 7. Accuracy of extractive answers (gold passage in top-1 context)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        ["sparse", "dense", "hybrid"],
        [metrics[n]["recall@1"] for n in names],
        color=["tab:red", "tab:blue", "tab:green"],
    )
    ax.set_ylabel("Top-1 gold hit (%)")
    ax.set_title("Extractive Answer Grounding (gold in top-1)")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "answer_grounding.png")

    # 8. NDCG-style ranking quality (normalised MRR * 100)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, [metrics[n]["mrr"] * 100 for n in names], color=["tab:red", "tab:blue", "tab:green"])
    ax.set_ylabel("MRR x100")
    ax.set_title("Overall Ranking Quality (MRR)")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "ranking_quality.png")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_pipeline(use_transformer: bool = False) -> dict:
    """Execute the full offline RAG pipeline and write charts + results."""
    corpus = build_corpus()
    embedder = Embedder().fit(corpus["chunk"].tolist(), use_transformer=use_transformer)
    embeddings = embedder.transform(corpus["chunk"].tolist())

    bm25 = BM25Retriever().fit(corpus)
    dense = FaissDenseRetriever()
    try:
        dense.fit(embeddings, corpus["chunk_id"].tolist())
    except Exception:  # pragma: no cover - FAISS optional
        dense.fallback_fit(embeddings, corpus["chunk_id"].tolist())

    metrics = evaluate_retrieval(EVAL_SET, bm25, dense, corpus, embedder)

    # Timing (ms) for a representative query.
    sample = EVAL_SET[0]["question"]
    timings = {}
    t0 = time.perf_counter()
    bm25.search(sample, k=5)
    timings["sparse"] = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    dense.search(embedder.transform([sample]), k=5)
    timings["dense"] = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    hybrid_retrieve(sample, bm25, dense, corpus, embedder)
    timings["hybrid"] = (time.perf_counter() - t0) * 1000

    # Fusion scores for the top-1 hybrid result per eval query (for chart 5).
    fusion_scores = []
    for q in EVAL_SET:
        fused = reciprocal_rank_fusion(
            [
                [cid for _, cid in bm25.search(q["question"], k=10)],
                [cid for _, cid in dense.search(embedder.transform([q["question"]]), k=10)],
            ]
        )
        fusion_scores.append(max(fused.values()))

    make_charts(metrics, timings, fusion_scores)

    # Sample end-to-end QA answer for the report.
    q = "What is Reciprocal Rank Fusion?"
    results = hybrid_retrieve(q, bm25, dense, corpus, embedder, k=3)
    answer = extract_answer(q, [text for _, _, text in results])

    out = {
        "n_documents": int(corpus["doc_id"].nunique()),
        "n_passages": int(len(corpus)),
        "metrics": metrics,
        "timings_ms": {k: round(v, 3) for k, v in timings.items()},
        "sample_question": q,
        "sample_answer": answer,
        "sample_sources": [cid for _, cid, _ in results],
    }
    (OUTPUTS / "results.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    result = run_pipeline()
    print("Hybrid RAG pipeline complete.")
    print(f"Documents: {result['n_documents']}  Passages: {result['n_passages']}")
    print(
        "Recall@5 — sparse / dense / hybrid:",
        result["metrics"]["sparse"]["recall@5"],
        result["metrics"]["dense"]["recall@5"],
        result["metrics"]["hybrid"]["recall@5"],
    )
    print(
        "MRR — sparse / dense / hybrid:",
        result["metrics"]["sparse"]["mrr"],
        result["metrics"]["dense"]["mrr"],
        result["metrics"]["hybrid"]["mrr"],
    )
    print(f"\nQ: {result['sample_question']}")
    print(f"A: {result['sample_answer']}")
    print(f"Sources: {result['sample_sources']}")
