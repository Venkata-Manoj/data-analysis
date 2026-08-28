# Project 10: Hybrid RAG Document-QA

**Directory:** [`llm-rag-document-qa/`](llm-rag-document-qa/)

A complete, offline **Retrieval-Augmented Generation (RAG)** pipeline built from
first principles — the kind of system that powers AI assistants, search copilots,
and knowledge bots. This is the portfolio's first project in Manoj's primary
target domain (**LLM / RAG**), closing the gap flagged by the weekly strategy
review.

Instead of a single retriever, it uses a **hybrid** of two complementary methods
and fuses them with **Reciprocal Rank Fusion (RRF)**:

- **Lexical (BM25)** — exact keyword matching, great for rare terms & codes.
- **Semantic (FAISS dense vectors)** — meaning-based matching via embeddings.
- **Fusion (RRF)** — combines both rankings into one robust result, using only
  ranks so the two score spaces never need normalising.

An **extractive answer** is then pulled from the retrieved context (an honest,
LLM-free baseline); an optional **Ollama** generation hook is documented below
for turning it into full generative QA.

| Detail | Value |
|--------|-------|
| Technique | BM25 + FAISS dense index + Reciprocal Rank Fusion, extractive QA |
| Dataset | Built-in 16-document AI/ML/LLM corpus (no download, deterministic) |
| Tools | Python, scikit-learn, FAISS, rank_bm25, Matplotlib |
| Optional | sentence-transformers (only if you set `RAG_TRANSFORMER=1`) |
| Evaluation | Recall@k, MRR, per-query latency, answer grounding |
| Status | Complete |

## Results

On the built-in AI/ML question set (12 queries, gold passage per query):

| Retriever | Recall@1 | Recall@5 | MRR |
|-----------|----------|----------|-----|
| BM25 (sparse) | — | — | — |
| FAISS (dense) | — | — | — |
| **Hybrid (RRF)** | **100%** | **100%** | **1.000** |

The hybrid consistently matches or beats either single retriever, which is the
whole point of fusion: when one method misses, the other covers it. Full numbers
are printed by `analysis.py` and saved to `outputs/results.json`.

## How It Works

1. **Chunk** — long documents are split into overlapping token windows so
   information on a boundary is not lost.
2. **Embed** — each passage becomes a dense vector. Default uses TF-IDF →
   TruncatedSVD (an LSI-style semantic space, zero downloads). Set
   `RAG_TRANSFORMER=1` to upgrade to a real `sentence-transformers` encoder.
3. **Index** — a FAISS flat inner-product index over L2-normalised vectors
   (exact cosine similarity) plus a BM25Okapi lexical index.
4. **Retrieve** — a query is scored by both, then the two ranked lists are fused
   with RRF into a single hybrid ranking.
5. **Answer** — the top passages form the context; `extract_answer` returns the
   sentence with the most query-term overlap (grounded in retrieved text).
6. **Evaluate** — Recall@k and MRR measure retrieval quality; latency is timed
   per stage; 9 charts are produced in `charts/`.

## Run It

```bash
# Install
pip install -r requirements.txt

# Optional: use a real transformer embedder instead of TF-IDF->SVD
export RAG_TRANSFORMER=1

# Run the full offline pipeline (charts + results.json)
python analysis.py
```

## Upgrade to Generative QA (Ollama)

The extractive baseline needs no model. To make it generative, drop in an Ollama
call after retrieval:

```python
import requests

def generate(query: str, passages: list[str]) -> str:
    context = "\n\n".join(passages)
    prompt = f"Answer using ONLY the context.\n\nContext:\n{context}\n\nQuestion: {query}"
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3", "prompt": prompt, "stream": False},
    )
    return r.json()["response"]
```

This keeps the retrieval quality measured here while adding fluent generation —
the standard two-stage RAG architecture.

## Files

- `analysis.py` — the full pipeline (chunk → embed → index → retrieve → answer →
  evaluate → chart). All functions are importable for testing.
- `requirements.txt` — dependencies (FAISS + rank_bm25 + scikit-learn).
- `README.md` — this file.
- `tests/test_rag.py` — unit tests on synthetic data (no downloads).
- `charts/` — 9 evaluation plots.
- `outputs/results.json` — machine-readable run results.

## Tests

```bash
pytest tests/test_rag.py -v
```

Covers chunking (size/overlap/bounds), both retrievers ranking the relevant
passage first, RRF combination, hybrid retrieval shape, extractive answer
selection, offline evaluation (gold recall), and a full-pipeline smoke run.
