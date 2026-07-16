#!/usr/bin/env python3
"""
Topic Modeling on 20 Newsgroups — Unsupervised Text Mining

Compares three topic modeling algorithms (NMF, LDA, TruncatedSVD/LSA)
on a curated subset of the 20 Newsgroups dataset. Extracts latent themes
across computer, recreation, science, politics, and for-sale documents.

Author: Data Analysis Portfolio
Date: 2026-07-16
"""

import json
import warnings
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from wordcloud import WordCloud

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ─── Paths ───────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHARTS_DIR = BASE_DIR / "charts"
OUTPUTS_DIR = BASE_DIR / "outputs"

for d in [DATA_DIR, CHARTS_DIR, OUTPUTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Configuration ───────────────────────────────────────────────────────
RANDOM_STATE = 42
N_COMPONENTS = 6  # number of topics to extract
MAX_FEATURES = 5000
N_TOP_WORDS = 15
SAMPLE_PER_CATEGORY = 200  # limit per category for faster LDA training

# Select 8 diverse categories for interesting (but separable) topics
CATEGORIES = [
    "comp.graphics",
    "comp.sys.ibm.pc.hardware",
    "rec.sport.baseball",
    "rec.sport.hockey",
    "sci.space",
    "sci.med",
    "talk.politics.guns",
    "misc.forsale",
]

LABEL_MAP = {cat: i for i, cat in enumerate(CATEGORIES)}

print("=" * 60)
print("  Topic Modeling — 20 Newsgroups")
print("=" * 60)

# ─── 1. Load Data ────────────────────────────────────────────────────────
print("\n[1/7] Loading 20 Newsgroups dataset...")
newsgroups = fetch_20newsgroups(
    subset="all",
    categories=CATEGORIES,
    remove=("headers", "footers", "quotes"),
    data_home=str(DATA_DIR),
    shuffle=True,
    random_state=RANDOM_STATE,
)

docs = newsgroups.data
true_labels = [LABEL_MAP[newsgroups.target_names[t]] for t in newsgroups.target]
category_names = newsgroups.target_names

print(f"  Total documents: {len(docs):,}")
print(f"  Categories: {len(CATEGORIES)}")
for name in CATEGORIES:
    count = sum(1 for t in newsgroups.target if newsgroups.target_names[t] == name)
    print(f"    {name:35s} {count:4d} docs")

# Subsample for faster training (keep class balance)
print(f"\n  Subsampling to {SAMPLE_PER_CATEGORY} docs per category...")
rng = np.random.RandomState(RANDOM_STATE)
subset_indices = []
for cat_name in CATEGORIES:
    cat_idx = CATEGORIES.index(cat_name)
    cat_doc_indices = [i for i, t in enumerate(newsgroups.target) if newsgroups.target_names[t] == cat_name]
    chosen = rng.choice(cat_doc_indices, min(SAMPLE_PER_CATEGORY, len(cat_doc_indices)), replace=False)
    subset_indices.extend(chosen)

subset_indices = sorted(subset_indices)
docs = [docs[i] for i in subset_indices]
true_labels = [true_labels[i] for i in subset_indices]

print(f"  Subsampled: {len(docs):,} documents ({len(subset_indices):,} total)")
for name in CATEGORIES:
    count = sum(1 for i, t in enumerate(true_labels) if t == CATEGORIES.index(name))
    print(f"    {name:35s} {count:4d} docs")

# ─── 2. Text Preprocessing + Vectorization ───────────────────────────────
print("\n[2/7] Vectorizing with TF-IDF (unigrams + bigrams)...")

tfidf_vectorizer = TfidfVectorizer(
    max_features=MAX_FEATURES,
    max_df=0.95,
    min_df=2,
    stop_words="english",
    ngram_range=(1, 2),
    sublinear_tf=True,
)

tfidf_matrix = tfidf_vectorizer.fit_transform(docs)
feature_names = tfidf_vectorizer.get_feature_names_out()

print(f"  Vocabulary size: {len(feature_names):,}")
print(f"  TF-IDF matrix: {tfidf_matrix.shape}")

# Also create a count matrix for LDA
count_vectorizer = CountVectorizer(
    max_features=MAX_FEATURES,
    max_df=0.95,
    min_df=2,
    stop_words="english",
    ngram_range=(1, 1),
)
count_matrix = count_vectorizer.fit_transform(docs)

print(f"  Count matrix: {count_matrix.shape}")

# ─── 3. Train Topic Models ───────────────────────────────────────────────
print("\n[3/7] Training topic models...")

models = {}

# --- NMF ---
print("  NMF...", end=" ", flush=True)
nmf = NMF(n_components=N_COMPONENTS, random_state=RANDOM_STATE, max_iter=500, init="nndsvdar")
nmf_w = nmf.fit_transform(tfidf_matrix)
nmf_h = nmf.components_
models["NMF (Non-Negative Matrix Factorization)"] = (nmf, nmf_w, nmf_h, tfidf_vectorizer, "tfidf")
print("done")

# --- LDA (online for speed) ---
print("  LDA (online)...", end=" ", flush=True)
lda = LatentDirichletAllocation(
    n_components=N_COMPONENTS,
    random_state=RANDOM_STATE,
    max_iter=50,
    learning_method="online",
    learning_offset=50.0,
    batch_size=128,
    n_jobs=-1,
)
lda_w = lda.fit_transform(count_matrix)
lda_h = lda.components_
models["LDA (Latent Dirichlet Allocation)"] = (lda, lda_w, lda_h, count_vectorizer, "count")
print("done")

# --- TruncatedSVD (LSA) ---
print("  TruncatedSVD (LSA)...", end=" ", flush=True)
svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=RANDOM_STATE, n_iter=20)
svd_w = svd.fit_transform(tfidf_matrix)
# Normalize after transform
svd_w = svd_w / (np.linalg.norm(svd_w, axis=1, keepdims=True) + 1e-10)
svd_h = svd.components_


class SVDWrapper:
    """Wrapper for TruncatedSVD to maintain compatibility with model interface."""

    def __init__(self, svd_model):
        self.named_steps = {"truncatedsvd": svd_model}


models["TruncatedSVD (LSA)"] = (SVDWrapper(svd), svd_w, svd_h, tfidf_vectorizer, "tfidf")
print("done")

# ─── 4. Helper: Display Top Words ────────────────────────────────────────
print("\n[4/7] Extracting top words per topic...")


def get_top_words(components, vectorizer, n_top=N_TOP_WORDS, model_type="tfidf"):
    """Get top n words per topic."""
    if model_type == "tfidf":
        names = vectorizer.get_feature_names_out()
    else:
        names = vectorizer.get_feature_names_out()

    topic_words = []
    for topic_idx, topic in enumerate(components):
        top_indices = topic.argsort()[-n_top:][::-1]
        words = [names[i] for i in top_indices]
        weights = [topic[i] for i in top_indices]
        topic_words.append(list(zip(words, weights)))
    return topic_words


all_topic_words = {}
for model_name, (_, _, components, vectorizer, vec_type) in models.items():
    all_topic_words[model_name] = get_top_words(components, vectorizer, model_type=vec_type)

# Print topic-word summaries
for model_name, topic_words in all_topic_words.items():
    print(f"\n  ── {model_name} ──")
    for t_idx, words in enumerate(topic_words):
        word_str = ", ".join(f"{w} ({v:.3f})" for w, v in words[:8])
        print(f"    Topic {t_idx + 1}: {word_str}")

# ─── 5. Evaluate: Topic Coherence ────────────────────────────────────────
print("\n[5/7] Computing topic coherence (simple word-pair coherence)...")


def coherence_from_components(components, vectorizer, dtm, n_top=10):
    """Coherence using top words' average pairwise co-document frequency."""
    dtm_binary = (dtm > 0).astype(np.float64)

    topic_scores = []
    for topic_idx in range(components.shape[0]):
        top_indices = components[topic_idx].argsort()[-n_top:][::-1]
        pair_scores = []
        for i in range(len(top_indices)):
            col_i = dtm_binary[:, top_indices[i]].toarray().ravel()
            for j in range(i + 1, len(top_indices)):
                col_j = dtm_binary[:, top_indices[j]].toarray().ravel()
                both = np.logical_and(col_i, col_j).sum()
                wi = col_i.sum()
                if wi > 0:
                    pair_scores.append(both / wi)
        topic_scores.append(np.mean(pair_scores) if pair_scores else 0.0)
    return topic_scores


coherence_results = {}
for model_name, (_, _, components, vectorizer, vec_type) in models.items():
    dtm = count_matrix  # use count matrix for co-occurrence
    scores = coherence_from_components(components, vectorizer, dtm, n_top=10, model_type=vec_type)
    coherence_results[model_name] = {
        "per_topic": [round(s, 4) for s in scores],
        "mean": round(np.mean(scores), 4),
    }
    print(f"  {model_name}: mean coherence = {coherence_results[model_name]['mean']:.4f}")
    for t_idx, s in enumerate(scores):
        print(f"    Topic {t_idx + 1}: {s:.4f}")


# Save coherence results
with open(OUTPUTS_DIR / "coherence_scores.json", "w") as f:
    json.dump(coherence_results, f, indent=2)

# ─── 6. Visualizations ───────────────────────────────────────────────────
print("\n[6/7] Generating visualizations...")

sns.set_style("whitegrid")
plt.rcParams.update({"figure.max_open_warning": 0, "font.size": 10})

# Find best model for detailed visualization (NMF generally produces cleanest topics)
best_model_name = "NMF (Non-Negative Matrix Factorization)"
best_components = models[best_model_name][2]
best_vectorizer = models[best_model_name][3]
best_vec_type = models[best_model_name][4]
best_names = best_vectorizer.get_feature_names_out()


def plot_wordclouds(components, vectorizer, names, model_label, n_top=50):
    """Generate word cloud per topic."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for topic_idx in range(min(N_COMPONENTS, len(axes))):
        top_indices = components[topic_idx].argsort()[-n_top:][::-1]
        word_freq = {names[i]: float(components[topic_idx][i]) for i in top_indices}

        wc = WordCloud(
            width=800,
            height=400,
            background_color="white",
            max_words=n_top,
            random_state=RANDOM_STATE,
            colormap="viridis",
        ).generate_from_frequencies(word_freq)

        axes[topic_idx].imshow(wc, interpolation="bilinear")
        axes[topic_idx].axis("off")
        axes[topic_idx].set_title(f"Topic {topic_idx + 1}", fontsize=14, fontweight="bold")

    for idx in range(N_COMPONENTS, len(axes)):
        fig.delaxes(axes[idx])

    fig.suptitle(f"Topic Word Clouds — {model_label}", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(CHARTS_DIR / "01-topic-wordclouds.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ Saved: 01-topic-wordclouds.png")


def plot_topic_term_heatmap(components, vectorizer, names, model_label, n_top=12):
    """Heatmap of top terms per topic."""
    n_topics = components.shape[0]
    top_terms = []
    for t_idx in range(n_topics):
        top_indices = components[t_idx].argsort()[-n_top:][::-1]
        top_terms.extend(names[top_indices])

    top_terms = list(dict.fromkeys(top_terms))  # unique, preserve order
    term_to_idx = {term: idx for idx, term in enumerate(top_terms)}

    heat_data = np.zeros((n_topics, len(top_terms)))
    for t_idx in range(n_topics):
        for term in top_terms:
            idx = np.where(names == term)[0]
            if len(idx) > 0:
                heat_data[t_idx, term_to_idx[term]] = components[t_idx, idx[0]]

    fig, ax = plt.subplots(figsize=(max(12, len(top_terms) * 0.5), 6))
    sns.heatmap(
        heat_data,
        xticklabels=top_terms,
        yticklabels=[f"Topic {t + 1}" for t in range(n_topics)],
        cmap="YlOrRd",
        ax=ax,
        cbar_kws={"label": "Term Weight"},
    )
    ax.set_title(f"Topic-Term Heatmap — {model_label}", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(CHARTS_DIR / "02-topic-term-heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ Saved: 02-topic-term-heatmap.png")


def plot_model_comparison(all_topic_words_dict, coherence_dict):
    """Compare models: side-by-side topic words and coherence."""
    model_names = list(all_topic_words_dict.keys())
    n_models = len(model_names)

    fig, axes = plt.subplots(n_models, 2, figsize=(20, n_models * 4))
    if n_models == 1:
        axes = axes.reshape(1, -1)

    for row, model_name in enumerate(model_names):
        topic_words = all_topic_words_dict[model_name]

        # Column 1: Top words per topic
        ax = axes[row, 0]
        ax.axis("off")
        y_pos = 0.95
        ax.text(0.5, 1.02, model_name, transform=ax.transAxes, fontsize=13, fontweight="bold", ha="center", va="bottom")
        for t_idx, words in enumerate(topic_words):
            word_str = ", ".join(w for w, _ in words[:10])
            ax.text(0.02, y_pos, f"T{t_idx + 1}:", fontweight="bold", fontsize=9, transform=ax.transAxes, va="top")
            ax.text(0.15, y_pos, word_str, fontsize=8, transform=ax.transAxes, va="top")
            y_pos -= 0.18

        # Column 2: Coherence bar chart
        ax = axes[row, 1]
        scores = coherence_dict[model_name]["per_topic"]
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(scores)))
        bars = ax.bar(
            [f"T{t + 1}" for t in range(len(scores))],
            scores,
            color=colors,
            edgecolor="white",
        )
        ax.axhline(
            y=coherence_dict[model_name]["mean"],
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Mean: {coherence_dict[model_name]['mean']:.4f}",
        )
        ax.set_ylabel("Coherence Score")
        ax.set_title(f"Topic Coherence — {model_name}", fontsize=12)
        ax.legend(fontsize=8)
        for bar, score in zip(bars, scores):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{score:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.suptitle("Model Comparison: Topics & Coherence", fontsize=16, fontweight="bold")
    plt.tight_layout()
    fig.savefig(CHARTS_DIR / "03-model-comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ Saved: 03-model-comparison.png")


def plot_document_topic_distribution(topic_dist, model_label, true_labels, n_sample=500):
    """Show which topics dominate which ground-truth categories."""
    rng = np.random.RandomState(RANDOM_STATE)
    if topic_dist.shape[0] > n_sample:
        idx = rng.choice(topic_dist.shape[0], n_sample, replace=False)
        topic_dist = topic_dist[idx]
        true_labels_subset = [true_labels[i] for i in idx]
    else:
        true_labels_subset = true_labels

    dominant_topics = np.argmax(topic_dist, axis=1)

    # Cross-tab: ground truth category × dominant topic
    categories = sorted(set(true_labels_subset))
    cat_names = [CATEGORIES[c] for c in categories]

    confusion = np.zeros((len(categories), topic_dist.shape[1]))
    for i, cat in enumerate(true_labels_subset):
        cat_idx = categories.index(cat) if cat in categories else -1
        if cat_idx >= 0:
            confusion[cat_idx, dominant_topics[i]] += 1

    # Normalize rows
    row_sums = confusion.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    confusion_normalized = confusion / row_sums

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        confusion_normalized,
        xticklabels=[f"Topic {t + 1}" for t in range(topic_dist.shape[1])],
        yticklabels=[c.split(".")[-1].replace("_", " ") for c in cat_names],
        cmap="Blues",
        annot=True,
        fmt=".2f",
        ax=ax,
        cbar_kws={"label": "Proportion"},
    )
    ax.set_title(f"Topic Distribution by Category — {model_label}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Assigned Topic")
    ax.set_ylabel("Ground Truth Category")
    plt.tight_layout()
    fig.savefig(CHARTS_DIR / "04-topic-category-heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ Saved: 04-topic-category-heatmap.png")


def plot_topic_strength_distribution(topic_dist, model_label):
    """Histogram of max topic probabilities — shows topic confidence."""
    max_probs = topic_dist.max(axis=1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(max_probs, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(
        x=np.median(max_probs), color="red", linestyle="--", linewidth=1.5, label=f"Median: {np.median(max_probs):.3f}"
    )
    ax.set_xlabel("Maximum Topic Probability")
    ax.set_ylabel("Number of Documents")
    ax.set_title(f"Topic Assignment Confidence — {model_label}", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(CHARTS_DIR / "05-topic-confidence-histogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ Saved: 05-topic-confidence-histogram.png")


def plot_3d_topic_space(svd_model, true_labels, model_label):
    """3D projection of documents in topic space (using SVD)."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    try:
        coords = svd_model[:, :3]  # first 3 dimensions
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        categories_unique = sorted(set(true_labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(categories_unique)))
        markers = ["o", "s", "^", "D", "v", "<", ">", "p"]

        for i, cat in enumerate(categories_unique):
            mask = [t == cat for t in true_labels]
            label_short = CATEGORIES[cat].split(".")[-1].replace("_", " ")
            # Sample for 3D (too many points = noise)
            idx = np.where(mask)[0]
            if len(idx) > 200:
                idx = np.random.RandomState(RANDOM_STATE).choice(idx, 200, replace=False)
            ax.scatter(
                coords[idx, 0],
                coords[idx, 1],
                coords[idx, 2],
                c=[colors[i]],
                marker=markers[i % len(markers)],
                label=label_short,
                alpha=0.6,
                s=15,
            )

        ax.set_title(f"3D Topic Space — {model_label}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_zlabel("Component 3")
        ax.legend(fontsize=8, loc="upper left", markerscale=2)
        plt.tight_layout()
        fig.savefig(CHARTS_DIR / "06-3d-topic-space.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  ✓ Saved: 06-3d-topic-space.png")
    except Exception as e:
        print(f"  ✗ 3D plot skipped: {e}")


# Generate all plots
plot_wordclouds(best_components, best_vectorizer, best_names, best_model_name)
plot_topic_term_heatmap(best_components, best_vectorizer, best_names, best_model_name)
plot_model_comparison(all_topic_words, coherence_results)
plot_document_topic_distribution(nmf_w, "NMF", true_labels)
plot_topic_strength_distribution(nmf_w, "NMF")
plot_3d_topic_space(svd_w, true_labels, "TruncatedSVD (LSA)")

print(f"\n  All charts saved to: {CHARTS_DIR}")

# ─── 7. Summary ──────────────────────────────────────────────────────────
print("\n[7/7] Generating summary...")

best_model = max(coherence_results, key=lambda k: coherence_results[k]["mean"])
print(f"\n  Best model by coherence: {best_model}")
print(f"  Mean coherence: {coherence_results[best_model]['mean']:.4f}")

summary = {
    "dataset": "20 Newsgroups (8 categories)",
    "total_documents": len(docs),
    "categories": CATEGORIES,
    "topic_modeling_algorithms": list(models.keys()),
    "best_model": best_model,
    "coherence_results": coherence_results,
    "topics": {},
}

for t_idx in range(N_COMPONENTS):
    words = all_topic_words[best_model_name][t_idx]
    summary["topics"][f"Topic {t_idx + 1}"] = {
        "top_words": [w for w, _ in words[:15]],
        "dominant_category": max(
            range(len(CATEGORIES)),
            key=lambda c: (
                np.bincount(
                    [true_labels[i] for i in range(len(true_labels)) if np.argmax(nmf_w[i]) == t_idx],
                    minlength=len(CATEGORIES),
                )[c]
                if any(np.argmax(nmf_w[i]) == t_idx for i in range(len(true_labels)))
                else 0
            ),
        ),
    }

with open(OUTPUTS_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

# ─── Final Report ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  TOPIC MODELING — COMPLETE")
print("=" * 60)
print(
    f"\n  Dataset: 20 Newsgroups ({len(docs):,} docs, {len(CATEGORIES)} categories, {SAMPLE_PER_CATEGORY} per category)"
)
print(f"  Topics extracted: {N_COMPONENTS}")
print(f"  Algorithms: {len(models)}")
print(f"  Charts saved: {len(list(CHARTS_DIR.glob('*.png')))}")
print(f"  Outputs: {OUTPUTS_DIR}")
print("\n  Model Coherence Comparison:")
for model_name, scores in coherence_results.items():
    print(f"    {model_name:40s} {scores['mean']:.4f}")
print(f"\n  Best model: {best_model} ({coherence_results[best_model]['mean']:.4f})")
print("\n  Charts:")
for chart in sorted(CHARTS_DIR.glob("*.png")):
    print(f"    {chart.name}")
print("\n  Done.")
