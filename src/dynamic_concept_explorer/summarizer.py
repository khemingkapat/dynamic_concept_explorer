"""Summarization for topic nodes.

Two layers live here:

* Coursework baselines (``textrank``, ``lexrank_summary``, ``bart_summary``) extracted from
  ``summarize/CPE393_BBC_Summarization_Homework_Student.ipynb``. Their behaviour is unchanged.
* Node overviews (``summarize_node``): fast 2-sentence definitions for every discovered concept,
  using the article lead when it exists, full-body TextRank when it does not, and BART on demand.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

import networkx as nx
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
except Exception:
    pass
from lexrank import STOPWORDS, LexRank
from nltk.tokenize import sent_tokenize

BART_MODEL_NAME = "facebook/bart-large-cnn"
BART_MAX_INPUT_TOKENS = 1024

DEFAULT_NODE_SENTENCES = 2
# Sentences outside this word range are headings, list fragments, or run-ons: poor overviews.
MIN_SENTENCE_WORDS = 6
MAX_SENTENCE_WORDS = 60
# TextRank is O(N^2) in sentences; long articles are capped to keep node latency bounded.
MAX_TEXTRANK_SENTENCES = 250

Abstractive = Callable[[str], str]


# --------------------------------------------------------------------------------------
# Coursework baselines
# --------------------------------------------------------------------------------------


def target_sentence_count(sentences: list[str], ratio: float = 0.28) -> int:
    """Number of sentences to keep for a compression ratio (notebook Cell 7)."""
    return max(1, min(len(sentences), round(len(sentences) * ratio)))


def textrank(text: str, ratio: float = 0.28) -> list[str]:
    """Baseline TextRank (notebook Cell 7): TF-IDF cosine graph scored with ``nx.pagerank``.

    Returns the top ``ratio`` share of sentences in their original order.
    """
    sentences = sent_tokenize(text)
    if len(sentences) <= 1:
        return sentences

    tfidf = TfidfVectorizer(stop_words="english").fit_transform(sentences)
    similarity = cosine_similarity(tfidf)
    np.fill_diagonal(similarity, 0)

    graph = nx.from_numpy_array(similarity)
    scores = nx.pagerank(graph)

    k = target_sentence_count(sentences, ratio)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_indices = sorted([idx for idx, _ in ranked[:k]])

    return [sentences[i] for i in top_indices]


def lexrank_summary(
    text: str,
    corpus: list[list[str]] | None = None,
    ratio: float = 0.28,
    threshold: float | None = 0.25,
) -> list[str]:
    """Baseline LexRank (notebook Cell 10), returning sentences in original order."""
    sentences = sent_tokenize(text)
    if len(sentences) <= 1:
        return sentences

    k = target_sentence_count(sentences, ratio)
    if corpus is None:
        corpus = [sentences]

    lxr = LexRank(corpus, stopwords=STOPWORDS["en"])
    summary_sentences = lxr.get_summary(sentences, summary_size=k, threshold=threshold)

    summary_set = set(summary_sentences)
    ordered_summary = [s for s in sentences if s in summary_set]
    return ordered_summary if ordered_summary else summary_sentences


def bart_summary(
    text: str,
    model=None,
    tokenizer=None,
    device=None,
    max_new_tokens: int = 120,
    min_new_tokens: int = 35,
) -> tuple[str, bool]:
    """Baseline BART abstractive summary (notebook Cell 12).

    Returns ``(summary, was_truncated)`` where ``was_truncated`` flags inputs over 1024 tokens.
    """
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(BART_MODEL_NAME)
    if model is None:
        model = AutoModelForSeq2SeqLM.from_pretrained(BART_MODEL_NAME)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device).eval()
    elif device is None:
        device = next(model.parameters()).device

    encoded_full = tokenizer(text, truncation=False, add_special_tokens=True)
    was_truncated = len(encoded_full["input_ids"]) > BART_MAX_INPUT_TOKENS
    inputs = tokenizer(
        text, max_length=BART_MAX_INPUT_TOKENS, truncation=True, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            num_beams=4,
            do_sample=False,
        )

    summary_text = tokenizer.decode(ids[0], skip_special_tokens=True)
    return summary_text, was_truncated


# --------------------------------------------------------------------------------------
# Node overviews
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class NodeSummary:
    """Short overview of one topic node and how it was produced."""

    text: str
    sentences: list[str]
    method: str
    elapsed_ms: float


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences, paragraph by paragraph, dropping empty fragments."""
    sentences: list[str] = []
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if paragraph:
            sentences.extend(s.strip() for s in sent_tokenize(paragraph) if s.strip())
    return sentences


def _is_overview_candidate(sentence: str) -> bool:
    return MIN_SENTENCE_WORDS <= len(sentence.split()) <= MAX_SENTENCE_WORDS


def pagerank_scores(
    similarity: np.ndarray,
    *,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    """PageRank over a dense weighted similarity matrix by vectorised power iteration.

    Numerically equivalent to ``nx.pagerank`` on the same graph (uniform teleport, dangling
    rows redistributed uniformly) without the cost of building a NetworkX graph object.
    """
    n = similarity.shape[0]
    if n == 0:
        return np.zeros(0)

    row_sums = similarity.sum(axis=1)
    dangling = row_sums == 0
    transition = np.divide(
        similarity,
        row_sums[:, None],
        out=np.zeros_like(similarity, dtype=float),
        where=~dangling[:, None],
    )

    scores = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        previous = scores
        dangling_mass = previous[dangling].sum() / n
        scores = damping * (previous @ transition + dangling_mass) + (1.0 - damping) / n
        if np.abs(scores - previous).sum() < n * tol:
            break
    return scores


def rank_sentences(sentences: list[str]) -> np.ndarray:
    """TextRank centrality score for each sentence (TF-IDF cosine graph + PageRank)."""
    n = len(sentences)
    if n == 0:
        return np.zeros(0)
    try:
        # Rows are L2-normalised, so the sparse product X @ X.T is the cosine similarity.
        tfidf = TfidfVectorizer(stop_words="english").fit_transform(sentences)
    except ValueError:  # every sentence was empty or stopwords only
        return np.full(n, 1.0 / n)

    similarity = (tfidf @ tfidf.T).toarray()
    np.fill_diagonal(similarity, 0.0)
    return pagerank_scores(similarity)


def lead_sentences(lead: str, n_sentences: int = DEFAULT_NODE_SENTENCES) -> list[str]:
    """First ``n_sentences`` usable sentences of the article lead (Lead-k baseline)."""
    candidates = [s for s in split_sentences(lead) if _is_overview_candidate(s)]
    return candidates[: max(0, n_sentences)]


def textrank_sentences(
    text: str,
    n_sentences: int = DEFAULT_NODE_SENTENCES,
    *,
    max_sentences: int = MAX_TEXTRANK_SENTENCES,
) -> list[str]:
    """Top ``n_sentences`` TextRank sentences of ``text``, restored to document order."""
    if n_sentences <= 0:
        return []
    candidates = [s for s in split_sentences(text) if _is_overview_candidate(s)]
    candidates = candidates[:max_sentences]
    if len(candidates) <= n_sentences:
        return candidates

    scores = rank_sentences(candidates)
    # Stable sort keeps the earlier sentence when scores tie.
    top = sorted(np.argsort(-scores, kind="stable")[:n_sentences].tolist())
    return [candidates[i] for i in top]


@lru_cache(maxsize=1)
def _load_bart():
    """Load BART once per process; later calls reuse the model."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BART_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(BART_MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, tokenizer, device


def bart_node_summary(text: str, *, max_new_tokens: int = 80, min_new_tokens: int = 20) -> str:
    """Abstractive node overview with a process-wide cached BART model."""
    model, tokenizer, device = _load_bart()
    summary, _ = bart_summary(
        text,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
    )
    return summary


def summarize_node(
    lead: str,
    body: str = "",
    *,
    method: str = "auto",
    n_sentences: int = DEFAULT_NODE_SENTENCES,
    abstractive: Abstractive | None = None,
) -> NodeSummary:
    """Produce a short overview for one topic node.

    ``method``:

    * ``"auto"`` (default): lead sentences; full-body TextRank when the lead is too thin.
    * ``"lead"``: first ``n_sentences`` of the lead only.
    * ``"textrank"``: TextRank over lead + body.
    * ``"bart"``: abstractive summary of the lead (body if there is no lead). ``abstractive``
      overrides the default BART model. If the model cannot be loaded or fails, the result
      falls back to the extractive ``"auto"`` path and ``method`` records that.

    ``NodeSummary.method`` reports what actually produced the text.
    """
    if method not in {"auto", "lead", "textrank", "bart"}:
        raise ValueError(f"Unknown summarization method: {method!r}")

    start = time.perf_counter()
    full_text = "\n\n".join(part for part in (lead, body) if part.strip())
    used = method

    if method == "bart":
        source = lead if lead.strip() else body
        sentences: list[str] = []
        if source.strip():
            try:
                text = (abstractive or bart_node_summary)(source).strip()
                sentences = split_sentences(text)
            except Exception:  # missing weights, no network, OOM: degrade, do not crash the UI
                sentences = []
        if not sentences:
            fallback = summarize_node(lead, body, method="auto", n_sentences=n_sentences)
            sentences, used = fallback.sentences, f"{fallback.method} (bart unavailable)"
    elif method == "lead":
        sentences = lead_sentences(lead, n_sentences)
    elif method == "textrank":
        sentences = textrank_sentences(full_text, n_sentences)
    else:
        sentences = lead_sentences(lead, n_sentences)
        used = "lead"
        if len(sentences) < n_sentences and body.strip():
            sentences = textrank_sentences(full_text, n_sentences)
            used = "textrank"

    return NodeSummary(
        text=" ".join(sentences),
        sentences=sentences,
        method=used,
        elapsed_ms=(time.perf_counter() - start) * 1000.0,
    )
