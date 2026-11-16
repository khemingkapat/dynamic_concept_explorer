"""Summarization for topic nodes.

Two layers live here:

* Coursework baselines (``textrank``, ``lexrank_summary``, ``bart_summary``) extracted from
  ``summarize/CPE393_BBC_Summarization_Homework_Student.ipynb``. Their behaviour is unchanged.
* Node overviews (``summarize_node``): 2-sentence overviews for every discovered concept using
  the same three methods (TextRank by default, LexRank, BART on demand), applied to the lead
  section or to the full article.
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

SUMMARY_METHODS = ("textrank", "lexrank", "bart")
SUMMARY_SCOPES = ("lead", "full")

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
    """Short overview of one topic node and how it was produced.

    ``method`` and ``scope`` report what actually produced the text, which can differ from
    what was requested when a fallback was taken.
    """

    text: str
    sentences: list[str]
    method: str
    scope: str
    elapsed_ms: float


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences, paragraph by paragraph, dropping empty fragments."""
    sentences: list[str] = []
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if paragraph:
            sentences.extend(s.strip() for s in sent_tokenize(paragraph) if s.strip())
    return sentences


def _candidate_sentences(text: str, max_sentences: int) -> list[str]:
    candidates = [
        s
        for s in split_sentences(text)
        if MIN_SENTENCE_WORDS <= len(s.split()) <= MAX_SENTENCE_WORDS
    ]
    return candidates[:max_sentences]


def rank_sentences(sentences: list[str]) -> np.ndarray:
    """TextRank score per sentence: TF-IDF cosine graph scored with ``nx.pagerank``.

    Same method as the baseline ``textrank``. The only change is graph construction: TF-IDF
    rows are L2-normalised, so the sparse product ``X @ X.T`` is the cosine similarity matrix
    and the dense pairwise computation is skipped.
    """
    n = len(sentences)
    if n == 0:
        return np.zeros(0)
    try:
        tfidf = TfidfVectorizer(stop_words="english").fit_transform(sentences)
    except ValueError:  # every sentence was empty or stopwords only
        return np.full(n, 1.0 / n)

    similarity = (tfidf @ tfidf.T).toarray()
    np.fill_diagonal(similarity, 0.0)
    scores = nx.pagerank(nx.from_numpy_array(similarity))
    return np.array([scores[i] for i in range(n)])


def textrank_sentences(
    text: str,
    n_sentences: int = DEFAULT_NODE_SENTENCES,
    *,
    max_sentences: int = MAX_TEXTRANK_SENTENCES,
) -> list[str]:
    """Top ``n_sentences`` TextRank sentences of ``text``, restored to document order."""
    if n_sentences <= 0:
        return []
    candidates = _candidate_sentences(text, max_sentences)
    if len(candidates) <= n_sentences:
        return candidates

    scores = rank_sentences(candidates)
    # Stable sort keeps the earlier sentence when scores tie.
    top = sorted(np.argsort(-scores, kind="stable")[:n_sentences].tolist())
    return [candidates[i] for i in top]


def lexrank_sentences(
    text: str,
    n_sentences: int = DEFAULT_NODE_SENTENCES,
    *,
    background: list[str] | None = None,
    threshold: float | None = 0.25,
    max_sentences: int = MAX_TEXTRANK_SENTENCES,
) -> list[str]:
    """Top ``n_sentences`` LexRank sentences of ``text``, restored to document order.

    ``background`` is a list of other documents (for example the other nodes' articles) used
    to estimate IDF. Without it, IDF is estimated from ``text`` alone.
    """
    if n_sentences <= 0:
        return []
    candidates = _candidate_sentences(text, max_sentences)
    if len(candidates) <= n_sentences:
        return candidates

    corpus = [split_sentences(doc) for doc in background or [] if doc.strip()]
    corpus.append(candidates)
    lxr = LexRank(corpus, stopwords=STOPWORDS["en"])
    picked = set(lxr.get_summary(candidates, summary_size=n_sentences, threshold=threshold))
    return [s for s in candidates if s in picked][:n_sentences]


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
    method: str = "textrank",
    scope: str = "lead",
    n_sentences: int = DEFAULT_NODE_SENTENCES,
    background: list[str] | None = None,
    abstractive: Abstractive | None = None,
) -> NodeSummary:
    """Produce a short overview for one topic node with one of the README methods.

    ``method``:

    * ``"textrank"`` (default): extractive TextRank.
    * ``"lexrank"``: extractive LexRank; ``background`` documents improve its IDF estimate.
    * ``"bart"``: abstractive BART, loaded on demand. ``abstractive`` overrides the default
      model. If the model cannot be loaded or fails, the result falls back to TextRank and
      ``NodeSummary.method`` records that.

    ``scope`` selects the text the method ranks:

    * ``"lead"`` (default): the lead section only. If it holds fewer than ``n_sentences``
      usable sentences, the full article is used instead.
    * ``"full"``: lead and body together.
    """
    if method not in SUMMARY_METHODS:
        raise ValueError(f"Unknown summarization method: {method!r}")
    if scope not in SUMMARY_SCOPES:
        raise ValueError(f"Unknown summarization scope: {scope!r}")

    start = time.perf_counter()
    full_text = "\n\n".join(part for part in (lead, body) if part.strip())
    lead_is_enough = len(_candidate_sentences(lead, n_sentences)) >= n_sentences
    used_scope = "lead" if scope == "lead" and lead_is_enough else "full"
    source = lead if used_scope == "lead" else full_text
    used_method = method

    sentences: list[str] = []
    if method == "bart" and source.strip():
        # BART reads at most 1024 tokens, so the lead goes first when the scope is the article.
        try:
            sentences = split_sentences((abstractive or bart_node_summary)(source).strip())
        except Exception:  # missing weights, no network, OOM: degrade, do not crash the UI
            sentences = []
        if not sentences:
            used_method = "textrank (bart unavailable)"
    elif method == "lexrank":
        sentences = lexrank_sentences(source, n_sentences, background=background)

    if method == "textrank" or used_method.startswith("textrank ("):
        sentences = textrank_sentences(source, n_sentences)

    return NodeSummary(
        text=" ".join(sentences),
        sentences=sentences,
        method=used_method,
        scope=used_scope,
        elapsed_ms=(time.perf_counter() - start) * 1000.0,
    )
