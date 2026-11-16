"""Unsupervised LDA topic modeling and node clustering for discovered concept graphs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import nltk
from gensim import corpora, models
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import simple_preprocess
from nltk.stem import WordNetLemmatizer

# Download WordNet data
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

lemmatizer = WordNetLemmatizer()

_DEFAULT_PASSES = 10


@dataclass(frozen=True)
class TopicDescriptor:
    """Human-readable summary of a single LDA topic."""

    topic_id: int
    label: str
    keywords: list[tuple[str, float]]


@dataclass(frozen=True)
class NodeTopicAssignment:
    """Dominant LDA topic assignment for a single graph node."""

    node_id: str
    topic_id: int
    confidence: float
    distribution: list[tuple[int, float]]


@dataclass(frozen=True)
class TopicClusteringResult:
    """Full output of LDA clustering across a neighborhood's active documents."""

    num_topics: int
    topics: dict[int, TopicDescriptor]
    assignments: dict[str, NodeTopicAssignment]
    tokenized_docs: dict[str, list[str]]
    dictionary: corpora.Dictionary
    corpus: list[list[tuple[int, int]]]
    model: models.LdaModel


def preprocess_text(
    text: str,
    extra_stopwords: Iterable[str] | None = None,
) -> list[str]:
    """Tokenize, normalize, remove stopwords, and lemmatize (verb + noun) input text."""
    stopwords = STOPWORDS
    if extra_stopwords:
        normalized_extra = {word.strip().lower() for word in extra_stopwords if word.strip()}
        if normalized_extra:
            stopwords = STOPWORDS.union(normalized_extra)

    tokens = simple_preprocess(text, deacc=True, min_len=3)
    return [
        lemmatizer.lemmatize(lemmatizer.lemmatize(word, pos="v"), pos="n")
        for word in tokens
        if word not in stopwords
    ]


def build_corpus_and_dictionary(
    tokenized_docs: list[list[str]],
    no_below: int = 5,
    no_above: float = 0.5,
) -> tuple[corpora.Dictionary, list[list[tuple[int, int]]]]:
    """Build a Gensim dictionary (with extreme-frequency filtering) and BoW corpus."""
    dictionary = corpora.Dictionary(tokenized_docs)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=None)
    corpus = [dictionary.doc2bow(tokens) for tokens in tokenized_docs]
    return dictionary, corpus


def train_lda(
    corpus: list[list[tuple[int, int]]],
    dictionary: corpora.Dictionary,
    num_topics: int,
    passes: int = 10,
    random_state: int = 42,
    alpha: str | float = "symmetric",
    eta: str | float | None = None,
) -> models.LdaModel:
    """Train a Gensim LDA model on a BoW corpus and dictionary."""
    return models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=passes,
        random_state=random_state,
        alpha=alpha,
        eta=eta,
    )


def print_model_topics(model: models.LdaModel, num_words: int = 8) -> list[tuple[int, str]]:
    """Return formatted topic strings from a trained Gensim LDA model."""
    return list(model.print_topics(num_topics=model.num_topics, num_words=num_words))


def cluster_topic_nodes(
    documents: Mapping[str, str],
    *,
    num_topics: int = 4,
    no_below: int = 2,
    no_above: float = 0.85,
    alpha: str | float = "symmetric",
    eta: str | float | None = None,
    extra_stopwords: Iterable[str] | None = None,
    num_keywords: int = 5,
    random_state: int = 42,
) -> TopicClusteringResult:
    """Cluster active node documents using LDA and assign each node a dominant topic."""
    if not documents:
        raise ValueError("documents must not be empty")
    if num_topics < 1:
        raise ValueError("num_topics must be >= 1")
    if num_keywords < 1:
        raise ValueError("num_keywords must be >= 1")

    tokenized_docs = {
        node_id: preprocess_text(text, extra_stopwords=extra_stopwords)
        for node_id, text in documents.items()
    }
    node_ids = list(tokenized_docs.keys())
    tokenized_list = [tokenized_docs[node_id] for node_id in node_ids]

    dictionary, corpus = build_corpus_and_dictionary(
        tokenized_list,
        no_below=no_below,
        no_above=no_above,
    )
    if len(dictionary) == 0:
        dictionary, corpus = build_corpus_and_dictionary(
            tokenized_list,
            no_below=1,
            no_above=1.0,
        )
    if len(dictionary) == 0:
        raise ValueError("Corpus contains no valid vocabulary after preprocessing.")

    effective_topics = max(1, min(num_topics, len(documents)))
    model = train_lda(
        corpus=corpus,
        dictionary=dictionary,
        num_topics=effective_topics,
        passes=_DEFAULT_PASSES,
        random_state=random_state,
        alpha=alpha,
        eta=eta,
    )

    topics: dict[int, TopicDescriptor] = {}
    for topic_id in range(effective_topics):
        raw_keywords = model.show_topic(topic_id, num_keywords)
        keywords = [(word, float(weight)) for word, weight in raw_keywords]
        label = ", ".join(word for word, _ in keywords[:3]) or f"Topic {topic_id}"
        topics[topic_id] = TopicDescriptor(
            topic_id=topic_id,
            label=label,
            keywords=keywords,
        )

    assignments: dict[str, NodeTopicAssignment] = {}
    uniform_dist = [(tid, 1.0 / effective_topics) for tid in range(effective_topics)]
    for node_id, bow in zip(node_ids, corpus, strict=True):
        if not bow:
            assignments[node_id] = NodeTopicAssignment(
                node_id=node_id,
                topic_id=0,
                confidence=0.0,
                distribution=list(uniform_dist),
            )
            continue

        raw_dist = model.get_document_topics(bow, minimum_probability=0.0)
        distribution = [(int(tid), float(prob)) for tid, prob in raw_dist]
        dominant_tid, confidence = max(distribution, key=lambda item: item[1])
        assignments[node_id] = NodeTopicAssignment(
            node_id=node_id,
            topic_id=dominant_tid,
            confidence=confidence,
            distribution=distribution,
        )

    return TopicClusteringResult(
        num_topics=effective_topics,
        topics=topics,
        assignments=assignments,
        tokenized_docs=tokenized_docs,
        dictionary=dictionary,
        corpus=corpus,
        model=model,
    )
