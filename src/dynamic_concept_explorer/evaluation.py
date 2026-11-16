"""Offline evaluation metrics for graph topology, LDA topic coherence, and ROUGE."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import networkx as nx
from gensim.models import CoherenceModel
from rouge_score import rouge_scorer

from dynamic_concept_explorer.topic_modeling import TopicClusteringResult


@dataclass(frozen=True)
class GraphTopologyMetrics:
    """Structural metrics for a discovered concept neighborhood graph."""

    node_count: int
    edge_count: int
    density: float
    in_degree_centrality: dict[str, float]
    out_degree_centrality: dict[str, float]
    betweenness_centrality: dict[str, float]
    pagerank: dict[str, float]
    clustering_coefficients: dict[str, float]
    average_clustering: float
    reciprocity: float


@dataclass(frozen=True)
class TopicCoherenceResult:
    """Gensim C_v topic coherence scores for an LDA clustering result."""

    mean_coherence: float
    topic_coherences: dict[int, float]
    num_topics: int


@dataclass(frozen=True)
class RougeScore:
    """Precision, recall, and F1 for a single ROUGE variant."""

    precision: float
    recall: float
    fmeasure: float


@dataclass(frozen=True)
class DocumentRougeResult:
    """ROUGE-1, ROUGE-2, and ROUGE-L scores for one summary/reference pair."""

    rouge1: RougeScore
    rouge2: RougeScore
    rougeL: RougeScore


@dataclass(frozen=True)
class RougeEvaluationResult:
    """Per-document and macro-averaged ROUGE scores across a corpus."""

    per_document: dict[str, DocumentRougeResult]
    mean_rouge1: RougeScore
    mean_rouge2: RougeScore
    mean_rougeL: RougeScore


def _build_digraph(edges_or_graph: Mapping[str, Iterable[str]] | nx.DiGraph) -> nx.DiGraph:
    if isinstance(edges_or_graph, nx.DiGraph):
        return edges_or_graph

    graph = nx.DiGraph()
    for source, targets in edges_or_graph.items():
        graph.add_node(source)
        for target in targets:
            graph.add_edge(source, target)
    return graph


def compute_graph_topology(
    edges_or_graph: Mapping[str, Iterable[str]] | nx.DiGraph,
) -> GraphTopologyMetrics:
    """Compute centrality, PageRank, and undirected clustering metrics on a concept graph."""
    graph = _build_digraph(edges_or_graph)
    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()

    if node_count == 0:
        return GraphTopologyMetrics(
            node_count=0,
            edge_count=0,
            density=0.0,
            in_degree_centrality={},
            out_degree_centrality={},
            betweenness_centrality={},
            pagerank={},
            clustering_coefficients={},
            average_clustering=0.0,
            reciprocity=0.0,
        )

    if node_count == 1:
        only_node = next(iter(graph.nodes()))
        return GraphTopologyMetrics(
            node_count=1,
            edge_count=edge_count,
            density=0.0,
            in_degree_centrality={only_node: 0.0},
            out_degree_centrality={only_node: 0.0},
            betweenness_centrality={only_node: 0.0},
            pagerank={only_node: 1.0},
            clustering_coefficients={only_node: 0.0},
            average_clustering=0.0,
            reciprocity=0.0,
        )

    in_deg = {n: float(v) for n, v in nx.in_degree_centrality(graph).items()}
    out_deg = {n: float(v) for n, v in nx.out_degree_centrality(graph).items()}
    betweenness = {n: float(v) for n, v in nx.betweenness_centrality(graph).items()}
    pagerank = {n: float(v) for n, v in nx.pagerank(graph).items()}

    undirected = graph.to_undirected(as_view=True)
    clustering = {n: float(v) for n, v in nx.clustering(undirected).items()}
    avg_clustering = float(nx.average_clustering(undirected))
    reciprocity = float(nx.reciprocity(graph)) if edge_count > 0 else 0.0

    return GraphTopologyMetrics(
        node_count=node_count,
        edge_count=edge_count,
        density=float(nx.density(graph)),
        in_degree_centrality=in_deg,
        out_degree_centrality=out_deg,
        betweenness_centrality=betweenness,
        pagerank=pagerank,
        clustering_coefficients=clustering,
        average_clustering=avg_clustering,
        reciprocity=reciprocity,
    )


def _safe_float(value: float) -> float:
    val = float(value)
    return 0.0 if math.isnan(val) else val


def compute_topic_coherence(
    clustering: TopicClusteringResult,
    *,
    top_n: int = 10,
) -> TopicCoherenceResult:
    """Compute Gensim C_v topic coherence for a trained LDA clustering result."""
    if top_n < 1:
        raise ValueError("top_n must be >= 1")

    texts = [tokens for tokens in clustering.tokenized_docs.values() if tokens]
    if not texts or clustering.num_topics < 1:
        return TopicCoherenceResult(
            mean_coherence=0.0,
            topic_coherences={},
            num_topics=0,
        )

    vocab = clustering.dictionary.token2id
    topic_ids = sorted(range(clustering.num_topics))
    topics_words: list[list[str]] = []
    valid_topic_ids: list[int] = []

    for topic_id in topic_ids:
        raw_words = [word for word, _ in clustering.model.show_topic(topic_id, topn=top_n)]
        in_vocab = [word for word in raw_words if word in vocab]
        if in_vocab:
            topics_words.append(in_vocab)
            valid_topic_ids.append(topic_id)

    if not topics_words:
        return TopicCoherenceResult(
            mean_coherence=0.0,
            topic_coherences={tid: 0.0 for tid in topic_ids},
            num_topics=clustering.num_topics,
        )

    coherence_model = CoherenceModel(
        topics=topics_words,
        texts=texts,
        dictionary=clustering.dictionary,
        coherence="c_v",
        processes=1,
    )
    per_topic_raw = coherence_model.get_coherence_per_topic()
    topic_coherences = {tid: 0.0 for tid in topic_ids}
    for tid, score in zip(valid_topic_ids, per_topic_raw, strict=True):
        topic_coherences[tid] = _safe_float(score)

    mean_coherence = sum(topic_coherences.values()) / len(topic_ids)
    return TopicCoherenceResult(
        mean_coherence=mean_coherence,
        topic_coherences=topic_coherences,
        num_topics=clustering.num_topics,
    )


def _to_rouge_score(score_obj: object) -> RougeScore:
    return RougeScore(
        precision=float(score_obj.precision),
        recall=float(score_obj.recall),
        fmeasure=float(score_obj.fmeasure),
    )


def _mean_rouge_score(scores: list[RougeScore]) -> RougeScore:
    count = len(scores)
    return RougeScore(
        precision=sum(s.precision for s in scores) / count,
        recall=sum(s.recall for s in scores) / count,
        fmeasure=sum(s.fmeasure for s in scores) / count,
    )


def compute_rouge_score(
    hypothesis: str,
    reference: str,
    *,
    use_stemmer: bool = True,
) -> DocumentRougeResult:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L between a generated summary and reference."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=use_stemmer)
    scores = scorer.score(reference, hypothesis)
    return DocumentRougeResult(
        rouge1=_to_rouge_score(scores["rouge1"]),
        rouge2=_to_rouge_score(scores["rouge2"]),
        rougeL=_to_rouge_score(scores["rougeL"]),
    )


def evaluate_summaries(
    hypotheses: Mapping[str, str],
    references: Mapping[str, str],
    *,
    use_stemmer: bool = True,
) -> RougeEvaluationResult:
    """Compute per-document and macro-averaged ROUGE-1/2/L across matching document IDs."""
    if not hypotheses or not references:
        raise ValueError("hypotheses and references must not be empty")
    if set(hypotheses.keys()) != set(references.keys()):
        raise ValueError("hypotheses and references must have the exact same document keys")

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=use_stemmer)
    per_document: dict[str, DocumentRougeResult] = {}

    for doc_id in hypotheses:
        scores = scorer.score(references[doc_id], hypotheses[doc_id])
        per_document[doc_id] = DocumentRougeResult(
            rouge1=_to_rouge_score(scores["rouge1"]),
            rouge2=_to_rouge_score(scores["rouge2"]),
            rougeL=_to_rouge_score(scores["rougeL"]),
        )

    results = list(per_document.values())
    return RougeEvaluationResult(
        per_document=per_document,
        mean_rouge1=_mean_rouge_score([r.rouge1 for r in results]),
        mean_rouge2=_mean_rouge_score([r.rouge2 for r in results]),
        mean_rougeL=_mean_rouge_score([r.rougeL for r in results]),
    )
