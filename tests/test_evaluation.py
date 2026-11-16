"""Offline unit tests for graph topology, LDA C_v coherence, and ROUGE evaluation."""

from __future__ import annotations

import math

import pytest

from dynamic_concept_explorer import evaluation, topic_modeling


def test_compute_graph_topology_triangle_one_way_links() -> None:
    edges = {
        "/wiki/A": {"/wiki/B", "/wiki/C"},
        "/wiki/B": {"/wiki/C"},
        "/wiki/C": set(),
    }
    metrics = evaluation.compute_graph_topology(edges)

    assert metrics.node_count == 3
    assert metrics.edge_count == 3
    assert metrics.density == pytest.approx(0.5)
    assert metrics.reciprocity == pytest.approx(0.0)
    # Undirected view turns one-way A->B, A->C, B->C into a complete triangle.
    assert metrics.average_clustering == pytest.approx(1.0)
    assert metrics.in_degree_centrality["/wiki/C"] == pytest.approx(1.0)
    assert metrics.out_degree_centrality["/wiki/A"] == pytest.approx(1.0)
    assert sum(metrics.pagerank.values()) == pytest.approx(1.0, rel=1e-4)
    assert metrics.pagerank["/wiki/C"] > metrics.pagerank["/wiki/A"]


def test_compute_graph_topology_identifies_bridge_betweenness() -> None:
    edges = {
        "A1": {"A2", "Bridge"},
        "A2": {"A1"},
        "Bridge": {"A1", "B1"},
        "B1": {"B2", "Bridge"},
        "B2": {"B1"},
    }
    metrics = evaluation.compute_graph_topology(edges)
    assert metrics.betweenness_centrality["Bridge"] > metrics.betweenness_centrality["A2"]
    assert metrics.betweenness_centrality["Bridge"] > metrics.betweenness_centrality["B2"]


def test_compute_graph_topology_handles_empty_single_and_edgeless_graphs() -> None:
    empty = evaluation.compute_graph_topology({})
    assert empty.node_count == 0
    assert empty.edge_count == 0
    assert empty.average_clustering == 0.0
    assert empty.pagerank == {}

    single = evaluation.compute_graph_topology({"/wiki/Seed": set()})
    assert single.node_count == 1
    assert single.edge_count == 0
    assert single.pagerank == {"/wiki/Seed": 1.0}
    assert single.average_clustering == 0.0
    assert single.reciprocity == 0.0

    edgeless = evaluation.compute_graph_topology({"A": set(), "B": set()})
    assert edgeless.node_count == 2
    assert edgeless.edge_count == 0
    assert edgeless.reciprocity == 0.0
    assert edgeless.pagerank["A"] == pytest.approx(0.5)


def test_compute_topic_coherence_returns_valid_cv_scores() -> None:
    documents = {
        "/wiki/Neural_network": (
            "Neural networks use layers of artificial neurons, backpropagation, "
            "activation functions, and gradient descent for deep learning."
        ),
        "/wiki/Convolutional_network": (
            "Convolutional neural networks apply learned filters, pooling layers, "
            "and backpropagation for image recognition tasks."
        ),
        "/wiki/Gradient_descent": (
            "Gradient descent optimizes neural network weights using derivatives, "
            "learning rates, and backpropagation."
        ),
        "/wiki/Graph_theory": (
            "Graph theory studies vertices, edges, paths, centrality, "
            "and spanning trees in discrete networks."
        ),
        "/wiki/Shortest_path": (
            "Shortest path algorithms compute minimum weight paths between "
            "vertices and edges in weighted graphs."
        ),
        "/wiki/PageRank": (
            "PageRank measures vertex centrality in directed graphs using "
            "edges, random walks, and stationary distributions."
        ),
    }
    clustering = topic_modeling.cluster_topic_nodes(
        documents,
        num_topics=2,
        no_below=1,
        no_above=1.0,
        random_state=42,
    )
    coherence = evaluation.compute_topic_coherence(clustering, top_n=5)

    assert coherence.num_topics == 2
    assert set(coherence.topic_coherences.keys()) == {0, 1}
    assert not math.isnan(coherence.mean_coherence)
    assert 0.0 <= coherence.mean_coherence <= 1.0
    for score in coherence.topic_coherences.values():
        assert 0.0 <= score <= 1.0

    with pytest.raises(ValueError, match="top_n"):
        evaluation.compute_topic_coherence(clustering, top_n=0)


def test_compute_rouge_score_exact_and_partial_matches() -> None:
    exact = evaluation.compute_rouge_score(
        "Neural networks learn patterns from data.",
        "Neural networks learn patterns from data.",
    )
    assert exact.rouge1.fmeasure == pytest.approx(1.0)
    assert exact.rouge2.fmeasure == pytest.approx(1.0)
    assert exact.rougeL.fmeasure == pytest.approx(1.0)

    partial = evaluation.compute_rouge_score(
        "Neural networks learn from examples.",
        "Neural networks learn patterns from data.",
    )
    assert 0.0 < partial.rouge1.fmeasure < 1.0
    assert partial.rouge1.precision > partial.rouge1.recall
    assert 0.0 < partial.rouge2.fmeasure < 1.0
    assert 0.0 < partial.rougeL.fmeasure < 1.0


def test_evaluate_summaries_macro_averages_and_validates_keys() -> None:
    hypotheses = {
        "doc1": "Neural networks learn patterns from data.",
        "doc2": "Graphs model vertices and edges.",
    }
    references = {
        "doc1": "Neural networks learn patterns from data.",
        "doc2": "Graphs model vertices and edges and paths.",
    }
    result = evaluation.evaluate_summaries(hypotheses, references)

    assert set(result.per_document.keys()) == {"doc1", "doc2"}
    assert result.per_document["doc1"].rouge1.fmeasure == pytest.approx(1.0)
    assert result.mean_rouge1.fmeasure == pytest.approx(
        (result.per_document["doc1"].rouge1.fmeasure + result.per_document["doc2"].rouge1.fmeasure)
        / 2.0
    )

    with pytest.raises(ValueError, match="must not be empty"):
        evaluation.evaluate_summaries({}, {})

    with pytest.raises(ValueError, match="exact same document keys"):
        evaluation.evaluate_summaries({"doc1": "a"}, {"doc2": "b"})
